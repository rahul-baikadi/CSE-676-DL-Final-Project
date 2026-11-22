import argparse
import copy
import random
import time

import torch
import torch.nn as nn

from src.data import KeywordStoryDataset, filter_complete
from src.rewards import get_reward, keyword_score
from src.tokenizer import TextTokenizer
from src.utils import get_device, load_checkpoint, load_pairs, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="checkpoints/pretrain/best.pt")
    p.add_argument("--reward", choices=["rule", "llm"], default="rule")
    p.add_argument("--reward_model", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--prompt_samples", type=int, default=20_000)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--rollout_size", type=int, default=64)
    p.add_argument("--minibatch_size", type=int, default=16)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=255)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x, mask):
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + 1e-8) * mask


def logprobs_and_values(model, value_head, src_ids, src_mask, tgt_in, resp):
    h_enc = model.encode(src_ids, src_mask)
    h = model.decode(tgt_in, h_enc, src_mask)
    logp = model.lm_head(h).log_softmax(-1).gather(-1, resp.unsqueeze(-1)).squeeze(-1)
    values = value_head(h).squeeze(-1) if value_head is not None else None
    return logp, values


def gae(rewards, values, mask, gamma, lam):
    B, L = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, device=rewards.device)
    for t in reversed(range(L)):
        next_mask = mask[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        next_v = values[:, t + 1] * next_mask if t + 1 < L else 0.0
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * next_mask * last
        adv[:, t] = last
    adv = adv * mask
    return adv, adv + values


@torch.no_grad()
def rollout(policy, value_head, ref, ds, pairs, tok, reward_fn, args, device):
    idx = random.sample(range(len(pairs)), args.rollout_size)
    items = [ds[i] for i in idx]
    words = [pairs[i][0] for i in idx]
    src_ids = torch.stack([it["src_ids"] for it in items]).to(device)
    src_mask = torch.stack([it["src_mask"] for it in items]).to(device)
    eos = tok.tokenizer.eos_token_id

    resp = policy.generate(src_ids, src_mask, eos, max_new_tokens=args.max_new_tokens,
                           temperature=1.0, top_k=None)
    B, L = resp.shape
    is_eos = resp == eos
    finished = is_eos.any(1)
    lengths = torch.where(finished, is_eos.float().argmax(1) + 1, torch.full_like(finished, L, dtype=torch.long))
    mask = (torch.arange(L, device=device)[None] < lengths[:, None]).float()

    stories = [tok.decode(resp[i, : lengths[i]]) for i in range(B)]
    scores = reward_fn(words, stories, finished.tolist()).to(device).float()

    tgt_in = torch.cat([torch.full((B, 1), eos, device=device), resp[:, :-1]], 1)
    old_logp, values = logprobs_and_values(policy, value_head, src_ids, src_mask, tgt_in, resp)
    ref_logp, _ = logprobs_and_values(ref, None, src_ids, src_mask, tgt_in, resp)

    kl = (old_logp - ref_logp) * mask
    rewards = -args.beta * kl
    rewards[torch.arange(B), lengths - 1] += scores

    adv, returns = gae(rewards, values * mask, mask, args.gamma, args.lam)
    adv = masked_whiten(adv, mask)

    stats = {
        "score": scores.mean().item(),
        "keyword": sum(keyword_score(w, s) for w, s in zip(words, stories)) / B,
        "kl": kl.sum(1).mean().item(),
        "len": lengths.float().mean().item(),
        "finished": finished.float().mean().item(),
    }
    batch = dict(src_ids=src_ids, src_mask=src_mask, tgt_in=tgt_in, resp=resp, mask=mask,
                 old_logp=old_logp, old_values=values, adv=adv, returns=returns)
    return batch, stats, stories[0], words[0]


def ppo_update(policy, value_head, optimizer, batch, args):
    B = batch["resp"].size(0)
    params = list(policy.parameters()) + list(value_head.parameters())
    for _ in range(args.ppo_epochs):
        for mb in torch.randperm(B).split(args.minibatch_size):
            b = {k: v[mb] for k, v in batch.items()}
            logp, values = logprobs_and_values(policy, value_head, b["src_ids"], b["src_mask"],
                                               b["tgt_in"], b["resp"])
            ratio = torch.exp(logp - b["old_logp"])
            pg = -torch.min(ratio * b["adv"],
                            ratio.clamp(1 - args.clip, 1 + args.clip) * b["adv"])
            v_clip = b["old_values"] + (values - b["old_values"]).clamp(-args.clip, args.clip)
            vf = 0.5 * torch.max((values - b["returns"]) ** 2, (v_clip - b["returns"]) ** 2)
            loss = masked_mean(pg, b["mask"]) + args.vf_coef * masked_mean(vf, b["mask"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = args.out or f"checkpoints/ppo_{args.reward}"
    tok = TextTokenizer()

    policy, ckpt = load_checkpoint(args.init, device)
    config = ckpt["config"]
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    value_head = nn.Linear(config["d_model"], 1).to(device)
    nn.init.zeros_(value_head.weight)
    nn.init.zeros_(value_head.bias)
    policy.eval()

    reward_kwargs = {"model_name": args.reward_model} if args.reward == "llm" else {}
    reward_fn = get_reward(args.reward, **reward_kwargs)

    pairs = filter_complete(load_pairs("train", args.prompt_samples), tok, config["max_tgt_len"])
    ds = KeywordStoryDataset(pairs, tok, config["max_src_len"], config["max_tgt_len"])
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(value_head.parameters()),
                                  lr=args.lr, weight_decay=0.0)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch, stats, sample, sample_words = rollout(policy, value_head, ref, ds, pairs, tok,
                                                     reward_fn, args, device)
        ppo_update(policy, value_head, optimizer, batch, args)
        print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in stats.items())
              + f" {time.time() - t0:.0f}s")
        if step % 10 == 0:
            print(f"  {sample_words} -> {sample[:250]!r}")
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(f"{out}/policy.pt", policy, config, step=step, reward=args.reward,
                            beta=args.beta, seed=args.seed, value_head=value_head.state_dict())


if __name__ == "__main__":
    main()
