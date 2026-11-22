import argparse
import gc
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import create_keyword_dataloader
from src.rewards import get_reward
from src.rewards.rule import RuleReward, distinct_2, keyword_score
from src.tokenizer import TextTokenizer
from src.train import evaluate_loss
from src.utils import get_device, load_checkpoint, load_test_pairs, set_seed

_WORD = re.compile(r"[a-z]+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="name=path pairs; the first one is the baseline for the judge")
    p.add_argument("--test_samples", type=int, default=500)
    p.add_argument("--val_samples", type=int, default=2000,
                   help="validation entries used for checkpoint selection; excluded from test")
    p.add_argument("--test_pool", type=int, default=2000)
    p.add_argument("--max_tgt_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=255)
    # Defaults match the distribution PPO samples from and optimizes.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0, help="0 disables top-k")
    p.add_argument("--ppl_model", default="gpt2")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mauve", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--best_of", type=int, nargs="*", default=[],
                   help="best-of-N baselines for the FIRST checkpoint, e.g. --best_of 4 16")
    p.add_argument("--best_of_rewards", nargs="+", choices=["rule", "llm"], default=["rule"],
                   help="rewards used to select best-of-N samples")
    p.add_argument("--reward_model", default="microsoft/Phi-3.5-mini-instruct")
    return p.parse_args()


def exact_keyword_score(words, story):
    s = story.lower()
    return sum(bool(re.search(rf"\b{re.escape(w.lower())}\b", s)) for w in words) / len(words)


@torch.no_grad()
def generate_all(model, loader, tok, args, device, shuffle_src=False):
    stories, finished = [], []
    eos = tok.tokenizer.eos_token_id
    for batch in loader:
        src_ids, src_mask = batch["src_ids"].to(device), batch["src_mask"].to(device)
        if shuffle_src:
            perm = torch.roll(torch.arange(src_ids.size(0)), 1)
            src_ids, src_mask = src_ids[perm], src_mask[perm]
        out = model.generate(src_ids, src_mask, eos, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k or None)
        for row in out:
            hit = (row == eos).nonzero()
            end = hit[0].item() if len(hit) else len(row)
            stories.append(tok.decode(row[:end]))
            finished.append(bool(len(hit)))
    return stories, finished


@torch.no_grad()
def generate_best_of_n(model, loader, pairs, tok, args, device, n, reward_fn):
    """Sample n stories per prompt and keep the one with the highest reward.
    Same reward PPO optimizes, so this is the inference-time-compute baseline."""
    stories, finished = [], []
    eos = tok.tokenizer.eos_token_id
    offset = 0
    for batch in loader:
        src_ids, src_mask = batch["src_ids"].to(device), batch["src_mask"].to(device)
        B = src_ids.size(0)
        words = [pairs[offset + i][0] for i in range(B)]
        offset += B
        out = model.generate(src_ids.repeat_interleave(n, 0), src_mask.repeat_interleave(n, 0), eos,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k or None)
        cands, fins = [], []
        for row in out:
            hit = (row == eos).nonzero()
            end = hit[0].item() if len(hit) else len(row)
            cands.append(tok.decode(row[:end]))
            fins.append(bool(len(hit)))
        rep_words = [w for w in words for _ in range(n)]
        best = reward_fn(rep_words, cands, fins).view(B, n).argmax(1)
        for i in range(B):
            j = i * n + best[i].item()
            stories.append(cands[j])
            finished.append(fins[j])
    return stories, finished


@torch.no_grad()
def external_ppl(stories, name, device, batch_size=16):
    stories = [s for s in stories if s.strip()]   # empty generations break batching
    if not stories:
        return float("nan")
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    total, n_tokens = 0.0, 0
    for i in range(0, len(stories), batch_size):
        enc = tok(stories[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        n = (labels[:, 1:] != -100).sum().item()
        if n == 0:
            continue
        total += lm(**enc, labels=labels).loss.item() * n
        n_tokens += n
    del lm
    return math.exp(total / max(n_tokens, 1))


def template_rate(stories, n=5):
    starts = [" ".join(_WORD.findall(s.lower())[:n]) for s in stories]
    return Counter(starts).most_common(1)[0][1] / len(stories)


def keyword_usage(words, story, window=6):
    """(keyword tokens / story tokens, whether all keywords fit inside a short window)."""
    toks = _WORD.findall(story.lower())
    targets = [w.lower() for w in words]
    hits = [(i, t) for i, t in enumerate(toks) if t in targets]
    density = len(hits) / max(len(toks), 1)
    needed = set(targets)
    if {t for _, t in hits} != needed:
        return density, False
    for start, _ in hits:
        if needed <= {t for i, t in hits if start <= i < start + window}:
            return density, True
    return density, False


def bootstrap_ci(values, n_boot, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    lo, hi = np.percentile(values[idx].mean(1), [2.5, 97.5])
    return [float(lo), float(hi)]


def story_metrics(pairs, stories, finished, n_boot=1000, seed=0):
    words = [w for w, _ in pairs]
    kw_exact = [exact_keyword_score(w, s) for w, s in zip(words, stories)]
    all_kw = [k == 1.0 for k in kw_exact]
    usage = [keyword_usage(w, s) for w, s in zip(words, stories)]
    rule = RuleReward()(words, stories, finished).tolist()
    return {
        "keyword_exact": sum(kw_exact) / len(kw_exact),
        "keyword_exact_ci": bootstrap_ci(kw_exact, n_boot, seed),
        "keyword_lemma": sum(keyword_score(w, s) for w, s in zip(words, stories)) / len(stories),
        "all_keywords": sum(all_kw) / len(all_kw),
        "all_keywords_ci": bootstrap_ci(all_kw, n_boot, seed),
        "rule_reward": sum(rule) / len(rule),
        "rule_reward_ci": bootstrap_ci(rule, n_boot, seed),
        "keyword_density": sum(d for d, _ in usage) / len(usage),
        "keyword_crammed": sum(c for _, c in usage) / len(usage),
        "avg_words": sum(len(_WORD.findall(s.lower())) for s in stories) / len(stories),
        "eos_rate": sum(finished) / len(finished),
        "distinct_2": sum(distinct_2(s) for s in stories) / len(stories),
        "template_rate": template_rate(stories),
    }


def add_mauve(m, refs, stories, device):
    import mauve
    m["mauve"] = mauve.compute_mauve(p_text=refs, q_text=stories,
                                     device_id=0 if device.type == "cuda" else -1,
                                     verbose=False).mauve


def judge_pairs(names):
    """Each method vs the baseline, plus each PPO run vs best-of-N with the same reward."""
    base = names[0]
    out = [(base, n) for n in names[1:]]
    for n in names:
        if n.startswith("ppo_llm"):
            out += [(b, n) for b in names if "_llm_bo" in b]
        elif n.startswith("ppo_"):
            out += [(b, n) for b in names if "_bo" in b and "_llm_bo" not in b]
    return out


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    tok = TextTokenizer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save():
        (out / "metrics.json").write_text(json.dumps(results, indent=2))

    pairs = load_test_pairs(tok, args.max_tgt_len, args.test_samples,
                            args.val_samples, args.test_pool)
    refs = [s for _, s in pairs]
    metrics = lambda stories, finished: story_metrics(pairs, stories, finished, args.n_boot, args.seed)
    results = {"references": metrics(refs, [True] * len(refs))}
    results["references"]["ext_ppl"] = external_ppl(refs, args.ppl_model, device)
    save()
    all_stories, all_finished = {}, {}

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        model, ckpt = load_checkpoint(path, device)
        cfg = ckpt["config"]
        loader = create_keyword_dataloader(pairs, tok, args.batch_size, cfg["max_src_len"],
                                           cfg["max_tgt_len"], shuffle=False)
        set_seed(args.seed)
        stories, finished = generate_all(model, loader, tok, args, device)
        shuffled, _ = generate_all(model, loader, tok, args, device, shuffle_src=True)

        m = metrics(stories, finished)
        val_loss = evaluate_loss(model, loader, device)
        m["val_ppl"] = math.exp(val_loss)
        m["ext_ppl"] = external_ppl(stories, args.ppl_model, device)
        m["keyword_exact_shuffled_src"] = float(np.mean(
            [exact_keyword_score(w, s) for (w, _), s in zip(pairs, shuffled)]))
        if args.mauve:
            add_mauve(m, refs, stories, device)
        results[name] = m
        all_stories[name] = stories
        all_finished[name] = finished
        save()
        print(name, json.dumps(m))

        if spec == args.checkpoints[0] and args.best_of:
            for reward_name in args.best_of_rewards:
                kwargs = {"model_name": args.reward_model} if reward_name == "llm" else {}
                reward_fn = get_reward(reward_name, **kwargs)
                tag = "" if reward_name == "rule" else "_llm"
                for n in args.best_of:
                    bo_name = f"{name}{tag}_bo{n}"
                    set_seed(args.seed)
                    bo_stories, bo_fin = generate_best_of_n(model, loader, pairs, tok, args, device,
                                                            n, reward_fn)
                    bm = metrics(bo_stories, bo_fin)
                    bm["ext_ppl"] = external_ppl(bo_stories, args.ppl_model, device)
                    bm["samples_per_prompt"] = n
                    bm["selection_reward"] = reward_name
                    if args.mauve:
                        add_mauve(bm, refs, bo_stories, device)
                    results[bo_name] = bm
                    all_stories[bo_name] = bo_stories
                    all_finished[bo_name] = bo_fin
                    save()
                    print(bo_name, json.dumps(bm))
                del reward_fn
                free_memory()
        del model
        free_memory()

    if args.judge and len(all_stories) > 1:
        from src.judge import PairwiseJudge
        judge = PairwiseJudge(args.judge_model)
        words = [w for w, _ in pairs]
        results["judge"] = {
            f"{cand}_vs_{base}": judge.win_rate(words, all_stories[base], all_stories[cand])
            for base, cand in judge_pairs(list(all_stories))
        }
        save()
        print("judge", json.dumps(results["judge"]))

    save()
    samples = [{"words": w, "reference": r, **{n: s[i] for n, s in all_stories.items()},
                "finished": {n: f[i] for n, f in all_finished.items()}}
               for i, (w, r) in enumerate(pairs[:50])]
    (out / "samples.json").write_text(json.dumps(samples, indent=2))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
