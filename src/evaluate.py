import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import create_keyword_dataloader
from src.rewards.rule import distinct_2, keyword_score
from src.tokenizer import TextTokenizer
from src.train import evaluate_loss
from src.utils import get_device, load_checkpoint, load_pairs, set_seed

_WORD = re.compile(r"[a-z]+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="name=path pairs; the first one is the baseline for the judge")
    p.add_argument("--test_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--ppl_model", default="gpt2")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--mauve", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
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
                             temperature=args.temperature, top_k=args.top_k)
        for row in out:
            hit = (row == eos).nonzero()
            end = hit[0].item() if len(hit) else len(row)
            stories.append(tok.decode(row[:end]))
            finished.append(bool(len(hit)))
    return stories, finished


@torch.no_grad()
def external_ppl(stories, name, device, batch_size=16):
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


def story_metrics(pairs, stories, finished):
    words = [w for w, _ in pairs]
    kw_exact = [exact_keyword_score(w, s) for w, s in zip(words, stories)]
    return {
        "keyword_exact": sum(kw_exact) / len(kw_exact),
        "keyword_lemma": sum(keyword_score(w, s) for w, s in zip(words, stories)) / len(stories),
        "all_keywords": sum(k == 1.0 for k in kw_exact) / len(kw_exact),
        "avg_words": sum(len(_WORD.findall(s.lower())) for s in stories) / len(stories),
        "eos_rate": sum(finished) / len(finished),
        "distinct_2": sum(distinct_2(s) for s in stories) / len(stories),
        "template_rate": template_rate(stories),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    tok = TextTokenizer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs("validation", args.test_samples)
    refs = [s for _, s in pairs]
    results = {"references": story_metrics(pairs, refs, [True] * len(refs))}
    results["references"]["ext_ppl"] = external_ppl(refs, args.ppl_model, device)
    all_stories = {}

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        model, ckpt = load_checkpoint(path, device)
        cfg = ckpt["config"]
        loader = create_keyword_dataloader(pairs, tok, args.batch_size, cfg["max_src_len"],
                                           cfg["max_tgt_len"], shuffle=False)
        set_seed(args.seed)
        stories, finished = generate_all(model, loader, tok, args, device)
        shuffled, _ = generate_all(model, loader, tok, args, device, shuffle_src=True)

        m = story_metrics(pairs, stories, finished)
        val_loss = evaluate_loss(model, loader, device)
        m["val_ppl"] = math.exp(val_loss)
        m["ext_ppl"] = external_ppl(stories, args.ppl_model, device)
        m["keyword_exact_shuffled_src"] = story_metrics(pairs, shuffled, finished)["keyword_exact"]
        if args.mauve:
            import mauve
            m["mauve"] = mauve.compute_mauve(p_text=refs, q_text=stories, device_id=0 if device.type == "cuda" else -1,
                                             verbose=False).mauve
        results[name] = m
        all_stories[name] = stories
        print(name, json.dumps({k: round(v, 4) for k, v in m.items()}))
        del model

    if args.judge and len(all_stories) > 1:
        from src.judge import PairwiseJudge
        judge = PairwiseJudge(args.judge_model)
        words = [w for w, _ in pairs]
        names = list(all_stories)
        base = names[0]
        results["judge"] = {
            f"{n}_vs_{base}": judge.win_rate(words, all_stories[base], all_stories[n])
            for n in names[1:]
        }
        print("judge", json.dumps(results["judge"]))

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    samples = [{"words": w, "reference": r, **{n: s[i] for n, s in all_stories.items()}}
               for i, (w, r) in enumerate(pairs[:50])]
    (out / "samples.json").write_text(json.dumps(samples, indent=2))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
