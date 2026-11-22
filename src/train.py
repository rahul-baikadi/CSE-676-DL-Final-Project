import argparse
import math
import time

import torch

from src.data import create_keyword_dataloader
from src.model import Seq2SeqTransformer
from src.tokenizer import TextTokenizer
from src.utils import get_device, load_pairs, save_checkpoint, set_seed, to_device

SAMPLE_PROMPTS = [["dog", "ball", "happy"], ["rain", "jump", "friend"], ["cake", "sad", "mom"]]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_samples", type=int, default=100_000)
    p.add_argument("--val_samples", type=int, default=2_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_enc_layers", type=int, default=2)
    p.add_argument("--n_dec_layers", type=int, default=2)
    p.add_argument("--max_src_len", type=int, default=32)
    p.add_argument("--max_tgt_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out", default="checkpoints/pretrain")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit_one_batch", action="store_true")
    return p.parse_args()


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return f


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, n_tokens = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        _, loss = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        total += loss.item() * n
        n_tokens += n
    model.train()
    return total / n_tokens


@torch.no_grad()
def print_samples(model, tok, device, max_src_len):
    for words in SAMPLE_PROMPTS:
        src = tok.tokenizer.encode(", ".join(words))[:max_src_len]
        src_ids = torch.tensor([src], device=device)
        src_mask = torch.ones_like(src_ids)
        out = model.generate(src_ids, src_mask, tok.tokenizer.eos_token_id,
                             max_new_tokens=80, temperature=0.8, top_k=50)
        print(f"  {words} -> {tok.decode(out[0])[:300]!r}")
    model.train()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    tok = TextTokenizer()

    train_pairs = load_pairs("train", args.train_samples)
    val_pairs = load_pairs("validation", args.val_samples)
    loader_kw = dict(max_src_len=args.max_src_len, max_length=args.max_tgt_len)
    train_loader = create_keyword_dataloader(train_pairs, tok, args.batch_size, shuffle=True, **loader_kw)
    val_loader = create_keyword_dataloader(val_pairs, tok, args.batch_size, shuffle=False, **loader_kw)

    config = dict(vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                  n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
                  max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len, dropout=args.dropout)
    model = Seq2SeqTransformer(**config).to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(train_pairs)} val={len(val_pairs)}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    total_steps = args.max_steps or args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args.warmup_steps, total_steps))

    if args.overfit_one_batch:
        batch = to_device(next(iter(train_loader)), device)
        for step in range(300):
            _, loss = model(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}")
        return

    best_val, step, t0 = float("inf"), 0, time.time()
    model.train()
    while step < total_steps:
        for batch in train_loader:
            batch = to_device(batch, device)
            _, loss = model(**batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate_loss(model, val_loader, device)
                print(f"[eval] step {step} val_loss {val_loss:.4f} ppl {math.exp(val_loss):.2f}")
                print_samples(model, tok, device, args.max_src_len)
                save_checkpoint(f"{args.out}/last.pt", model, config, step=step, val_loss=val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(f"{args.out}/best.pt", model, config, step=step, val_loss=val_loss)

            if step >= total_steps:
                break

    print(f"done. best val_loss {best_val:.4f} ppl {math.exp(best_val):.2f}")


if __name__ == "__main__":
    main()
