"""Keyword coverage vs fluency (GPT-2 perplexity), averaged across PPO seeds."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = re.compile(r"_s\d+(?=$|_vs_)")

metrics = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

groups = defaultdict(list)
for k, v in metrics.items():
    if isinstance(v, dict) and "keyword_exact" in v:
        groups[SEED.sub("", k)].append(v)


def stat(runs, col):
    vals = [r[col] for r in runs if col in r]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def cell(runs, col):
    mean, sd = stat(runs, col)
    return f"{mean:.3f} ± {sd:.3f}" if len(runs) > 1 else f"{mean:.3f}"


fig, ax = plt.subplots(figsize=(6, 4.5))
for name, runs in groups.items():
    marker = "*" if name == "references" else ("s" if "_bo" in name else ("^" if name.startswith("ppo") else "o"))
    x, xerr = stat(runs, "ext_ppl")
    y, yerr = stat(runs, "keyword_exact")
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, markersize=8, capsize=2)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("GPT-2 perplexity of generated stories (lower = more fluent)")
ax.set_ylabel("Keyword coverage (exact match)")
ax.set_title("Constraint satisfaction vs fluency (mean ± SD across seeds)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "coverage_vs_ppl.png", dpi=200)

cols = ["keyword_exact", "all_keywords", "rule_reward", "ext_ppl", "distinct_2", "template_rate",
        "keyword_crammed", "keyword_density", "eos_rate", "avg_words"]
lines = ["| method | seeds | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
for name, runs in groups.items():
    lines.append(f"| {name} | {len(runs)} | " + " | ".join(cell(runs, c) for c in cols) + " |")

if "judge" in metrics:
    judged = defaultdict(list)
    for k, v in metrics["judge"].items():
        judged[SEED.sub("", k)].append(v)
    lines += ["", "| judge comparison | seeds | win | tie | loss |", "|---|---|---|---|---|"]
    for k, runs in judged.items():
        lines.append(f"| {k} | {len(runs)} | " + " | ".join(cell(runs, c) for c in ("win", "tie", "loss")) + " |")

(out / "results_table.md").write_text("\n".join(lines))
print("\n".join(lines))
