"""Keyword coverage vs fluency (GPT-2 perplexity) for every method in metrics.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

metrics = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

rows = [(k, v) for k, v in metrics.items() if isinstance(v, dict) and "keyword_exact" in v]
fig, ax = plt.subplots(figsize=(6, 4.5))
for name, m in rows:
    marker = "*" if name == "references" else ("s" if "_bo" in name else ("^" if name.startswith("ppo") else "o"))
    ax.scatter(m["ext_ppl"], m["keyword_exact"], marker=marker, s=90)
    ax.annotate(name, (m["ext_ppl"], m["keyword_exact"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("GPT-2 perplexity of generated stories (lower = more fluent)")
ax.set_ylabel("Keyword coverage (exact match)")
ax.set_title("Constraint satisfaction vs fluency")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "coverage_vs_ppl.png", dpi=200)

cols = ["keyword_exact", "all_keywords", "ext_ppl", "distinct_2", "template_rate", "eos_rate", "avg_words"]
lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
for name, m in rows:
    lines.append(f"| {name} | " + " | ".join(f"{m.get(c, float('nan')):.3f}" for c in cols) + " |")
if "judge" in metrics:
    lines += ["", "| judge (vs sft) | win | tie | loss |", "|---|---|---|---|"]
    for k, v in metrics["judge"].items():
        lines.append(f"| {k} | {v['win']:.2f} | {v['tie']:.2f} | {v['loss']:.2f} |")
(out / "results_table.md").write_text("\n".join(lines))
print("\n".join(lines))
