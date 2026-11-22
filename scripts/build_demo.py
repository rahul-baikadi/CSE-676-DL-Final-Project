"""Build a self-contained, offline story comparison page from evaluation results.

Usage: python scripts/build_demo.py results/ [results/demo.html]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards.rule import RuleReward, _WORD, distinct_2, keyword_score, lemmas  # noqa: E402

METRICS = ["keyword_exact", "all_keywords", "ext_ppl", "distinct_2", "keyword_crammed", "rule_reward"]


def story_record(words, story, finished, reward):
    targets = [lemmas(w) for w in words]
    tokens = set(_WORD.findall(story.lower()))
    hits = sorted(t for t in tokens if any(lemmas(t) & target for target in targets))
    used = [w for w, target in zip(words, targets) if any(lemmas(t) & target for t in tokens)]
    record = {
        "text": story,
        "hits": hits,
        "missing": [w for w in words if w not in used],
        "coverage": keyword_score(words, story),
        "words": len(_WORD.findall(story.lower())),
        "repeat": 1 - distinct_2(story),
    }
    if finished is not None:
        record["finished"] = finished
        record["reward"] = reward._score(words, story, finished)
    return record


def build_data(results):
    samples = json.loads((results / "samples.json").read_text())
    metrics = json.loads((results / "metrics.json").read_text())
    reward = RuleReward()
    methods = [k for k in samples[0] if k not in ("words", "finished")]
    prompts = []
    for row in samples:
        flags = row.get("finished", {})
        stories = {}
        for m in methods:
            done = True if m == "reference" else flags.get(m)
            stories[m] = story_record(row["words"], row[m], done, reward)
        prompts.append({"words": row["words"], "stories": stories})

    summary = {k: {c: v[c] for c in METRICS if c in v} for k, v in metrics.items()
               if isinstance(v, dict) and "keyword_exact" in v}
    if "references" in summary:
        summary["reference"] = summary.pop("references")   # samples.json uses the singular key
    ppo =[re.fullmatch(r"ppo_(rule|llm)_b(\d+)_s(\d+)", m) for m in methods]
    ppo = [p for p in ppo if p]
    best_of = sorted({int(x.group(1)) for x in (re.search(r"_bo(\d+)$", m) for m in methods) if x})
    return {
        "prompts": prompts,
        "summary": summary,
        "judge": metrics.get("judge", {}),
        "base": next(m for m in methods if m != "reference" and "_bo" not in m and not m.startswith("ppo_")),
        "bestOf": best_of,
        "ppo": {r: {"betas": sorted({p.group(2) for p in ppo if p.group(1) == r},
                                    key=lambda t: 0.0 if t == "0" else float("0." + t[1:])),
                    "seeds": sorted({p.group(3) for p in ppo if p.group(1) == r}, key=int)}
                for r in sorted({p.group(1) for p in ppo}, reverse=True)},
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Story Comparison Demo</title>
<style>
:root { --blue:#005BBB; --ink:#222; --muted:#5f6670; --line:#d8e2ee; --tint:#f1f6fb; --hit:#ffe08a; --miss:#b3261e; --ok:#17693a; }
* { box-sizing:border-box; }
body { margin:0; background:#fff; color:var(--ink); font-family:Arial, Helvetica, sans-serif; }
header { background:var(--blue); color:#fff; padding:16px 24px; }
h1 { margin:0; font:600 26px Cambria, Georgia, serif; }
header p { margin:4px 0 0; font-size:14px; opacity:.92; }
main { padding:16px 24px 28px; max-width:1800px; margin:auto; }
.bar { display:flex; flex-wrap:wrap; gap:12px 22px; align-items:center; margin-bottom:14px; }
.group { display:flex; align-items:center; gap:6px; }
.group > span { color:var(--muted); font-size:14px; }
button { font:inherit; font-size:15px; min-height:38px; padding:6px 12px; border:1px solid var(--line); background:#fff; color:var(--blue); border-radius:7px; cursor:pointer; }
button:hover:not(:disabled) { background:var(--tint); }
button:focus-visible { outline:3px solid #7bb2ea; outline-offset:1px; }
button[aria-pressed=true] { background:var(--blue); border-color:var(--blue); color:#fff; }
button:disabled { color:#a4abb4; cursor:not-allowed; }
.keywords { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:14px; font-size:18px; }
.chip { background:var(--hit); padding:4px 11px; border-radius:999px; font-weight:700; }
.counter { color:var(--muted); font-size:14px; margin-left:6px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(285px,1fr)); gap:14px; }
.card { border:1px solid var(--line); border-radius:10px; overflow:hidden; display:flex; flex-direction:column; min-width:0; }
.card h2 { margin:0; padding:10px 13px; background:var(--tint); color:var(--blue); font-size:17px; }
.stats { display:flex; flex-wrap:wrap; gap:4px 13px; padding:8px 13px; font-size:13.5px; color:var(--muted); border-bottom:1px solid var(--line); }
.stats b { color:var(--ink); }
.missing { padding:7px 13px 0; color:var(--miss); font-size:13.5px; font-weight:700; }
.complete { padding:7px 13px 0; color:var(--ok); font-size:13.5px; font-weight:700; }
.story { margin:0; padding:10px 13px 14px; font-size:16px; line-height:1.48; white-space:pre-wrap; overflow-wrap:anywhere; max-height:48vh; overflow:auto; }
mark { background:var(--hit); color:inherit; border-radius:3px; padding:0 2px; font-weight:700; }
section { margin-top:20px; }
h3 { margin:0 0 8px; color:var(--blue); font-size:18px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; min-width:760px; font-size:14.5px; }
th,td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child,td:first-child { text-align:left; font-weight:700; }
th { background:var(--tint); color:var(--blue); }
.note { margin:8px 0 0; color:var(--muted); font-size:13px; }
@media (max-width:640px){ header,main{padding-left:16px;padding-right:16px;} h1{font-size:22px;} }
</style>
</head>
<body>
<header><h1>Learning vs. Sampling</h1><p>Same keywords, stories from each method</p></header>
<main>
  <div class="bar">
    <div class="group"><button id="prev">← Previous</button><button id="next">Next →</button><button id="random">Random</button></div>
    <div class="group" id="reward"></div>
    <div class="group" id="beta"></div>
    <div class="group" id="seed"></div>
    <div class="group"><button id="reference" aria-pressed="false">Show reference</button></div>
  </div>
  <div class="keywords" id="keywords"></div>
  <div class="grid" id="cards"></div>
  <section><h3>Across all test prompts</h3><div class="scroll"><table id="summary"></table></div>
  <p class="note" id="judge"></p>
  <p class="note">Prompts are in test-set order and were not hand-picked. Keyword matches include word forms, such as "jump" and "jumped". Use ← and → to move between prompts.</p></section>
</main>
<script id="demo-data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("demo-data").textContent);
const rewards = Object.keys(D.ppo);
const S = { i:0, reward:rewards[0] || "rule", beta:null, seed:null, reference:false };
const $ = id => document.getElementById(id);
const betaLabel = t => t === "0" ? "0" : "0." + t.slice(1);
const fmt = (v, d=3) => v === undefined || v === null || Number.isNaN(v) ? "–" : Number(v).toFixed(d);
const pct = v => v === undefined || v === null ? "–" : (100 * v).toFixed(1) + "%";
function el(tag, props={}, kids=[]) { const n = Object.assign(document.createElement(tag), props); n.append(...kids); return n; }

function syncPpo() {
  const c = D.ppo[S.reward]; if (!c) return;
  if (!c.betas.includes(S.beta)) S.beta = c.betas.includes("005") ? "005" : c.betas[0];
  if (!c.seeds.includes(S.seed)) S.seed = c.seeds[0];
}
function ppoName() { return `ppo_${S.reward}_b${S.beta}_s${S.seed}`; }
function boName(n) { return `${D.base}${S.reward === "llm" ? "_llm" : ""}_bo${n}`; }
function methods() {
  const list = [{ key:D.base, title:"Supervised" }];
  D.bestOf.forEach(n => list.push({ key:boName(n), title:`Best-of-${n}` }));
  if (D.ppo[S.reward]) list.push({ key:ppoName(), title:`PPO (β = ${betaLabel(S.beta)})` });
  if (S.reference) list.push({ key:"reference", title:"Reference story" });
  return list;
}
function choice(host, label, values, current, show, set) {
  host.replaceChildren(el("span", { textContent:label }));
  values.forEach(v => {
    const b = el("button", { textContent:show(v), onclick:() => { set(v); syncPpo(); render(); } });
    b.setAttribute("aria-pressed", String(v === current));
    host.append(b);
  });
}
function highlighted(r) {
  const pre = el("p", { className:"story" }), hits = new Set(r.hits);
  r.text.split(/([A-Za-z]+)/).forEach(part => {
    pre.append(hits.has(part.toLowerCase()) ? el("mark", { textContent:part }) : document.createTextNode(part));
  });
  return pre;
}
function card(m, prompt) {
  const r = prompt.stories[m.key];
  if (!r) return el("article", { className:"card" }, [el("h2", { textContent:m.title }), el("p", { className:"story", textContent:"Not available in these results." })]);
  const total = prompt.words.length, used = total - r.missing.length;
  const stat = (k, v) => el("span", {}, [k + " ", el("b", { textContent:v })]);
  const stats = [stat("Keywords", `${used}/${total}`), stat("Words", r.words), stat("Repeat", pct(r.repeat))];
  if (r.reward !== undefined) stats.push(stat("Reward", fmt(r.reward, 2)));
  if (r.finished === false) stats.push(el("span", { textContent:"Cut off" }));
  const status = r.missing.length ? el("div", { className:"missing", textContent:"Missing: " + r.missing.join(", ") })
                                   : el("div", { className:"complete", textContent:"All keywords used" });
  return el("article", { className:"card" }, [el("h2", { textContent:m.title }), el("div", { className:"stats" }, stats), status, highlighted(r)]);
}
function summary(list) {
  const heads = ["Method", "Coverage", "All 3 keywords", "GPT-2 perplexity", "Diversity", "Crammed", "Rule reward", "Judge win vs. supervised"];
  const head = el("thead", {}, [el("tr", {}, heads.map(h => el("th", { textContent:h, scope:"col" })))]);
  const body = el("tbody", {}, list.filter(m => D.summary[m.key]).map(m => {
    const s = D.summary[m.key], j = D.judge[`${m.key}_vs_${D.base}`];
    const cells = [m.title, pct(s.keyword_exact), pct(s.all_keywords), fmt(s.ext_ppl, 1), fmt(s.distinct_2), pct(s.keyword_crammed), fmt(s.rule_reward), j ? pct(j.win) : "–"];
    return el("tr", {}, cells.map(c => el("td", { textContent:c })));
  }));
  $("summary").replaceChildren(head, body);
  const top = D.bestOf.length ? Math.max(...D.bestOf) : null, j = top && D.ppo[S.reward] ? D.judge[`${ppoName()}_vs_${boName(top)}`] : null;
  $("judge").textContent = j ? `LLM judge, this PPO run vs. Best-of-${top}: ${pct(j.win)} win, ${pct(j.tie)} tie, ${pct(j.loss)} loss.` : "";
}
function render() {
  const prompt = D.prompts[S.i], c = D.ppo[S.reward], list = methods();
  choice($("reward"), "Reward", rewards, S.reward, v => v === "rule" ? "Rule" : "LLM", v => S.reward = v);
  choice($("beta"), "β", c ? c.betas : [], S.beta, betaLabel, v => S.beta = v);
  choice($("seed"), "Seed", c ? c.seeds : [], S.seed, v => v, v => S.seed = v);
  $("reference").setAttribute("aria-pressed", String(S.reference));
  $("keywords").replaceChildren(...prompt.words.map(w => el("span", { className:"chip", textContent:w })), el("span", { className:"counter", textContent:`Prompt ${S.i + 1} of ${D.prompts.length}` }));
  $("cards").replaceChildren(...list.map(m => card(m, prompt)));
  summary(list);
}
function move(step) { S.i = (S.i + step + D.prompts.length) % D.prompts.length; render(); }
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("random").onclick = () => { S.i = Math.floor(Math.random() * D.prompts.length); render(); };
$("reference").onclick = () => { S.reference = !S.reference; render(); };
addEventListener("keydown", e => { if (e.key === "ArrowLeft") move(-1); if (e.key === "ArrowRight") move(1); });
syncPpo(); render();
</script>
</body>
</html>
"""


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "demo.html"
    data = json.dumps(build_data(results), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    out.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
