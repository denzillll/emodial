"""Step 6 - automatic evaluation of every system that has replies in outputs/replies.

    python scripts/06_evaluate.py [--skip-ppl]

Writes outputs/results/results.csv (one row per system and seed) and
outputs/results/summary.md (mean +/- std over seeds, plus the human-reply reference).
"""
import _bootstrap  # noqa: F401
import random
import re

import numpy as np
import pandas as pd

from emodial.config import cli, device
from emodial.data.common import read_jsonl
from emodial.eval.metrics import Judge, distinct_n, emotion_agreement, gate_analysis, js_distance, negative_rate
from emodial.generator.model import EmoDial
from emodial.generator.train import perplexity
from emodial.pipeline import Paths, examples, load_emotion_cache, load_tau

args, cfg = cli(__doc__, lambda p: p.add_argument("--skip-ppl", action="store_true"))
P, dev, ec = Paths(cfg), device(), cfg["eval"]
tau = load_tau(P)
judge = Judge(ec["judge"], dev)
cache = load_emotion_cache(P)
test_all = examples(cfg, P, "test", cache) if not args.skip_ppl else []
P.results.mkdir(parents=True, exist_ok=True)

files = sorted(f for f in P.replies.glob("*.jsonl") if not f.stem.endswith("_swap"))
if not files:
    raise SystemExit(f"No replies found in {P.replies}; run 05_generate.py first.")

human_cache: dict = {}


def human_probs(rows):
    missing = [r for r in rows if r["id"] not in human_cache]
    if missing:
        for r, pr in zip(missing, judge.probs([r["response"] for r in missing])):
            human_cache[r["id"]] = pr
    return np.array([human_cache[r["id"]] for r in rows])


def block(rows, gen, ref, prefix=""):
    shift = np.array([r["shift"] > tau for r in rows])
    texts = [r["generated"] for r in rows]
    m = {"n": len(rows), "n_shift": int(shift.sum()),
         "distinct1": distinct_n(texts, 1), "distinct2": distinct_n(texts, 2),
         "avg_len": float(np.mean([len(t.split()) for t in texts])),
         "neg_echo_shift": negative_rate(gen[shift], ec["negative_threshold"]),
         "neg_echo_shift_human": negative_rate(ref[shift], ec["negative_threshold"])}
    m.update(emotion_agreement(gen, ref))
    m.update({f"{k}_shift": v for k, v in emotion_agreement(gen[shift], ref[shift]).items()})
    return {prefix + k: v for k, v in m.items()}


records = []
for f in files:
    variant, seed = re.match(r"(.+)_s(\d+)$", f.stem).groups()
    rows = read_jsonl(f)
    print(f"== {f.stem}: {len(rows)} replies")
    gen, ref = judge.probs([r["generated"] for r in rows]), human_probs(rows)
    rec = {"variant": variant, "seed": int(seed), **block(rows, gen, ref)}
    for src in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        rec.update(block([rows[i] for i in idx], gen[idx], ref[idx], prefix=f"{src}:"))
    if rows[0].get("gate") is not None:
        rec.update(gate_analysis([r["gate"] for r in rows], [r["shift"] for r in rows]))
    swap_f = f.with_name(f.stem + "_swap.jsonl")
    if swap_f.exists():
        sw = read_jsonl(swap_f)
        by_id = {r["id"]: i for i, r in enumerate(rows)}
        idx = [by_id[s["id"]] for s in sw]
        sp = judge.probs([s["generated"] for s in sw])
        rec["swap_top1_changed"] = float((sp.argmax(1) != gen[idx].argmax(1)).mean())
        rec["swap_js_distance"] = float(np.mean([js_distance(a, b) for a, b in zip(sp, gen[idx])]))
        rec["swap_text_changed"] = float(np.mean([s["generated"] != rows[i]["generated"] for s, i in zip(sw, idx)]))
    if not args.skip_ppl:
        model = EmoDial.load(P.model_dir(variant, int(seed))).to(dev)
        rec["ppl"] = perplexity(model, test_all, cfg["generator"], dev)["ppl"]
        for src in sorted({r["source"] for r in test_all}):
            rec[f"{src}:ppl"] = perplexity(model, [r for r in test_all if r["source"] == src], cfg["generator"], dev)["ppl"]
        if variant != "b0":  # perplexity with emotions shuffled across examples
            donors = test_all[:]
            random.Random(0).shuffle(donors)
            swapped = [{**r, "p": d["p"], "delta": d["delta"]} for r, d in zip(test_all, donors)]
            rec["ppl_swapped_emotion"] = perplexity(model, swapped, cfg["generator"], dev)["ppl"]
        del model
    records.append(rec)

df = pd.DataFrame(records).sort_values(["variant", "seed"])
df.to_csv(P.results / "results.csv", index=False)

# summary: mean +/- std over seeds for the headline metrics
headline = ["ppl", "ppl_swapped_emotion", "distinct1", "distinct2", "avg_len", "neg_echo_shift",
            "emotion_top1_agreement", "emotion_js_distance", "emotion_top1_agreement_shift",
            "gate_mean", "gate_spearman_vs_shift", "swap_top1_changed"]
headline = [h for h in headline if h in df.columns]
order = {v: i for i, v in enumerate(["b0", "b1", "etg_nodelta", "etg_nogate", "etg"])}
lines = ["| system | " + " | ".join(headline) + " |", "|---" * (len(headline) + 1) + "|"]
for v, g in sorted(df.groupby("variant"), key=lambda x: order.get(x[0], 99)):
    cells = []
    for h in headline:
        vals = g[h].dropna()
        cells.append("" if vals.empty else f"{vals.mean():.3f}" + (f" ± {vals.std():.3f}" if len(vals) > 1 else ""))
    lines.append(f"| {v} | " + " | ".join(cells) + " |")
hum = df.iloc[0]
lines.append(f"\nHuman reference replies: negative-echo rate on shift turns = {hum['neg_echo_shift_human']:.3f} "
             f"(tau = {tau:.3f}, shift turns = {hum['n_shift']})")
(P.results / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
