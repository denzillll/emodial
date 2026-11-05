"""Step 3 - run the frozen recogniser on every user turn of ED + ESConv, cache p_t,
report Delta statistics, choose tau, and sanity-check predictions against ED labels.

    python scripts/03_cache_emotions.py
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

from emodial.config import cli, device
from emodial.data.common import iter_examples, user_turn_key
from emodial.emotions import ED_COARSE, EMOTIONS
from emodial.generator.data import attach_emotions
from emodial.pipeline import Paths, load_dialogues
from emodial.recogniser import model as R

args, cfg = cli(__doc__)
P, rc, dev = Paths(cfg), cfg["recogniser"], device()
tok, model = R.load_classifier(str(P.recogniser))

dialogues = {s: load_dialogues(P, cfg["data"]["sources"], s) for s in ("train", "valid", "test")}
rows, keys = [], []
for split_dialogues in dialogues.values():
    for d in split_dialogues:
        for i, t in enumerate(d["turns"]):
            if t["role"] == "user":
                rows.append({"prev": d["turns"][i - 1]["text"] if i > 0 else "", "text": t["text"]})
                keys.append(user_turn_key(d["id"], i))
print(f"scoring {len(rows)} user turns on {dev} ...")
probs = R.predict_proba(model, tok, rows, rc["eval_batch_size"], rc["max_len"], dev, rc["fp16"])
cache = {k: [round(float(x), 5) for x in p] for k, p in zip(keys, probs)}
P.cache.mkdir(parents=True, exist_ok=True)
P.emotion_cache.write_text(json.dumps(cache))

# ---- Delta statistics per source / split --------------------------------------------
stats = {}
for split, ds in dialogues.items():
    ex = attach_emotions(list(iter_examples(ds, cfg["data"]["context_turns"])), cache)
    for src in cfg["data"]["sources"]:
        s = np.array([e["shift"] for e in ex if e["source"] == src])
        if len(s) == 0:
            continue
        nz = s[s > 0]
        stats[f"{src}_{split}"] = {
            "examples": int(len(s)), "with_previous_user_turn": float((s > 0).mean()),
            "shift_quantiles_nonzero": {q: float(np.quantile(nz, q)) for q in (0.25, 0.5, 0.75, 0.9)} if len(nz) else {}}
valid_ex = attach_emotions(list(iter_examples(dialogues["valid"], cfg["data"]["context_turns"])), cache)
nz = np.array([e["shift"] for e in valid_ex if e["shift"] > 0])
tau = float(np.quantile(nz, cfg["eval"]["shift_quantile"])) if len(nz) else 0.5
for split, ds in dialogues.items():
    ex = attach_emotions(list(iter_examples(ds, cfg["data"]["context_turns"])), cache)
    for src in cfg["data"]["sources"]:
        n = sum(1 for e in ex if e["source"] == src and e["shift"] > tau)
        if f"{src}_{split}" in stats:
            stats[f"{src}_{split}"]["shift_turns_above_tau"] = n

# ---- sanity check: first ED speaker turn vs (coarse) conversation label ------------------
hits = hits_nn = total = 0
non_neutral = [i for i, e in enumerate(EMOTIONS) if e != "neutral"]
for d in dialogues["test"]:
    if d["source"] != "ed" or d["label"] not in ED_COARSE:
        continue
    p = np.array(cache[user_turn_key(d["id"], 0)])
    total += 1
    hits += EMOTIONS[int(np.argmax(p))] == ED_COARSE[d["label"]]
    hits_nn += EMOTIONS[non_neutral[int(np.argmax(p[non_neutral]))]] == ED_COARSE[d["label"]]
stats["tau"] = tau
stats["ed_sanity_check"] = {"n": total,
                            "coarse_label_agreement": hits / total if total else None,
                            "coarse_label_agreement_ignoring_neutral": hits_nn / total if total else None}
P.delta_stats.write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
