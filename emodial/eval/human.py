"""Human evaluation: build blinded rating sheets and analyse the filled-in results.

Two tasks (see proposal, Appendix C):
  (a) ratings  - 20 contexts x {b0, b1, etg_nodelta, etg}; empathy / relevance / fluency 1-5
  (b) pairwise - 50 shift-turn contexts, ETG vs B1, "which reply better responds to how
                 the speaker feels now?" -> A / B / tie
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RATING_SYSTEMS = ["b0", "b1", "etg_nodelta", "etg"]


def _ctx(r: dict) -> str:
    who = {"user": "Speaker", "listener": "Listener"}
    return "\n".join(f"{who[role]}: {t}" for role, t in zip(r["context_roles"], r["context"]))


def build_sheets(replies: dict[str, dict[str, dict]], tau: float, out_dir: str | Path, raters: list[str],
                 n_rating: int = 20, n_pairwise: int = 50, seed: int = 0) -> None:
    """replies[system][example_id] -> row with context, p/delta, shift and 'generated'."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = sorted(set.intersection(*(set(replies[s]) for s in RATING_SYSTEMS)))
    ref = replies["etg"]
    shift_ids = [i for i in common if ref[i]["shift"] > tau]
    calm_ids = [i for i in common if ref[i]["shift"] <= tau]
    rating_ids = rng.sample(shift_ids, min(n_rating // 2, len(shift_ids)))
    rating_ids += rng.sample(calm_ids, min(n_rating - len(rating_ids), len(calm_ids)))
    pair_pool = [i for i in shift_ids if i not in rating_ids] or shift_ids
    pair_ids = rng.sample(pair_pool, min(n_pairwise, len(pair_pool)))

    key = {"rating": [], "pairwise": []}
    rating_rows, pair_rows = [], []
    for i in rating_ids:
        systems = RATING_SYSTEMS[:]
        rng.shuffle(systems)
        for slot, s in enumerate(systems):
            item = f"R{len(key['rating']):04d}"
            key["rating"].append({"item": item, "example_id": i, "system": s})
            rating_rows.append({"item": item, "context": _ctx(ref[i]), "reply": replies[s][i]["generated"],
                                "empathy": "", "relevance": "", "fluency": ""})
    for i in pair_ids:
        a, b = ("etg", "b1") if rng.random() < 0.5 else ("b1", "etg")
        item = f"P{len(key['pairwise']):04d}"
        key["pairwise"].append({"item": item, "example_id": i, "A": a, "B": b})
        pair_rows.append({"item": item, "context": _ctx(ref[i]), "reply_A": replies[a][i]["generated"],
                          "reply_B": replies[b][i]["generated"], "choice (A/B/tie)": ""})
    for rater in raters:  # each rater gets their own random item order
        rr, pr = rating_rows[:], pair_rows[:]
        rng.shuffle(rr)
        rng.shuffle(pr)
        pd.DataFrame(rr).to_csv(out / f"ratings_{rater}.csv", index=False)
        pd.DataFrame(pr).to_csv(out / f"pairwise_{rater}.csv", index=False)
    (out / "KEY_do_not_share_with_raters.json").write_text(json.dumps(key, indent=1))
    print(f"wrote sheets for {len(raters)} raters to {out}")


def analyse(sheet_dir: str | Path) -> dict:
    import krippendorff

    d = Path(sheet_dir)
    key = json.loads((d / "KEY_do_not_share_with_raters.json").read_text())
    rkey = pd.DataFrame(key["rating"]).set_index("item")
    pkey = pd.DataFrame(key["pairwise"]).set_index("item")
    results: dict = {}

    ratings = []
    for f in sorted(d.glob("ratings_*.csv")):
        df = pd.read_csv(f)
        df["rater"] = f.stem.split("_", 1)[1]
        ratings.append(df)
    if ratings:
        r = pd.concat(ratings).join(rkey, on="item")
        for dim in ("empathy", "relevance", "fluency"):
            r[dim] = pd.to_numeric(r[dim], errors="coerce")
            results[f"mean_{dim}"] = r.groupby("system")[dim].mean().round(3).to_dict()
            mat = r.pivot_table(index="rater", columns="item", values=dim).to_numpy()
            if mat.shape[0] >= 2 and np.isfinite(mat).sum() > 1:
                results[f"alpha_{dim}"] = float(krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal"))

    pairs = []
    for f in sorted(d.glob("pairwise_*.csv")):
        df = pd.read_csv(f)
        pairs.append(df)
    if pairs:
        p = pd.concat(pairs).join(pkey, on="item")
        choice = p["choice (A/B/tie)"].astype(str).str.strip().str.upper()
        p["winner"] = np.where(choice == "A", p["A"], np.where(choice == "B", p["B"], "tie"))
        judgments = p["winner"].value_counts().to_dict()
        # one vote per item (majority over raters) so repeated raters do not inflate the test
        per_item = p.groupby("item")["winner"].agg(lambda w: w.value_counts().idxmax()
                                                   if (w.value_counts() == w.value_counts().max()).sum() == 1 else "tie")
        etg, b1, ties = int((per_item == "etg").sum()), int((per_item == "b1").sum()), int((per_item == "tie").sum())
        test = binomtest(etg, etg + b1, 0.5) if etg + b1 else None
        results["pairwise"] = {"items_etg_wins": etg, "items_b1_wins": b1, "items_tied": ties,
                               "sign_test_p": float(test.pvalue) if test else None,
                               "all_judgments": {k: int(v) for k, v in judgments.items()}}
    return results
