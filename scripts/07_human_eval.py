"""Step 7 - human evaluation sheets and analysis.

    python scripts/07_human_eval.py make      # blinded CSV sheets, one pair per rater
    python scripts/07_human_eval.py analyse   # after raters fill in their CSVs

Uses replies from seed `seed` (default config seed) of b0, b1, etg_nodelta and etg.
"""
import _bootstrap  # noqa: F401
import json

from emodial.config import cli
from emodial.data.common import read_jsonl
from emodial.eval.human import RATING_SYSTEMS, analyse, build_sheets
from emodial.pipeline import Paths, load_tau

args, cfg = cli(__doc__, lambda p: p.add_argument("action", choices=["make", "analyse"]))
P, h = Paths(cfg), cfg["human"]
if args.action == "make":
    replies = {s: {r["id"]: r for r in read_jsonl(P.replies / f"{s}_s{cfg['seed']}.jsonl")} for s in RATING_SYSTEMS}
    build_sheets(replies, load_tau(P), P.human, h["raters"], h["n_rating"], h["n_pairwise"], cfg["seed"])
else:
    res = analyse(P.human)
    (P.results).mkdir(parents=True, exist_ok=True)
    (P.results / "human_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
