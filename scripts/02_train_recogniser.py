"""Step 2 - fine-tune the context-aware RoBERTa emotion recogniser on DailyDialog + MELD.

    python scripts/02_train_recogniser.py
"""
import _bootstrap  # noqa: F401
import json

from emodial.config import cli, device, seed_everything
from emodial.data.common import read_jsonl
from emodial.pipeline import Paths
from emodial.recogniser import model as R

args, cfg = cli(__doc__)
seed_everything(cfg["seed"])
P, rc, dev = Paths(cfg), cfg["recogniser"], device()
train_rows = read_jsonl(P.processed / "erc_train.jsonl")
valid_rows = read_jsonl(P.processed / "erc_valid.jsonl")
test_rows = read_jsonl(P.processed / "erc_test.jsonl")
print(f"train {len(train_rows)}  valid {len(valid_rows)}  test {len(test_rows)}  on {dev}")

summary = R.train(rc, train_rows, valid_rows, P.recogniser, dev)
tok, model = R.load_classifier(str(P.recogniser))
summary["test"] = R.evaluate(model, tok, test_rows, rc, dev)
for src in sorted({r["source"] for r in test_rows}):
    summary[f"test_{src}"] = R.evaluate(model, tok, [r for r in test_rows if r["source"] == src], rc, dev)
(P.recogniser / "metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k.startswith("test")}, indent=2))
