"""Step 1 - download raw data and convert it to the shared JSONL formats.

    python scripts/01_prepare_data.py [--no-download]
"""
import _bootstrap  # noqa: F401
from collections import Counter

from emodial.config import cli
from emodial.data.common import write_jsonl
from emodial.data.download import fetch_all
from emodial.data.erc import parse_dailydialog_dir, parse_meld_csv, utterance_pairs
from emodial.data.generation import parse_ed_csv, parse_esconv
from emodial.emotions import EMOTIONS
from emodial.pipeline import Paths

args, cfg = cli(__doc__, lambda p: p.add_argument("--no-download", action="store_true"))
P = Paths(cfg)
if not args.no_download:
    fetch_all(P.raw)

# --- generation data -----------------------------------------------------------------
ed_dir = P.raw / "empatheticdialogues"
for split in ("train", "valid", "test"):
    f = ed_dir / f"{split}.csv"
    if f.exists():
        d = parse_ed_csv(f, split)
        write_jsonl(d, P.dialogues("ed", split))
        print(f"ED {split}: {len(d)} conversations")
    else:
        print(f"!! missing {f}")

if (P.raw / "ESConv.json").exists():
    for split, d in parse_esconv(P.raw / "ESConv.json", cfg["data"]["esconv_split_seed"]).items():
        write_jsonl(d, P.dialogues("esconv", split))
        print(f"ESConv {split}: {len(d)} conversations")

# --- recogniser data -------------------------------------------------------------------
splits = {"train": ("train", "train"), "valid": ("validation", "dev"), "test": ("test", "test")}
for ours, (dd_name, meld_name) in splits.items():
    dialogues = []
    try:
        dialogues += parse_dailydialog_dir(P.raw / "dailydialog", dd_name)
    except StopIteration:
        print(f"!! DailyDialog {dd_name} not found under {P.raw / 'dailydialog'}")
    meld_f = P.raw / "meld" / f"{meld_name}_sent_emo.csv"
    if meld_f.exists():
        dialogues += parse_meld_csv(meld_f, meld_name)
    rows = utterance_pairs(dialogues)
    write_jsonl(rows, P.processed / f"erc_{ours}.jsonl")
    counts = Counter(EMOTIONS[r["label"]] for r in rows)
    print(f"ERC {ours}: {len(rows)} utterances  {dict(counts)}")
