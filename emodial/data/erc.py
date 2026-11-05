"""Emotion-recognition-in-conversation data for the recogniser: DailyDialog + MELD.

Output format (JSONL), one dialogue per line:
    {"id": "dd_train_12", "source": "dailydialog", "utterances": [...], "labels": ["neutral", "joy", ...]}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..emotions import DAILYDIALOG_MAP, LABEL2ID
from .common import clean


def parse_dailydialog_dir(root: str | Path, split: str) -> list[dict]:
    """Read dialogues_<split>.txt and dialogues_emotion_<split>.txt found anywhere under root.

    split is one of train / validation / test (DailyDialog's own names).
    """
    root = Path(root)
    text_f = next(root.rglob(f"dialogues_{split}.txt"))
    emo_f = next(root.rglob(f"dialogues_emotion_{split}.txt"))
    texts = text_f.read_text(encoding="utf-8").splitlines()
    emos = emo_f.read_text(encoding="utf-8").splitlines()
    out = []
    for i, (t, e) in enumerate(zip(texts, emos)):
        utts = [clean(u) for u in t.split("__eou__") if u.strip()]
        labs = [DAILYDIALOG_MAP[int(x)] for x in e.split()]
        if len(utts) != len(labs):  # a handful of known misalignments: skip them
            continue
        out.append({"id": f"dd_{split}_{i}", "source": "dailydialog", "utterances": utts, "labels": labs})
    return out


def parse_meld_csv(path: str | Path, split: str) -> list[dict]:
    df = pd.read_csv(path)
    out = []
    for did, g in df.sort_values(["Dialogue_ID", "Utterance_ID"]).groupby("Dialogue_ID"):
        labs = [e.strip().lower() for e in g["Emotion"]]
        if any(l not in LABEL2ID for l in labs):
            continue
        out.append({"id": f"meld_{split}_{did}", "source": "meld",
                    "utterances": [clean(str(u)) for u in g["Utterance"]], "labels": labs})
    return out


def utterance_pairs(dialogues: list[dict]) -> list[dict]:
    """Flatten to (previous turn, current utterance, label) rows for the classifier."""
    rows = []
    for d in dialogues:
        for j, (u, l) in enumerate(zip(d["utterances"], d["labels"])):
            rows.append({"prev": d["utterances"][j - 1] if j > 0 else "", "text": u,
                         "label": LABEL2ID[l], "source": d["source"]})
    return rows
