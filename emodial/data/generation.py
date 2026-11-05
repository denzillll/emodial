"""Parsers for the two generation datasets: EmpatheticDialogues and ESConv."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .common import clean, merge_consecutive


def parse_ed_csv(path: str | Path, split: str) -> list[dict]:
    """Parse an official EmpatheticDialogues CSV (train.csv / valid.csv / test.csv).

    Columns: conv_id, utterance_idx, context(label), prompt, speaker_idx, utterance, ...
    Commas inside text are encoded as "_comma_". Odd utterance_idx = speaker (our "user").
    """
    convs: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 6:
                continue
            conv_id, utt_idx, label, prompt, _spk, utt = parts[:6]
            try:
                idx = int(utt_idx)
            except ValueError:
                continue
            convs[conv_id].append((idx, label, prompt.replace("_comma_", ","), utt.replace("_comma_", ",")))
    dialogues = []
    for conv_id, rows in convs.items():
        rows.sort()
        turns = [{"role": "user" if idx % 2 == 1 else "listener", "text": clean(u)} for idx, _, _, u in rows]
        dialogues.append({
            "id": f"ed_{split}_{conv_id}",
            "source": "ed",
            "label": rows[0][1],
            "situation": clean(rows[0][2]),
            "turns": merge_consecutive(turns),
        })
    return dialogues


def parse_esconv(path: str | Path, seed: int = 42, ratios=(0.7, 0.15, 0.15)) -> dict[str, list[dict]]:
    """Parse ESConv.json and make a fixed random train/valid/test split (ESConv has none)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    n_tr = int(ratios[0] * len(order))
    n_va = int(ratios[1] * len(order))
    split_of = {}
    for rank, i in enumerate(order):
        split_of[i] = "train" if rank < n_tr else "valid" if rank < n_tr + n_va else "test"
    out: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for i, conv in enumerate(data):
        turns = [{"role": "user" if u["speaker"] == "seeker" else "listener", "text": clean(u["content"])}
                 for u in conv["dialog"]]
        out[split_of[i]].append({
            "id": f"esconv_{split_of[i]}_{i}",
            "source": "esconv",
            "label": conv.get("emotion_type", ""),
            "situation": clean(conv.get("situation", "")),
            "turns": merge_consecutive(turns),
        })
    return out
