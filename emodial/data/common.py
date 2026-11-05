"""Shared dialogue format and helpers.

Every generation dataset (ED, ESConv) is converted to the same JSONL format:

    {"id": "ed_train_hit:0_conv:1", "source": "ed", "label": "sentimental",
     "turns": [{"role": "user", "text": "..."}, {"role": "listener", "text": "..."}, ...]}

"user" is the person sharing feelings (ED speaker / ESConv seeker) and "listener" is the
person replying (ED listener / ESConv supporter). The generator learns listener turns.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."}


def clean(text: str) -> str:
    for a, b in QUOTES.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_consecutive(turns: list[dict]) -> list[dict]:
    """Join back-to-back turns by the same role into one turn."""
    out: list[dict] = []
    for t in turns:
        if not t["text"]:
            continue
        if out and out[-1]["role"] == t["role"]:
            out[-1]["text"] += " " + t["text"]
        else:
            out.append(dict(t))
    return out


def user_turn_key(dialog_id: str, turn_idx: int) -> str:
    """Key used to cache recogniser outputs for one user turn."""
    return f"{dialog_id}#{turn_idx}"


def iter_examples(dialogues: list[dict], context_turns: int = 3) -> Iterator[dict]:
    """Yield one generation example per listener turn that follows at least one user turn.

    Each example records which user turns supply p_t (latest user turn in the context) and
    p_{t-1} (the user turn before that, anywhere earlier in the dialogue), so Delta_t can be
    looked up from the emotion cache.
    """
    for d in dialogues:
        turns = d["turns"]
        user_idx_so_far: list[int] = []
        for i, t in enumerate(turns):
            if t["role"] == "user":
                user_idx_so_far.append(i)
                continue
            if not user_idx_so_far:
                continue  # listener speaks first: nothing to respond to
            cur = user_idx_so_far[-1]
            prev = user_idx_so_far[-2] if len(user_idx_so_far) > 1 else None
            ctx = turns[max(0, i - context_turns):i]
            yield {
                "id": f"{d['id']}@{i}",
                "dialog_id": d["id"],
                "source": d["source"],
                "context": [c["text"] for c in ctx],
                "context_roles": [c["role"] for c in ctx],
                "response": t["text"],
                "cur_key": user_turn_key(d["id"], cur),
                "prev_key": user_turn_key(d["id"], prev) if prev is not None else None,
            }
