"""Turn generation examples + cached emotions into model-ready tensors."""
from __future__ import annotations

import torch

from ..emotions import K


def attach_emotions(examples: list[dict], cache: dict[str, list[float]]) -> list[dict]:
    """Add p (current user turn) and delta (p_t - p_{t-1}, zero for the first user turn)."""
    out = []
    for ex in examples:
        p = cache.get(ex["cur_key"])
        if p is None:
            continue
        prev = cache.get(ex["prev_key"]) if ex["prev_key"] else None
        delta = [a - b for a, b in zip(p, prev)] if prev is not None else [0.0] * K
        out.append({**ex, "p": list(p), "delta": delta, "shift": sum(abs(x) for x in delta)})
    return out


def encode_context(tok, context: list[str], max_tokens: int) -> list[int]:
    """Turns joined by EOS, ending with EOS; truncated from the left (oldest tokens dropped)."""
    ids: list[int] = []
    for turn in context:
        ids += tok.encode(turn, add_special_tokens=False) + [tok.eos_token_id]
    return ids[-max_tokens:]


class Collator:
    def __init__(self, tok, max_context: int, max_response: int):
        self.tok, self.max_context, self.max_response = tok, max_context, max_response

    def __call__(self, rows):
        seqs, labs = [], []
        for r in rows:
            c = encode_context(self.tok, r["context"], self.max_context)
            y = self.tok.encode(r["response"], add_special_tokens=False)[: self.max_response] + [self.tok.eos_token_id]
            seqs.append(c + y)
            labs.append([-100] * len(c) + y)
        L = max(map(len, seqs))
        pad = self.tok.eos_token_id
        ids = torch.full((len(rows), L), pad)
        mask = torch.zeros((len(rows), L), dtype=torch.long)
        labels = torch.full((len(rows), L), -100)
        for i, (s, l) in enumerate(zip(seqs, labs)):  # right padding for training
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
            labels[i, : len(l)] = torch.tensor(l)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels,
                "p": torch.tensor([r["p"] for r in rows], dtype=torch.float),
                "delta": torch.tensor([r["delta"] for r in rows], dtype=torch.float)}
