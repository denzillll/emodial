"""Glue shared by the numbered scripts: paths, loading examples with emotions, test subsets."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .data.common import iter_examples, read_jsonl
from .generator.data import attach_emotions


class Paths:
    def __init__(self, cfg: dict):
        p = cfg["paths"]
        self.raw, self.processed, self.cache, self.out = (Path(p[k]) for k in ("raw", "processed", "cache", "outputs"))
        self.emotion_cache = self.cache / "emotions.json"
        self.delta_stats = self.cache / "delta_stats.json"
        self.recogniser = self.out / "recogniser"
        self.generator = self.out / "generator"
        self.replies = self.out / "replies"
        self.results = self.out / "results"
        self.human = self.out / "human_eval"

    def dialogues(self, source: str, split: str) -> Path:
        return self.processed / f"{source}_{split}.jsonl"

    def model_dir(self, variant: str, seed: int) -> Path:
        return self.generator / f"{variant}_s{seed}"


def load_dialogues(paths: Paths, sources, split: str) -> list[dict]:
    out = []
    for s in sources:
        f = paths.dialogues(s, split)
        if f.exists():
            out += read_jsonl(f)
        else:
            print(f"  (missing {f}; skipped)")
    return out


def load_emotion_cache(paths: Paths) -> dict:
    return json.loads(paths.emotion_cache.read_text())


def load_tau(paths: Paths) -> float:
    return json.loads(paths.delta_stats.read_text())["tau"]


def examples(cfg: dict, paths: Paths, split: str, cache: dict | None = None, limit: int | None = None,
             seed: int = 0) -> list[dict]:
    cache = cache if cache is not None else load_emotion_cache(paths)
    rows = attach_emotions(list(iter_examples(load_dialogues(paths, cfg["data"]["sources"], split),
                                              cfg["data"]["context_turns"])), cache)
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


def test_subset(cfg: dict, paths: Paths, cache: dict | None = None) -> list[dict]:
    """Fixed, seeded test subset per source, identical for every system."""
    rows = examples(cfg, paths, "test", cache)
    n = cfg["eval"]["max_examples_per_source"]
    out = []
    for src in cfg["data"]["sources"]:
        r = sorted([x for x in rows if x["source"] == src], key=lambda x: x["id"])
        out += random.Random(cfg["seed"]).sample(r, min(n, len(r))) if n else r
    return out
