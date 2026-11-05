"""Tiny config helper: YAML file + dotted command-line overrides (key.sub=value)."""
from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any

import yaml


def _set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("inherit", None)
    if base:
        parent = load_config(Path(path).parent / base)
        cfg = deep_merge(parent, cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set(cfg, key, val)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cli(description: str, extra=None) -> tuple[argparse.Namespace, dict]:
    """Standard CLI: --config path plus any number of key=value overrides."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    if extra:
        extra(p)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. generator.lr=3e-5")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    return args, cfg


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
