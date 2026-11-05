"""Small, dependency-free training helpers shared by the recogniser and the generator."""
from __future__ import annotations

import contextlib
import math

import torch
from torch.optim import AdamW


def make_optimizer(param_groups, total_steps: int, warmup_ratio: float = 0.06):
    opt = AdamW(param_groups, weight_decay=0.01)
    warm = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def autocast(device: str, enabled: bool):
    if device == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Stepper:
    """Gradient accumulation + optional fp16 loss scaling in one place."""

    def __init__(self, params, opt, sched, device: str, fp16: bool, accum: int = 1, clip: float = 1.0):
        self.params = [p for p in params if p.requires_grad]
        self.opt, self.sched, self.accum, self.clip = opt, sched, max(1, accum), clip
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and fp16))
        self.micro = 0

    def backward(self, loss: torch.Tensor) -> bool:
        self.scaler.scale(loss / self.accum).backward()
        self.micro += 1
        if self.micro % self.accum:
            return False
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        return True


def steps_for(n_batches: int, epochs: float, accum: int) -> int:
    return max(1, math.ceil(n_batches * epochs / max(1, accum)))
