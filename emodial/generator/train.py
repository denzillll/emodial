"""Training and perplexity for the generator."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..train_utils import Stepper, autocast, make_optimizer, steps_for
from .data import Collator
from .model import EmoDial


@torch.no_grad()
def perplexity(model: EmoDial, rows, gc: dict, device: str, batch_size: int | None = None) -> dict:
    model.eval().to(device)
    col = Collator(model.tok, gc["max_context_tokens"], gc["max_response_tokens"])
    dl = DataLoader(rows, batch_size=batch_size or gc["eval_batch_size"], shuffle=False, collate_fn=col)
    tot, n, gates = 0.0, 0, []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        with autocast(device, gc["fp16"]):
            loss_sum, n_tok, g = model(**b, reduction="sum")
        tot += loss_sum.sum().item()
        n += n_tok.sum().item()
        if g is not None:
            gates += g.float().cpu().tolist()
    return {"ppl": math.exp(tot / max(n, 1)), "gates": gates}


def train(model: EmoDial, train_rows, valid_rows, gc: dict, stage: dict, out_dir: str | Path, device: str) -> dict:
    """Train for stage['epochs'] epochs; keep the checkpoint with the best validation perplexity."""
    model.to(device)
    col = Collator(model.tok, gc["max_context_tokens"], gc["max_response_tokens"])
    dl = DataLoader(train_rows, batch_size=gc["batch_size"], shuffle=True, collate_fn=col)
    accum = gc["grad_accum"]
    total = steps_for(len(dl), stage["epochs"], accum)
    groups = [{"params": model.lm.parameters(), "lr": float(stage["lr"])}]
    if model.prompt is not None:
        groups.append({"params": model.prompt.parameters(), "lr": float(gc["prompt_lr"])})
    opt, sched = make_optimizer(groups, total, gc["warmup_ratio"])
    stepper = Stepper(model.parameters(), opt, sched, device, gc["fp16"], accum)
    eval_every = max(1, int(len(dl) * gc["eval_every_epochs"]))
    best, history, step = float("inf"), [], 0
    max_batches = int(len(dl) * stage["epochs"])
    while step < max_batches:
        for b in dl:
            model.train()
            b = {k: v.to(device) for k, v in b.items()}
            with autocast(device, gc["fp16"]):
                loss, _ = model(**b)
            stepper.backward(loss)
            step += 1
            if step % gc["log_every"] == 0:
                print(f"  step {step}/{max_batches}  loss {loss.item():.4f}")
            if step % eval_every == 0 or step == max_batches:
                ppl = perplexity(model, valid_rows, gc, device)["ppl"]
                history.append({"step": step, "valid_ppl": ppl})
                print(f"  step {step}: valid ppl {ppl:.3f}")
                if ppl < best:
                    best = ppl
                    model.save(out_dir)
            if step >= max_batches:
                break
    return {"best_valid_ppl": best, "history": history}
