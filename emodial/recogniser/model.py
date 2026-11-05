"""Context-aware utterance emotion classifier: RoBERTa over (previous turn, current utterance)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..emotions import EMOTIONS, K, LABEL2ID
from ..train_utils import Stepper, autocast, make_optimizer, steps_for


def load_classifier(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path, num_labels=K, id2label=dict(enumerate(EMOTIONS)), label2id=LABEL2ID,
        ignore_mismatched_sizes=True)
    return tok, model


def _collate(tok, max_len):
    def fn(rows):
        enc = tok([r["prev"] for r in rows], [r["text"] for r in rows], truncation="longest_first",
                  max_length=max_len, padding=True, return_tensors="pt")
        if "label" in rows[0]:
            enc["labels"] = torch.tensor([r["label"] for r in rows])
        return enc
    return fn


def class_weights(rows, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount([r["label"] for r in rows], minlength=K).astype(float) + 1.0
    w = (counts.sum() / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float)


@torch.no_grad()
def predict_proba(model, tok, rows, batch_size=64, max_len=128, device="cpu", fp16=True) -> np.ndarray:
    model.eval().to(device)
    dl = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=_collate(tok, max_len))
    out = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with autocast(device, fp16):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, K))


def evaluate(model, tok, rows, rc, device) -> dict:
    probs = predict_proba(model, tok, rows, rc["eval_batch_size"], rc["max_len"], device, rc["fp16"])
    y, yhat = np.array([r["label"] for r in rows]), probs.argmax(-1)
    rep = classification_report(y, yhat, labels=list(range(K)), target_names=EMOTIONS,
                                output_dict=True, zero_division=0)
    return {"weighted_f1": f1_score(y, yhat, average="weighted"),
            "macro_f1": f1_score(y, yhat, average="macro"),
            "per_class_f1": {e: rep[e]["f1-score"] for e in EMOTIONS}}


def train(rc: dict, train_rows, valid_rows, out_dir: str | Path, device: str) -> dict:
    tok, model = load_classifier(rc["model_name"])
    model.to(device)
    dl = DataLoader(train_rows, batch_size=rc["batch_size"], shuffle=True, collate_fn=_collate(tok, rc["max_len"]))
    total = steps_for(len(dl), rc["epochs"], 1)
    opt, sched = make_optimizer([{"params": model.parameters(), "lr": float(rc["lr"])}], total)
    stepper = Stepper(model.parameters(), opt, sched, device, rc["fp16"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_rows).to(device))
    best, bad, history = -1.0, 0, []
    for epoch in range(int(rc["epochs"])):
        model.train()
        for i, b in enumerate(dl):
            b = {k: v.to(device) for k, v in b.items()}
            labels = b.pop("labels")
            with autocast(device, rc["fp16"]):
                logits = model(**b).logits
            stepper.backward(loss_fn(logits.float(), labels))
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{len(dl)}")
        m = evaluate(model, tok, valid_rows, rc, device)
        history.append(m)
        print(f"epoch {epoch}: valid weighted F1 {m['weighted_f1']:.4f}  macro F1 {m['macro_f1']:.4f}")
        if m["weighted_f1"] > best:
            best, bad = m["weighted_f1"], 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            bad += 1
            if bad >= rc["patience"]:
                print("early stopping")
                break
    return {"best_valid_weighted_f1": best, "history": history}
