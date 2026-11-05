"""Automatic metrics: Distinct-n, the independent emotion judge, negative echo,
emotion agreement with the human reply, and gate analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..emotions import EMOTIONS, NEGATIVE_IDX


def distinct_n(texts: list[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.lower().split()
        g = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / total if total else 0.0


class Judge:
    """Independent 7-class emotion classifier (Hartmann, 2022) used ONLY for evaluation.

    Never our own recogniser, to avoid circularity.
    """

    def __init__(self, name: str, device: str = "cpu", batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device, self.bs = device, batch_size
        names = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        # map the judge's label order onto ours (identical for j-hartmann, but don't assume)
        self.order = [names.index(e) for e in EMOTIONS] if set(EMOTIONS) <= set(names) else list(range(len(EMOTIONS)))

    def probs(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.bs):
                enc = self.tok([t or "." for t in texts[i:i + self.bs]], truncation=True, max_length=128,
                               padding=True, return_tensors="pt").to(self.device)
                pr = self.torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
                out.append(pr[:, self.order])
        return np.concatenate(out) if out else np.zeros((0, len(EMOTIONS)))


def negative_rate(probs: np.ndarray, threshold: float = 0.5) -> float:
    if len(probs) == 0:
        return float("nan")
    return float((probs[:, NEGATIVE_IDX].sum(1) > threshold).mean())


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon distance (base 2, in [0, 1]); numerically safe for near-identical inputs."""
    a = np.clip(np.asarray(a, float), 1e-12, None)
    b = np.clip(np.asarray(b, float), 1e-12, None)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    js = 0.5 * np.sum(a * np.log2(a / m)) + 0.5 * np.sum(b * np.log2(b / m))
    return float(np.sqrt(max(js, 0.0)))


def emotion_agreement(gen: np.ndarray, ref: np.ndarray) -> dict:
    top1 = float((gen.argmax(1) == ref.argmax(1)).mean()) if len(gen) else float("nan")
    js = float(np.mean([js_distance(a, b) for a, b in zip(gen, ref)])) if len(gen) else float("nan")
    return {"emotion_top1_agreement": top1, "emotion_js_distance": js}


def gate_analysis(gates: list[float], shifts: list[float]) -> dict:
    g, s = np.asarray(gates, float), np.asarray(shifts, float)
    if len(g) < 3 or np.allclose(g, g[0]):
        return {"gate_mean": float(g.mean()) if len(g) else float("nan"), "gate_spearman_vs_shift": float("nan")}
    rho, pval = spearmanr(g, s)
    hi = g[s > np.median(s)].mean() if (s > np.median(s)).any() else float("nan")
    lo = g[s <= np.median(s)].mean()
    return {"gate_mean": float(g.mean()), "gate_spearman_vs_shift": float(rho), "gate_spearman_p": float(pval),
            "gate_mean_high_shift": float(hi), "gate_mean_low_shift": float(lo)}
