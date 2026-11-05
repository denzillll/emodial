"""DialoGPT plus an optional emotion soft prompt (B1 / ETG variants)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..emotions import K
from .data import encode_context
from .etg import EmotionPrompt


class EmoDial(nn.Module):
    def __init__(self, lm, tok, variant: str = "b0", m: int = 4):
        super().__init__()
        self.lm, self.tok, self.variant, self.m = lm, tok, variant, m
        hidden = lm.config.hidden_size if hasattr(lm.config, "hidden_size") else lm.config.n_embd
        self.prompt = None if variant == "b0" else EmotionPrompt(variant, K, m, hidden)

    # ------------------------------------------------------------------ loading / saving
    @classmethod
    def from_lm(cls, lm_path: str, variant: str, m: int = 4) -> "EmoDial":
        tok = AutoTokenizer.from_pretrained(lm_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lm = AutoModelForCausalLM.from_pretrained(lm_path)
        return cls(lm, tok, variant, m)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(out / "lm")
        self.tok.save_pretrained(out / "lm")
        if self.prompt is not None:
            torch.save(self.prompt.state_dict(), out / "emotion_prompt.pt")
        (out / "emodial.json").write_text(json.dumps({"variant": self.variant, "m": self.m}))

    @classmethod
    def load(cls, out_dir: str | Path) -> "EmoDial":
        out = Path(out_dir)
        meta = json.loads((out / "emodial.json").read_text())
        model = cls.from_lm(str(out / "lm"), meta["variant"], meta["m"])
        if model.prompt is not None:
            model.prompt.load_state_dict(torch.load(out / "emotion_prompt.pt", map_location="cpu"))
        return model

    # ------------------------------------------------------------------ core
    def _embed(self, input_ids, attention_mask, p, delta):
        emb = self.lm.get_input_embeddings()(input_ids)
        if self.prompt is None:
            return emb, attention_mask, None
        s, g = self.prompt(p.to(emb.dtype), delta.to(emb.dtype))
        s = s.to(emb.dtype)
        mask = torch.cat([torch.ones(s.shape[:2], dtype=attention_mask.dtype, device=emb.device), attention_mask], 1)
        return torch.cat([s, emb], 1), mask, g

    def forward(self, input_ids, attention_mask, labels, p, delta, reduction: str = "mean"):
        emb, mask, g = self._embed(input_ids, attention_mask, p, delta)
        if self.prompt is not None:
            labels = torch.cat([torch.full((labels.size(0), self.m), -100, device=labels.device), labels], 1)
        logits = self.lm(inputs_embeds=emb, attention_mask=mask).logits[:, :-1].float()
        tgt = labels[:, 1:]
        tok_loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                               ignore_index=-100, reduction="none").view(tgt.shape)
        n_tok = (tgt != -100).sum(1)
        if reduction == "sum":           # per-example summed loss and token counts (for perplexity)
            return tok_loss.sum(1), n_tok, g
        return tok_loss.sum() / n_tok.sum().clamp(min=1), g

    @torch.no_grad()
    def generate(self, rows: list[dict], gen_cfg: dict, max_context: int, device: str) -> tuple[list[str], list]:
        """Batched generation with left padding: [pad ... pad][soft prompt][context]."""
        self.eval()
        ctx = [encode_context(self.tok, r["context"], max_context) for r in rows]
        p = torch.tensor([r["p"] for r in rows], dtype=torch.float, device=device)
        d = torch.tensor([r["delta"] for r in rows], dtype=torch.float, device=device)
        emb_layer = self.lm.get_input_embeddings()
        pieces, gates = [], None
        if self.prompt is not None:
            s, g = self.prompt(p.to(emb_layer.weight.dtype), d.to(emb_layer.weight.dtype))
            gates = None if g is None else g.float().cpu().tolist()
        for i, c in enumerate(ctx):
            e = emb_layer(torch.tensor(c, device=device))
            pieces.append(torch.cat([s[i], e], 0) if self.prompt is not None else e)
        L = max(x.size(0) for x in pieces)
        H = pieces[0].size(1)
        emb = torch.zeros(len(rows), L, H, dtype=pieces[0].dtype, device=device)
        mask = torch.zeros(len(rows), L, dtype=torch.long, device=device)
        for i, x in enumerate(pieces):
            emb[i, L - x.size(0):] = x
            mask[i, L - x.size(0):] = 1
        out = self.lm.generate(inputs_embeds=emb, attention_mask=mask, pad_token_id=self.tok.eos_token_id,
                               eos_token_id=self.tok.eos_token_id, **gen_cfg)
        texts = [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]
        return texts, (gates if gates is not None else [None] * len(rows))
