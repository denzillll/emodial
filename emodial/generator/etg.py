"""Emotion Transition Gate (ETG) and the emotion-prompt variants compared in the project.

    g_t = sigmoid( w_g^T [ ||Delta_t||_1 ; p_t ] + b_g )                      (1)
    s_t = (1 - g_t) * W_c p_t  +  g_t * W_s [ p_t ; Delta_t ]                 (2)

s_t is reshaped into m soft-prompt vectors of the LM's hidden size and placed in front of
the context embeddings. Variants:

    b0          no emotion prompt at all (plain fine-tuned DialoGPT)
    b1          static prompt W_c p_t only (g = 0)
    etg_nodelta full ETG but Delta_t fixed to 0   -> controls for extra parameters
    etg_nogate  g fixed to 1 (shift branch only)  -> tests the learned gate
    etg         full model
"""
from __future__ import annotations

import torch
from torch import nn

VARIANTS = ["b0", "b1", "etg_nodelta", "etg_nogate", "etg"]


class EmotionPrompt(nn.Module):
    def __init__(self, variant: str, k: int, m: int, hidden: int, init_std: float = 0.02):
        super().__init__()
        if variant not in VARIANTS or variant == "b0":
            raise ValueError(f"EmotionPrompt is not used for variant {variant!r}")
        self.variant, self.k, self.m, self.hidden = variant, k, m, hidden
        self.static = nn.Linear(k, m * hidden)                      # W_c
        if variant != "b1":
            self.shift = nn.Linear(2 * k, m * hidden)               # W_s
        if variant in ("etg", "etg_nodelta"):
            self.gate = nn.Linear(k + 1, 1)                         # w_g, b_g
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=init_std)
                nn.init.zeros_(mod.bias)

    def forward(self, p: torch.Tensor, delta: torch.Tensor):
        """p, delta: (B, k). Returns soft prompt (B, m, hidden) and gate values (B,) or None."""
        if self.variant == "b1":
            return self.static(p).view(-1, self.m, self.hidden), None
        if self.variant == "etg_nodelta":
            delta = torch.zeros_like(delta)
        if self.variant == "etg_nogate":
            g = torch.ones(p.size(0), 1, device=p.device, dtype=p.dtype)
        else:
            g = torch.sigmoid(self.gate(torch.cat([delta.abs().sum(-1, keepdim=True), p], -1)))
        s = (1 - g) * self.static(p) + g * self.shift(torch.cat([p, delta], -1))
        return s.view(-1, self.m, self.hidden), g.squeeze(-1)

    def n_params(self) -> int:
        return sum(x.numel() for x in self.parameters())
