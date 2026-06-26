"""Exponential Moving Average of model parameters.

Boosts final validation accuracy ~0.3-1% nearly for free.
"""
from __future__ import annotations
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    """Maintains a shadow copy of model weights with exponential decay."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ep, mp in zip(self.ema.parameters(), model.parameters()):
            ep.mul_(d).add_(mp.detach(), alpha=1.0 - d)
        # also copy buffers (e.g., GroupNorm has none, BN running stats yes)
        for eb, mb in zip(self.ema.buffers(), model.buffers()):
            eb.copy_(mb.detach())

    def state_dict(self) -> dict:
        return self.ema.state_dict()
