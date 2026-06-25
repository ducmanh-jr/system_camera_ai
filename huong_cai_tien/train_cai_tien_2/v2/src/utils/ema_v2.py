"""Exponential Moving Average v2 — step-warmup decay.

Decay formula (de-facto standard from MoCo / EfficientNet):
    decay(t) = base * (1 - exp(-t / 2000))
which starts near 0 and converges to `base`. Prevents the EMA from being
contaminated by the initial random weights during the first ~1k steps.
"""
from __future__ import annotations
import math
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMAv2:
    def __init__(self, model: nn.Module, decay: float = 0.9995,
                 warmup_steps: int = 1000):
        self.base_decay = decay
        self.warmup_steps = max(1, warmup_steps)
        self.ema = deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema.eval()

    def _decay(self, step: int) -> float:
        # Warm up the decay from 0 to base over `warmup_steps` updates
        if step <= 0:
            return 0.0
        ramp = 1.0 - math.exp(-step / self.warmup_steps)
        return self.base_decay * ramp

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self._decay(step)
        for ep, mp in zip(self.ema.parameters(), model.parameters()):
            ep.mul_(d).add_(mp.detach(), alpha=1.0 - d)
        for eb, mb in zip(self.ema.buffers(), model.buffers()):
            eb.copy_(mb.detach())

    def state_dict(self) -> dict:
        return self.ema.state_dict()
