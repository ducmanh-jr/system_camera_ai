"""Weight initialization for training from scratch.

Rules:
    - Conv2d  -> Kaiming-normal with ReLU gain (fan_out)
    - Linear  -> Kaiming-normal
    - LSTM    -> orthogonal for recurrent weights, zero biases except forget=1
    - BN / GN / IN -> weight=1, bias=0

This is critical when training from scratch with no pretrained backbone:
proper init typically buys 1-3 percentage points of final accuracy and several
days of debugging time.
"""
from __future__ import annotations
import torch
import torch.nn as nn


@torch.no_grad()
def init_weights(module: nn.Module) -> None:
    """Apply to a model with ``model.apply(init_weights)``."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, p in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                # Set forget-gate bias to 1 (Jozefowicz et al. 2015)
                nn.init.zeros_(p)
                n = p.size(0)
                # bias order: i, f, g, o
                p.data[n // 4:n // 2].fill_(1.0)
