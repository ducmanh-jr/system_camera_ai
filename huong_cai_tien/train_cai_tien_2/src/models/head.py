"""CTC head — a single linear classifier over the vocabulary (including blank)."""
from __future__ import annotations
import torch
import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C)
        x = self.drop(x)
        return self.fc(x)
