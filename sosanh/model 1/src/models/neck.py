"""Sequence modeling necks: BiLSTM (default) and Transformer encoder (optional)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class BiLSTMNeck(nn.Module):
    """Stacked bidirectional LSTM, input/output shape (T, B, C)."""

    def __init__(self, in_channels: int, hidden: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_channels = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C_in)
        out, _ = self.rnn(x)
        return out  # (T, B, C_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        return x + self.pe[: x.size(0)]


class TransformerNeck(nn.Module):
    """Transformer encoder neck — input/output (T, B, C).

    Only use this if you have a synthetic dataset of >= 100K samples;
    otherwise BiLSTM converges better from scratch.
    """

    def __init__(self, in_channels: int, hidden: int = 256, num_layers: int = 4,
                 nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden) if in_channels != hidden else nn.Identity()
        self.pos = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
                                           dim_feedforward=hidden * 4,
                                           dropout=dropout, activation="gelu",
                                           batch_first=False, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_channels = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos(x)
        return self.enc(x)
