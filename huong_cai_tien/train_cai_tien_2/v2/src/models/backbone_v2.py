"""VGGLite v2 — adds Squeeze-Excitation, Stochastic Depth, optional BN/GN switch.

Why these changes
-----------------
- SE blocks let the network re-weight feature channels per sample. Digit
  confusion (0/8, 6/8, ...) usually comes from low-contrast strokes; SE helps.
- Stochastic depth (DropPath) is the cheapest known regularizer for CNNs at
  this depth — costs nothing at inference.
- BatchNorm gives a stronger gradient signal than GroupNorm when batch >= 64,
  which the v2 config uses (effective batch 128 with grad accum).

Output shape contract is identical to v1 backbone: (B, C, 1, T).
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn


def _norm(name: str, num_channels: int, gn_groups: int = 8) -> nn.Module:
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm2d(num_channels)
    if name == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if name == "gn":
        groups = min(gn_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"unknown norm: {name}")


class SqueezeExcite(nn.Module):
    """Standard SE-Net block. Channels → squeeze → excite → re-scale."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class DropPath(nn.Module):
    """Stochastic depth per sample (drops the whole residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class VGGBlockV2(nn.Module):
    """Two 3x3 convs + optional SE + optional pool. With residual & DropPath."""

    def __init__(self, in_c: int, out_c: int, pool=(2, 2), norm: str = "bn",
                 gn_groups: int = 8, use_se: bool = False, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.n1 = _norm(norm, out_c, gn_groups)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.n2 = _norm(norm, out_c, gn_groups)
        self.act = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_c) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path)
        # residual 1x1 if dims differ
        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1, bias=False)
        else:
            self.skip = nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool) if pool is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        y = self.se(y)
        y = self.drop_path(y) + identity
        y = self.act(y)
        y = self.pool(y)
        return y


class VGGLiteV2(nn.Module):
    """5-block VGGLite v2 with SE in blocks 3-5 and stochastic depth.

    Pooling pattern (2,2)-(2,2)-(2,1)-(2,1) + 3x1 conv: H 48→1, W /4.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = (64, 128, 256, 512, 512),
        norm: str = "bn",
        gn_groups: int = 8,
        use_se: bool = True,
        stochastic_depth: float = 0.1,
    ):
        super().__init__()
        assert len(channels) == 5
        c = channels
        # linearly ramped drop_path: 0 → stochastic_depth across blocks
        dp = [stochastic_depth * i / 4 for i in range(5)]
        self.block1 = VGGBlockV2(in_channels, c[0], (2, 2), norm, gn_groups, False, dp[0])
        self.block2 = VGGBlockV2(c[0], c[1], (2, 2), norm, gn_groups, False, dp[1])
        self.block3 = VGGBlockV2(c[1], c[2], (2, 1), norm, gn_groups, use_se, dp[2])
        self.block4 = VGGBlockV2(c[2], c[3], (2, 1), norm, gn_groups, use_se, dp[3])
        self.block5 = nn.Sequential(
            nn.Conv2d(c[3], c[4], kernel_size=(3, 1), padding=0, bias=False),
            _norm(norm, c[4], gn_groups),
            nn.ReLU(inplace=True),
            SqueezeExcite(c[4]) if use_se else nn.Identity(),
        )
        self.out_channels = c[4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        if x.size(2) != 1:
            x = nn.functional.adaptive_avg_pool2d(x, (1, x.size(3)))
        return x
