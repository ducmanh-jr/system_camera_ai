"""VGG-style lightweight backbone for OCR (CRNN-compatible)."""
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


class VGGLite(nn.Module):
    """5-block VGG-lite OCR backbone.

    Pooling pattern (2,2)-(2,2)-(2,1)-(2,1) + 3x1 conv shrinks H 48->1, W /4.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = (64, 128, 256, 512, 512),
        norm: str = "gn",
        gn_groups: int = 8,
    ):
        super().__init__()
        assert len(channels) == 5
        c = channels

        def block(in_c, out_c, pool=(2, 2)):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                _norm(norm, out_c, gn_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                _norm(norm, out_c, gn_groups),
                nn.ReLU(inplace=True),
            ]
            if pool is not None:
                layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
            return nn.Sequential(*layers)

        self.block1 = block(in_channels, c[0], pool=(2, 2))
        self.block2 = block(c[0], c[1], pool=(2, 2))
        self.block3 = block(c[1], c[2], pool=(2, 1))
        self.block4 = block(c[2], c[3], pool=(2, 1))
        # H=3 -> H=1 cleanly with 3x1 kernel, no padding on H
        self.block5 = nn.Sequential(
            nn.Conv2d(c[3], c[4], kernel_size=(3, 1), padding=0, bias=False),
            _norm(norm, c[4], gn_groups),
            nn.ReLU(inplace=True),
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
