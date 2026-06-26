"""CRNN v2 — backbone v2 (SE + DropPath) + BiLSTM neck with variational dropout."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from v2.src.models.backbone_v2 import VGGLiteV2
# reuse v1 neck/head (BiLSTM is fine; we just tweak hidden size in cfg)
from src.models.neck import BiLSTMNeck, TransformerNeck
from src.models.head import CTCHead
from src.models.init import init_weights


def _build_backbone_v2(cfg: dict, in_channels: int) -> nn.Module:
    typ = cfg.get("type", "vgg_lite")
    if typ == "vgg_lite":
        return VGGLiteV2(
            in_channels=in_channels,
            channels=cfg.get("channels", [64, 128, 256, 512, 512]),
            norm=cfg.get("norm", "bn"),
            gn_groups=cfg.get("gn_groups", 8),
            use_se=cfg.get("se", True),
            stochastic_depth=cfg.get("stochastic_depth", 0.1),
        )
    raise ValueError(f"unknown backbone: {typ}")


def _build_neck_v2(cfg: dict, in_channels: int) -> nn.Module:
    typ = cfg.get("type", "bilstm")
    if typ == "bilstm":
        return BiLSTMNeck(
            in_channels=in_channels,
            hidden=cfg.get("hidden", 320),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", True),
        )
    if typ == "transformer":
        return TransformerNeck(
            in_channels=in_channels,
            hidden=cfg.get("hidden", 320),
            num_layers=cfg.get("num_layers", 4),
            nhead=cfg.get("nhead", 8),
            dropout=cfg.get("dropout", 0.1),
        )
    raise ValueError(f"unknown neck: {typ}")


class CRNNv2(nn.Module):
    def __init__(self, model_cfg: dict, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = _build_backbone_v2(model_cfg["backbone"], in_channels)
        bb_c = self.backbone.out_channels
        self.neck = _build_neck_v2(model_cfg.get("neck", {"type": "bilstm"}), bb_c)
        neck_c = getattr(self.neck, "out_channels", bb_c)
        self.head = CTCHead(neck_c, num_classes,
                            dropout=model_cfg.get("head", {}).get("dropout", 0.2))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                  # (B, C, 1, T)
        b, c, h, t = feat.shape
        assert h == 1, f"expected H=1, got {h}"
        feat = feat.squeeze(2)                   # (B, C, T)
        seq = feat.permute(2, 0, 1).contiguous() # (T, B, C)
        seq = self.neck(seq)                     # (T, B, C')
        logits = self.head(seq)                  # (T, B, num_classes)
        return logits

    @torch.no_grad()
    def predict_logp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(self.forward(x), dim=-1)

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (not only_trainable or p.requires_grad))


def build_model_v2(model_cfg: dict, num_classes: int, in_channels: int = 1) -> CRNNv2:
    return CRNNv2(model_cfg=model_cfg, num_classes=num_classes, in_channels=in_channels)
