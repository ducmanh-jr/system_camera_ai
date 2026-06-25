"""End-to-end CRNN model for VN-plate OCR."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .backbone import VGGLite
from .neck import BiLSTMNeck, TransformerNeck
from .head import CTCHead
from .init import init_weights


def _build_backbone(cfg: dict, in_channels: int) -> nn.Module:
    typ = cfg.get("type", "vgg_lite")
    if typ == "vgg_lite":
        return VGGLite(
            in_channels=in_channels,
            channels=cfg.get("channels", [64, 128, 256, 512, 512]),
            norm=cfg.get("norm", "gn"),
            gn_groups=cfg.get("gn_groups", 8),
        )
    raise ValueError(f"unknown backbone: {typ}")


def _build_neck(cfg: dict, in_channels: int) -> nn.Module:
    typ = cfg.get("type", "bilstm")
    if typ == "bilstm":
        return BiLSTMNeck(in_channels=in_channels,
                          hidden=cfg.get("hidden", 256),
                          num_layers=cfg.get("num_layers", 2),
                          dropout=cfg.get("dropout", 0.2),
                          bidirectional=cfg.get("bidirectional", True))
    if typ == "transformer":
        return TransformerNeck(in_channels=in_channels,
                               hidden=cfg.get("hidden", 256),
                               num_layers=cfg.get("num_layers", 4),
                               nhead=cfg.get("nhead", 8),
                               dropout=cfg.get("dropout", 0.1))
    if typ == "none":
        return nn.Identity()
    raise ValueError(f"unknown neck: {typ}")


class CRNN(nn.Module):
    """Standard CRNN: CNN backbone -> sequence neck -> CTC head.

    Input  : (B, C_in, H, W)
    Output : log-softmax over vocab, shape (T, B, num_classes).
             T is the sequence length determined by the backbone (depends on W).
    """

    def __init__(self, model_cfg: dict, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = _build_backbone(model_cfg["backbone"], in_channels)
        bb_c = self.backbone.out_channels
        neck_cfg = model_cfg.get("neck", {"type": "bilstm"})
        self.neck = _build_neck(neck_cfg, bb_c)
        # determine neck output channels
        if isinstance(self.neck, nn.Identity):
            neck_c = bb_c
        else:
            neck_c = getattr(self.neck, "out_channels", bb_c)
        self.head = CTCHead(neck_c, num_classes, dropout=model_cfg.get("head", {}).get("dropout", 0.1))
        # Initialize ALL weights from scratch — no pretrained weights anywhere.
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (T, B, num_classes). Use log_softmax(-1) for CTCLoss."""
        feat = self.backbone(x)                  # (B, C, 1, T)
        b, c, h, t = feat.shape
        assert h == 1, f"expected H=1 after backbone, got {h}"
        feat = feat.squeeze(2)                   # (B, C, T)
        seq = feat.permute(2, 0, 1).contiguous() # (T, B, C)
        seq = self.neck(seq)                     # (T, B, C')
        logits = self.head(seq)                  # (T, B, num_classes)
        return logits

    @torch.no_grad()
    def predict_logp(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-softmax probabilities, shape (T, B, num_classes)."""
        logits = self.forward(x)
        return torch.nn.functional.log_softmax(logits, dim=-1)

    # ---- convenience ----
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (not only_trainable or p.requires_grad))


def build_model(model_cfg: dict, num_classes: int, in_channels: int = 1) -> CRNN:
    return CRNN(model_cfg=model_cfg, num_classes=num_classes, in_channels=in_channels)
