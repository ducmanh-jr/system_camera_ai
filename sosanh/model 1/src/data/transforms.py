"""Deterministic preprocessing for VN-plate OCR.

Pipeline (used in BOTH training and inference; augmentation is applied BEFORE this):

    1. Convert to grayscale (if cfg.grayscale=True)
    2. Resize keeping aspect ratio, then right-pad to (H, W)
    3. Optional CLAHE on the H×W grayscale crop
    4. To float tensor, scale to [0,1]
    5. Normalize to [-1, 1] with mean=0.5, std=0.5
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2
import torch


@dataclass
class PreprocessConfig:
    img_height: int = 48
    img_width: int = 192
    grayscale: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    mean: float = 0.5
    std: float = 0.5
    pad_value: int = 0  # padding pixel value before normalization


class Preprocessor:
    """Deterministic image preprocess. Outputs a torch.FloatTensor (C, H, W) in [-1, 1]."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        if cfg.apply_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit,
                                          tileGridSize=cfg.clahe_tile_grid_size)
        else:
            self._clahe = None

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """img: np.uint8 HxW or HxWx3 (BGR or RGB; we don't care because we convert)."""
        assert img.dtype == np.uint8, f"expect uint8, got {img.dtype}"
        # 1. Grayscale
        if self.cfg.grayscale:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # callers pass BGR (cv2 default)
            # else: already gray
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 2. Resize giữ aspect ratio, pad bên phải
        h0 = img.shape[0]
        w0 = img.shape[1]
        target_h = self.cfg.img_height
        target_w = self.cfg.img_width
        # scale theo height
        scale = target_h / h0
        new_w = int(round(w0 * scale))
        if new_w > target_w:
            # ảnh quá dẹt → squash về target_w (hiếm gặp với plate)
            new_w = target_w
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        # pad bên phải
        if resized.ndim == 2:
            padded = np.full((target_h, target_w), self.cfg.pad_value, dtype=np.uint8)
            padded[:, :new_w] = resized
        else:
            padded = np.full((target_h, target_w, resized.shape[2]),
                              self.cfg.pad_value, dtype=np.uint8)
            padded[:, :new_w, :] = resized

        # 3. CLAHE (chỉ với grayscale)
        if self._clahe is not None and padded.ndim == 2:
            padded = self._clahe.apply(padded)

        # 4. To tensor [0,1]
        if padded.ndim == 2:
            tensor = torch.from_numpy(padded).float().unsqueeze(0) / 255.0   # (1, H, W)
        else:
            tensor = torch.from_numpy(padded).float().permute(2, 0, 1) / 255.0  # (3, H, W)

        # 5. Normalize
        tensor = (tensor - self.cfg.mean) / self.cfg.std
        return tensor
