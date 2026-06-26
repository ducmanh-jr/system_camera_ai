"""Augmentation pipeline tailored for Vietnamese license plate OCR.

Key design choices:
  - NO flip (would invert characters)
  - Rotation limited to ±5° (real plates rarely tilt more)
  - Perspective ≤ 0.06 (camera angle realistic)
  - Aggressive ImageCompression + MotionBlur + Noise to mimic CCTV
  - CoarseDropout for occlusion (dirt, screw heads, stickers)
  - Color jitter mild because plate texture matters more than color

The augmentation operates on uint8 BGR numpy arrays.
"""
from __future__ import annotations
import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:  # pragma: no cover
    HAS_ALB = False


def build_train_augment() -> "A.Compose":
    """Return an Albumentations pipeline. Call with `aug(image=img)['image']`."""
    if not HAS_ALB:
        raise ImportError("albumentations required for training augmentation")
    return A.Compose([
        # geometric: small in-plane rotation + shear + perspective
        A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Affine(shear={'x': (-5, 5)}, scale=(0.95, 1.05),
                 translate_percent={'x': (-0.02, 0.02), 'y': (-0.02, 0.02)},
                 p=0.5, mode=cv2.BORDER_CONSTANT, cval=0),
        A.Perspective(scale=(0.02, 0.06), p=0.3, pad_mode=cv2.BORDER_CONSTANT,
                       pad_val=0, fit_output=False),
        # blur: motion / gauss / defocus
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
        ], p=0.4),
        # noise: gaussian / ISO
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        # compression artifacts (CCTV / re-encode)
        A.ImageCompression(quality_lower=40, quality_upper=85, p=0.5),
        # photometric: brightness / contrast / shadow
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0.0, 1, 1.0), num_shadows_lower=1,
                        num_shadows_upper=2, shadow_dimension=4, p=0.2),
        # dirty / occlusion (3 holes max)
        A.CoarseDropout(max_holes=3, max_height=10, max_width=20,
                         min_holes=1, min_height=4, min_width=6,
                         fill_value=0, p=0.2),
    ])


def build_val_augment() -> None:
    """No augmentation at validation/test time."""
    return None
