"""Augmentation v2 — Heavy, OCR-aware pipeline for Vietnamese license plates.

DIFFERENCES vs v1
-----------------
1. Three presets: light_ocr / medium_ocr / heavy_ocr (tune via cfg.augment.preset).
2. Rain + Fog (real Vietnamese weather; v1 had only shadow).
3. Stronger motion-blur range (up to 9 px) — matches highway/CCTV scenes.
4. JPEG quality down to 25 (low-bitrate IP cameras).
5. RandomErasing applied AFTER preprocess as a torch transform on tensor.
6. Digit-aware ElasticTransform (very small alpha) breaks stroke memorization.
7. NO MixUp / CutMix — they corrupt CTC target distribution.
8. NO horizontal flip — would invert characters.
9. Per-augmentation probabilities scaled by preset multiplier.

Usage
-----
    from src.data.augmentation_v2 import build_train_augment_v2
    aug = build_train_augment_v2(cfg["augment"])
    img = aug(image=img_bgr)["image"]

After preprocess, apply tensor_random_erasing(tensor, cfg).
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import cv2
import numpy as np
import torch

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:  # pragma: no cover
    HAS_ALB = False


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_train_augment_v2(cfg: Optional[Dict[str, Any]] = None) -> "A.Compose":
    """Return an Albumentations pipeline tuned per preset."""
    if not HAS_ALB:
        raise ImportError("albumentations>=1.3 required for v2 augmentation")
    cfg = cfg or {}
    preset = cfg.get("preset", "heavy_ocr")
    p_mult = {"light_ocr": 0.6, "medium_ocr": 0.8, "heavy_ocr": 1.0}[preset]

    rotate_limit = cfg.get("rotate_limit", 6)
    shear_limit = cfg.get("shear_limit", 6)
    perspective_scale = tuple(cfg.get("perspective_scale", [0.02, 0.08]))
    motion_blur_limit = cfg.get("motion_blur_limit", 9)
    gauss_var = tuple(cfg.get("gauss_noise_var", [10.0, 70.0]))
    iso_intensity = tuple(cfg.get("iso_noise_intensity", [0.1, 0.6]))
    jpeg_q = tuple(cfg.get("jpeg_quality", [25, 90]))
    bright = cfg.get("brightness_limit", 0.25)
    contr = cfg.get("contrast_limit", 0.25)
    shadow_p = cfg.get("random_shadow_prob", 0.30)
    rain_p = cfg.get("random_rain_prob", 0.10)
    fog_p = cfg.get("random_fog_prob", 0.10)
    cd_holes = cfg.get("coarse_dropout_max_holes", 4)

    return A.Compose([
        # -- Geometric (small, plate-realistic) ----------------------------
        A.Rotate(
            limit=rotate_limit, p=0.6 * p_mult,
            border_mode=cv2.BORDER_CONSTANT, value=0,
        ),
        A.Affine(
            shear={"x": (-shear_limit, shear_limit)},
            scale=(0.93, 1.07),
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            p=0.55 * p_mult,
            mode=cv2.BORDER_CONSTANT, cval=0,
        ),
        A.Perspective(
            scale=perspective_scale, p=0.35 * p_mult,
            pad_mode=cv2.BORDER_CONSTANT, pad_val=0, fit_output=False,
        ),
        # -- Elastic (tiny alpha) breaks memorized stroke shapes ----------
        A.ElasticTransform(
            alpha=10.0, sigma=4.0, alpha_affine=0,
            border_mode=cv2.BORDER_CONSTANT, value=0,
            p=0.15 * p_mult,
        ),

        # -- Blur (CCTV / motion) ------------------------------------------
        A.OneOf([
            A.MotionBlur(blur_limit=(3, motion_blur_limit), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
            A.ZoomBlur(max_factor=1.10, p=1.0),
        ], p=0.50 * p_mult),

        # -- Noise (sensor / ISO) ------------------------------------------
        A.OneOf([
            A.GaussNoise(var_limit=gauss_var, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=iso_intensity, p=1.0),
            A.MultiplicativeNoise(multiplier=(0.92, 1.08), p=1.0),
        ], p=0.45 * p_mult),

        # -- Compression artifacts (low-bitrate IP cameras) ----------------
        A.ImageCompression(
            quality_lower=jpeg_q[0], quality_upper=jpeg_q[1], p=0.55 * p_mult
        ),
        A.Downscale(
            scale_min=0.55, scale_max=0.85,
            interpolation=cv2.INTER_LINEAR, p=0.25 * p_mult,
        ),

        # -- Photometric (low light / sun glare / contrast) ----------------
        A.RandomBrightnessContrast(
            brightness_limit=bright, contrast_limit=contr, p=0.6 * p_mult,
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3 * p_mult),
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.15 * p_mult),

        # -- Weather (Vietnam) --------------------------------------------
        A.RandomShadow(
            shadow_roi=(0, 0.0, 1, 1.0),
            num_shadows_lower=1, num_shadows_upper=2,
            shadow_dimension=4, p=shadow_p * p_mult,
        ),
        A.RandomRain(
            slant_lower=-5, slant_upper=5, drop_length=8,
            drop_width=1, drop_color=(200, 200, 200),
            blur_value=2, brightness_coefficient=0.85,
            p=rain_p * p_mult,
        ),
        A.RandomFog(
            fog_coef_lower=0.05, fog_coef_upper=0.25,
            alpha_coef=0.08, p=fog_p * p_mult,
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.4), angle_lower=0, angle_upper=1,
            num_flare_circles_lower=1, num_flare_circles_upper=3,
            src_radius=80, p=0.05 * p_mult,
        ),

        # -- Occlusion (dirt, screws, stickers) ----------------------------
        A.CoarseDropout(
            max_holes=cd_holes, max_height=12, max_width=22,
            min_holes=1, min_height=4, min_width=6,
            fill_value=0, p=0.35 * p_mult,
        ),
    ])


def build_val_augment_v2() -> None:
    """No augmentation at validation/test time."""
    return None


# ---------------------------------------------------------------------------
# Post-preprocess (tensor-level) augmentation
# ---------------------------------------------------------------------------
class TensorRandomErasing:
    """RandomErasing applied AFTER preprocess on the [C,H,W] tensor.

    Useful because we want erasing in [-1, 1] normalized space (so the patch
    is truly noise-like, not pure black on un-normalized image).
    """

    def __init__(self, p: float = 0.25, scale=(0.02, 0.10), ratio=(0.5, 3.0),
                 value: float = 0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        C, H, W = tensor.shape
        area = H * W
        for _ in range(10):  # try 10 times to find valid patch
            target_area = float(torch.empty(1).uniform_(*self.scale).item()) * area
            log_ratio = (torch.log(torch.tensor(self.ratio[0])),
                         torch.log(torch.tensor(self.ratio[1])))
            aspect = float(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).exp().item())
            h = int(round((target_area * aspect) ** 0.5))
            w = int(round((target_area / aspect) ** 0.5))
            if h < H and w < W:
                top = int(torch.randint(0, H - h, (1,)).item())
                left = int(torch.randint(0, W - w, (1,)).item())
                tensor[:, top:top + h, left:left + w] = self.value
                return tensor
        return tensor
