"""Synthetic VN license-plate generator.

Generates rule-based images of Vietnamese license plates with realistic noise,
blur, JPEG artifacts, shadows, and perspective distortion. This is **not** a
pretrained model — it is purely procedural rendering with PIL/OpenCV, so it
fully complies with the constraint "train from scratch, no pretrained / transfer".

VN plate code structure (simplified):

    <prov-2digits><series-1or2letters>[0-9]? <4-or-5-digit-tail>

Examples:
    "59A1 00128"   "68HC 00042"   "29Z 5270"   "30E 99077"

We avoid the letter ``W`` (not used in VN plates) and rarely use ``I, O`` to
mimic the real distribution. The space is always one character between the
two halves.
"""
from __future__ import annotations
import os
import random
import string
from typing import List, Optional, Tuple
import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

# Province codes officially in use (sample — extend if desired)
VN_PROVINCE_CODES = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "43", "47", "48", "49", "50", "51", "52", "53", "54",
    "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66",
    "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78",
    "79", "80", "81", "82", "83", "84", "85", "86", "88", "89", "90", "92",
    "93", "94", "95", "97", "98", "99",
]

# Series letters that are actually observed in VN plates (no W; I/O rare)
SERIES_LETTERS_COMMON = list("ABCDEFGHKLMNPRSTUVXYZ")  # no W, no I, no O, no Q, no J
SERIES_LETTERS_BUSINESS = list("ABCDEFGHKLMNPRSTUVXYZ")
SUFFIX_DIGIT_PROB = 0.55  # probability of "A1", "B2" style (newer plates have it)

# Background colors for VN plates
PLATE_BG_COLORS = {
    "white": ((230, 230, 230), (40, 40, 40)),      # white bg, dark text
    "yellow": ((220, 200, 60), (30, 30, 30)),       # commercial / taxi
    "blue":   ((30, 60, 160),  (240, 240, 240)),    # government / military
    "red":    ((180, 30, 30),  (240, 240, 240)),    # military new
}


def random_plate_text(rng: random.Random) -> str:
    prov = rng.choice(VN_PROVINCE_CODES)
    series = rng.choice(SERIES_LETTERS_COMMON)
    # 1-letter or 2-letter series (e.g. "HC", "LD")
    if rng.random() < 0.10:
        series += rng.choice(SERIES_LETTERS_BUSINESS)
    suffix = ""
    if rng.random() < SUFFIX_DIGIT_PROB:
        suffix = rng.choice("12345")
    tail_len = rng.choice([4, 5, 5, 5])  # mostly 5
    tail = "".join(rng.choice("0123456789") for _ in range(tail_len))
    return f"{prov}{series}{suffix} {tail}"


def _draw_text(text: str, font_path: str, font_size: int,
               bg_color, fg_color, padding: int = 8) -> "Image.Image":
    if not HAS_PIL:
        raise ImportError("Pillow required for synthetic generation")
    font = ImageFont.truetype(font_path, font_size)
    # measure
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    W = text_w + padding * 2
    H = text_h + padding * 2
    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=fg_color)
    return img


def _perspective_warp(img: np.ndarray, rng: random.Random, mag: float = 0.06) -> np.ndarray:
    h, w = img.shape[:2]
    dx = mag * w
    dy = mag * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(-dx, dx),        rng.uniform(-dy, dy)],
        [w + rng.uniform(-dx, dx),    rng.uniform(-dy, dy)],
        [w + rng.uniform(-dx, dx),    h + rng.uniform(-dy, dy)],
        [rng.uniform(-dx, dx),        h + rng.uniform(-dy, dy)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out


def _add_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(2, 15)
    noise = rng.gauss(0, sigma)  # not used directly — numpy faster
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def _motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    # random angle
    ang = rng.uniform(0, 180)
    M = cv2.getRotationMatrix2D((k / 2, k / 2), ang, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / (kernel.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel)


def _jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    q = rng.randint(35, 90)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def render_plate(
    text: str,
    font_path: str,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Render a single plate with random style. Returns BGR uint8 image."""
    if rng is None:
        rng = random.Random()
    style = rng.choices(
        list(PLATE_BG_COLORS.keys()),
        weights=[6, 2, 1, 1],  # white plates are most common
        k=1,
    )[0]
    bg, fg = PLATE_BG_COLORS[style]
    font_size = rng.randint(48, 64)
    pil_img = _draw_text(text, font_path, font_size, bg, fg, padding=10)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # geometric
    if rng.random() < 0.6:
        img = _perspective_warp(img, rng, mag=rng.uniform(0.01, 0.06))
    # in-plane rotation
    if rng.random() < 0.5:
        ang = rng.uniform(-4, 4)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # noise
    if rng.random() < 0.6:
        img = _add_noise(img, rng)
    # motion blur
    if rng.random() < 0.4:
        img = _motion_blur(img, rng)
    # brightness/contrast jitter
    if rng.random() < 0.5:
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-25, 25)
        img = np.clip(img.astype(np.int16) * alpha + beta, 0, 255).astype(np.uint8)
    # final JPEG
    if rng.random() < 0.6:
        img = _jpeg(img, rng)
    return img


def generate_dataset(
    out_dir: str,
    num_samples: int,
    font_path: str,
    seed: int = 0,
) -> None:
    """Generate ``num_samples`` plates into ``<out_dir>/{images,labels.txt}``."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rng = random.Random(seed)
    labels_path = os.path.join(out_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            text = random_plate_text(rng)
            img = render_plate(text, font_path, rng)
            fname = f"syn_{i:07d}.png"
            cv2.imwrite(os.path.join(out_dir, "images", fname), img)
            f.write(f"{fname}\t{text}\n")
            if (i + 1) % 1000 == 0:
                print(f"[gen_synthetic] {i + 1}/{num_samples}")
