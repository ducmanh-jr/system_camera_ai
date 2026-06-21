import argparse
import copy
import csv
import io
import json
import math
import os
import random
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

PROVINCES = {
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
    "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37",
    "38", "39", "40", "41", "43", "47", "48", "49", "50", "51", "52", "53", "54",
    "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67",
    "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
    "81", "82", "83", "84", "85", "86", "88", "89", "90", "92", "93", "94", "95",
    "97", "98", "99",
}
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVXYZ") + ["Đ"]  # J included: verified in charset.txt
SPECIAL_MIDS = {"LD", "MĐ", "TĐ", "HC", "AB", "NG", "QT"}

NUMERIC_CANDIDATES = {
    "O": ["0"], "Q": ["0"], "D": ["0"], "Đ": ["0"],
    "I": ["1"], "L": ["1"], "T": ["1", "7"],
    "Z": ["2"], "A": ["4"], "S": ["5"], "G": ["6"], "B": ["8"],
}
LETTER_CANDIDATES = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L", "T"],
    "2": ["Z"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
}
LETTER_CONFUSIONS = {
    "D": ["Đ"], "Đ": ["D"], "O": ["Q"], "Q": ["O"], "C": ["G"], "G": ["C"],
    "H": ["N"], "N": ["H", "M"], "M": ["N"], "K": ["X"], "X": ["K"],
    "U": ["V"], "V": ["U", "Y"], "Y": ["V"], "P": ["R", "F"], "R": ["P"],
    "E": ["F"], "F": ["E", "P"], "T": ["I"], "I": ["T", "L", "J"], "L": ["I"],
    "J": ["I"],  # J verified in charset.txt; visually similar to I
}
DIGIT_CONFUSIONS = {
    "0": ["8", "6", "3"],
    "1": ["7", "0"],
    "2": ["0"],
    "3": ["8", "0"],
    "5": ["6"],
    "6": ["5", "0"],
    "7": ["1", "0"],
    "8": ["0", "3"],
    "9": ["8", "0"],
}
VALID_CHARS = set("0123456789 ABCDEFGHIJKLMNOPQRSTUVXYZĐ")  # W excluded: not in charset.txt


DEFAULT_CONFIG = {
    "epochs": 280,
    "batch_size": 128,
    "num_workers": 4,
    "synthetic_ratio": 4,
    "multi_sizes": "40x160,48x192,56x224",
    "img_h": 48,
    "img_w": 192,
    "model_dim": 256,
    "layers": 8,
    "heads": 8,
    "dropout": 0.12,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "warmup_steps": 1800,
    "ema_decay": 0.999,
    "semantic_weight": 0.08,
    "hard_multiplier": 2,
    "seed": 42,
    "amp": True,
    "patience": 30,
}


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if not path:
        return cfg
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        cfg.update(json.loads(text))
        return cfg
    try:
        import yaml
        loaded = yaml.safe_load(text) or {}
        cfg.update(loaded)
        return cfg
    except Exception:
        pass
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        if val.lower() in {"true", "false"}:
            cfg[key] = val.lower() == "true"
        else:
            try:
                cfg[key] = int(val)
            except ValueError:
                try:
                    cfg[key] = float(val)
                except ValueError:
                    cfg[key] = val.strip("\"'")
    return cfg


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(r["image_path"], r["label"]) for r in reader]


def parse_sizes(spec: str):
    sizes = []
    for item in spec.split(","):
        h, w = item.lower().strip().split("x")
        sizes.append((int(h), int(w)))
    return sizes


class Codec:
    def __init__(self, charset: str):
        self.blank = 0
        self.chars = list(charset)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos = {i + 1: c for i, c in enumerate(self.chars)}

    @property
    def num_classes(self):
        return len(self.chars) + 1

    def encode_ctc(self, labels):
        targets, lengths = [], []
        for label in labels:
            ids = [self.stoi[c] for c in label if c in self.stoi]
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def encode_presence(self, labels):
        y = torch.zeros((len(labels), len(self.chars)), dtype=torch.float32)
        for row, label in enumerate(labels):
            for c in set(label):
                if c in self.stoi:
                    y[row, self.stoi[c] - 1] = 1.0
        return y

    def decode(self, logits):
        # BestOCR returns batch-first logits [B, T, C] so DataParallel can gather
        # multiple GPUs along the batch dimension without breaking CTC shapes.
        ids = logits.argmax(-1).cpu().tolist()
        texts = []
        for seq in ids:
            out, prev = [], self.blank
            for idx in seq:
                if idx != self.blank and idx != prev:
                    out.append(self.itos.get(idx, ""))
                prev = idx
            texts.append("".join(out))
        return texts


def resize_pad(img: Image.Image, img_h: int, img_w: int):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(img_w / max(w, 1), img_h / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    canvas.paste(img, ((img_w - nw) // 2, (img_h - nh) // 2))
    return canvas


def pil_to_tensor(img: Image.Image):
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((arr - 0.5) / 0.5)


def random_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_motion_blur(img, difficulty):
    if random.random() > 0.20 * difficulty:
        return img
    radius = random.choice([3, 5])
    kernel = Image.new("L", (radius, radius), 0)
    draw = ImageDraw.Draw(kernel)
    if random.random() < 0.5:
        draw.line((0, radius // 2, radius, radius // 2), fill=255, width=1)
    else:
        draw.line((0, 0, radius, radius), fill=255, width=1)
    return img.filter(ImageFilter.Kernel((radius, radius), list(kernel.getdata()), scale=255))


def random_affine(img: Image.Image, difficulty: float):
    w, h = img.size
    # Robustness fix: Added random quad perspective warping to handle chéo/skew
    if random.random() < 0.40 * difficulty:
        dx = w * random.uniform(0.02, 0.08) * difficulty
        dy = h * random.uniform(0.03, 0.10) * difficulty
        if random.random() < 0.5:
            quad_data = [
                dx, dy,
                0, h - dy,
                w, h,
                w - dx, 0
            ]
        else:
            quad_data = [
                0, 0,
                dx, h,
                w - dx, h - dy,
                w, dy
            ]
        return img.transform((w, h), Image.Transform.QUAD, quad_data, Image.Resampling.BILINEAR, fillcolor=(245, 245, 245))

    shear = random.uniform(-0.10, 0.10) * difficulty
    tx = random.uniform(-0.03, 0.03) * difficulty * w
    ty = random.uniform(-0.04, 0.04) * difficulty * h
    return img.transform((w, h), Image.Transform.AFFINE, (1, shear, tx, 0, 1, ty), Image.Resampling.BILINEAR, fillcolor=(245, 245, 245))


def augment(img: Image.Image, difficulty: float):
    difficulty = max(0.0, min(1.0, difficulty))
    if random.random() < 0.45 + 0.25 * difficulty:
        # Robustness fix: increased rotation range from ±5° to ±15° based on
        # robustness test showing -8.6% accuracy drop at ±10-15° rotation
        angle = random.uniform(-3 - 12 * difficulty, 3 + 12 * difficulty)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(245, 245, 245))
    if random.random() < 0.25 + 0.25 * difficulty:
        img = random_affine(img, difficulty)
    if random.random() < 0.30 + 0.30 * difficulty:
        # Robustness fix: increased max blur radius to 3.0 (was 2.0 after P4 fix);
        # 2000-img test showed even light blur r=1.5 causes -17.6% accuracy drop,
        # meaning model needs to see harder blur during training.
        max_radius = 0.8 + 2.2 * difficulty  # max 3.0 at difficulty=1.0
        blur_radius = random.uniform(0.0, max_radius)
        if random.random() < 0.08 * difficulty:  # 8% chance of extreme blur
            blur_radius = random.uniform(3.0, 5.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = random_motion_blur(img, difficulty)
    if random.random() < 0.70:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.50))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.70, 1.35))
    if random.random() < 0.35 * difficulty:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(random.randint(1, 3)):
            x0 = random.randint(0, max(0, w - 2))
            y0 = random.randint(0, max(0, h - 2))
            x1 = min(w, x0 + random.randint(3, max(4, w // 8)))
            y1 = min(h, y0 + random.randint(2, max(3, h // 5)))
            shade = random.randint(210, 255)
            draw.rectangle([x0, y0, x1, y1], fill=(shade, shade, shade))
    if random.random() < 0.50:
        arr = np.asarray(img, dtype=np.float32)
        scale = random.uniform(1.0, 5.0 + 10.0 * difficulty) * 1.73
        arr += np.random.uniform(-scale, scale, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    if random.random() < 0.20 * difficulty:
        # Robustness fix: Simulate camera distance (low resolution / xa)
        # by resizing both width and height down by 0.50x - 0.85x, then back up.
        w, h = img.size
        scale_w = random.uniform(0.50, 0.85)
        scale_h = random.uniform(0.60, 0.85)
        small = img.resize((max(8, int(w * scale_w)), max(8, int(h * scale_h))), Image.Resampling.BILINEAR)
        img = small.resize((w, h), Image.Resampling.BILINEAR)

    # Robustness fix: JPEG compression augmentation — 2000-img test showed
    # quality=15 causes -11.1% accuracy drop; model had never seen JPEG artifacts.
    if random.random() < 0.15 * difficulty:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(10, 35))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def random_plate_label():
    province = random.choice(tuple(PROVINCES))
    kind = random.choices(["car", "motorbike", "special"], weights=[0.52, 0.34, 0.14], k=1)[0]
    if kind == "motorbike":
        mid = random.choice(LETTERS) + str(random.randint(1, 9))
    elif kind == "special":
        mid = random.choice(tuple(SPECIAL_MIDS))
        if mid in {"MĐ"} and random.random() < 0.40:
            mid += str(random.randint(1, 9))
    else:
        mid = random.choice(LETTERS)
        if random.random() < 0.28:
            mid += random.choice(LETTERS)
    tail = f"{random.randint(0, 99999):05d}" if random.random() > 0.08 else str(random.randint(1000, 9999))
    return f"{province}{mid} {tail}"


class SyntheticPlate:
    def __init__(self, sizes, charset):
        self.sizes = sizes
        self.charset = set(charset)
        self.difficulty = 0.15
        self.fonts = self._find_fonts()
        self._font_cache = {}
        self._size_cache = {}  # (text_len, h, size_frac_int) -> best font size

    def _find_fonts(self):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/Arial.ttf",
        ]
        return [p for p in candidates if Path(p).exists()] or [None]

    def set_epoch(self, epoch, epochs):
        self.difficulty = min(1.0, 0.08 + 0.92 * epoch / max(1, epochs))

    def _font(self, size):
        p = random.choice(self.fonts)
        if not p:
            return ImageFont.load_default()
        key = (p, size)
        if key not in self._font_cache:
            self._font_cache[key] = ImageFont.truetype(p, size)
        return self._font_cache[key]

    def _background(self, h, w):
        # Diverse plate colors: white (civilian), yellow (commercial), blue (govt), green (EV), red (military)
        plate_type = random.choices(
            ["white", "yellow", "blue", "green", "red"],
            weights=[0.45, 0.25, 0.15, 0.05, 0.10], k=1
        )[0]
        
        if plate_type == "yellow":
            # Yellow commercial plate: dark text
            base = (random.randint(235, 255), random.randint(200, 235), random.randint(0, 30))
            text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
            border_color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
        elif plate_type == "blue":
            # Blue government plate: white text
            base = (random.randint(0, 30), random.randint(50, 100), random.randint(170, 230))
            text_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
            border_color = (random.randint(210, 250), random.randint(210, 250), random.randint(210, 250))
        elif plate_type == "green":
            # Green EV plate: white text
            base = (random.randint(0, 30), random.randint(130, 175), random.randint(50, 100))
            text_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
            border_color = (random.randint(210, 250), random.randint(210, 250), random.randint(210, 250))
        elif plate_type == "red":
            # Red military plate: white text
            base = (random.randint(180, 220), random.randint(0, 30), random.randint(0, 30))
            text_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
            border_color = (random.randint(210, 250), random.randint(210, 250), random.randint(210, 250))
        else:
            # White civilian plate: dark text
            bg = random.randint(225, 255)
            base = (bg, bg, bg)
            text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
            border_color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))

        if random.random() < 0.40:
            phase = random.random() * math.pi
            x = np.arange(w, dtype=np.float32)
            sin_wave = np.sin(x / 14.0 + phase) * 10.0
            arr_r = np.clip(base[0] + sin_wave, 0, 255).astype(np.uint8)
            arr_g = np.clip(base[1] + sin_wave, 0, 255).astype(np.uint8)
            arr_b = np.clip(base[2] + sin_wave, 0, 255).astype(np.uint8)
            img = Image.fromarray(np.stack([np.tile(arr_r, (h, 1)), np.tile(arr_g, (h, 1)), np.tile(arr_b, (h, 1))], axis=-1))
        else:
            img = Image.new("RGB", (w, h), base)

        if random.random() < 0.15:
            tint = random.choice([(235, 240, 255), (245, 255, 245), (255, 248, 235)])
            img = Image.blend(img, Image.new("RGB", (w, h), tint), alpha=random.uniform(0.03, 0.12))
        return img, text_color, border_color

    def _draw_text_center(self, draw, text, w, h, y_frac, size_frac, text_color):
        cache_key = (text, h, int(size_frac * 100))
        if cache_key in self._size_cache:
            fs_found = self._size_cache[cache_key]
        else:
            fs_found = 8
            for fs in range(int(h * size_frac), 7, -1):
                font = self._font(fs)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if tw <= w * 0.92 and th <= h * 0.72:
                    fs_found = fs
                    break
            self._size_cache[cache_key] = fs_found
        font = self._font(fs_found)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = int((w - tw) / 2 + random.randint(-4, 4))
        y = int(h * y_frac + random.randint(-2, 2))
        
        # Shadow color: opposite brightness of text
        if random.random() < 0.20:
            shadow_val = 180 if sum(text_color)//3 < 128 else 40
            shadow = (shadow_val, shadow_val, shadow_val)
            draw.text((x + 1, y + 1), text, fill=shadow, font=font)
        draw.text((x, y), text, fill=text_color, font=font)

    def render(self, size):
        h, w = size
        label = "".join(c for c in random_plate_label() if c in self.charset)
        img, text_color, border_color = self._background(h, w)
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([1, 1, w - 2, h - 2], radius=3, outline=border_color, width=random.choice([1, 1, 2]))
        if random.random() < 0.35:
            for sx in (int(w * 0.10), int(w * 0.90)):
                sy = random.randint(int(h * 0.20), int(h * 0.80))
                r = max(1, h // 18)
                shade = random.randint(120, 210)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(shade, shade, shade), outline=(60, 60, 60))
        
        # Vietnamese layout standards:
        # Motorbike: top line has 4 chars ending in digit (e.g. 59G1), always 2-line
        # Car / special: mostly 1-line, sometimes 2-line
        parts = label.split(" ", 1)
        is_two_line = False
        if len(parts) == 2:
            top_part = parts[0]
            if len(top_part) == 4 and top_part[-1].isdigit():
                is_two_line = True
            else:
                is_two_line = random.random() < 0.25

        if is_two_line and " " in label:
            top, bottom = label.split(" ", 1)
            # Fix: size_frac capped at 0.38 per line so top(y=8%..46%) and
            # bottom(y=52%..90%) never overlap. Was 0.44/0.45 causing overlap.
            self._draw_text_center(draw, top, w, h, 0.08, 0.38, text_color)
            self._draw_text_center(draw, bottom, w, h, 0.52, 0.38, text_color)
        else:
            self._draw_text_center(draw, label, w, h, 0.20, random.uniform(0.52, 0.70), text_color)
        return augment(img, self.difficulty), label



class PlateDataset(Dataset):
    def __init__(self, rows, codec, sizes, train=False, synthetic_ratio=0, hard_multiplier=2):
        self.rows = rows
        self.codec = codec
        self.sizes = sizes
        self.train = train
        self.synthetic_ratio = synthetic_ratio
        # hard_multiplier controls probability of sampling hard examples
        # hard_prob = hard_multiplier / (1 + hard_multiplier), capped at 0.5
        self.hard_prob = min(0.5, hard_multiplier / (1 + hard_multiplier)) if hard_multiplier > 0 else 0.0
        self.synthetic = SyntheticPlate(sizes, codec.chars)
        self.epoch = 1
        self.epochs = 1
        self.hard_indices = []

    def set_epoch(self, epoch, epochs):
        self.epoch = epoch
        self.epochs = epochs
        self.synthetic.set_epoch(epoch, epochs)

    def set_hard_paths(self, paths):
        path_set = set(paths)
        self.hard_indices = [i for i, (p, _) in enumerate(self.rows) if p in path_set]

    def __len__(self):
        # STABLE length: never changes after construction -- required for DistributedSampler
        if not self.train:
            return len(self.rows)
        return len(self.rows) * (1 + self.synthetic_ratio)

    def __getitem__(self, idx):
        size = random.choice(self.sizes) if self.train else self.sizes[-1]
        h, w = size
        real_n = len(self.rows)
        synth_n = real_n * self.synthetic_ratio if self.train else 0

        # Stochastic hard sampling: replace real sample with a hard example with probability hard_prob
        if self.train and self.hard_indices and idx < real_n and random.random() < self.hard_prob:
            hard_idx = random.choice(self.hard_indices)
            path, label = self.rows[hard_idx]
            img = Image.open(path)
            img = resize_pad(img, h, w)
            img = augment(img, min(1.0, 0.2 + 0.8 * self.epoch / max(1, self.epochs)))
            return pil_to_tensor(img), label, path

        if self.train and idx >= real_n:
            img, label = self.synthetic.render(size)
            return pil_to_tensor(img), label, "<synthetic>"
        path, label = self.rows[idx % real_n]
        img = Image.open(path)
        img = resize_pad(img, h, w)
        if self.train:
            img = augment(img, min(1.0, 0.2 + 0.8 * self.epoch / max(1, self.epochs)))
        return pil_to_tensor(img), label, path


class ParallelCollate:
    def __init__(self, codec):
        self.codec = codec

    def __call__(self, batch):
        images, labels, paths = zip(*batch)
        max_h = max(x.shape[1] for x in images)
        max_w = max(x.shape[2] for x in images)
        padded = [F.pad(x, (0, max_w - x.shape[2], 0, max_h - x.shape[1]), value=1.0) for x in images]
        targets, target_lengths = self.codec.encode_ctc(labels)
        sem_targets = self.codec.encode_presence(labels)
        return torch.stack(padded), list(labels), list(paths), targets, target_lengths, sem_targets


class ConvBNGelu(nn.Module):
    def __init__(self, c1, c2, stride=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LocalMixBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dw = ConvBNGelu(dim, dim, groups=dim)
        self.pw = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(dim * 2, dim, 1))
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(x + self.pw(self.dw(x)))


class OCRBackbone(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNGelu(3, 64, 1),
            ConvBNGelu(64, 64, 2),
            ConvBNGelu(64, 128, 1),
            ConvBNGelu(128, 128, 2),
            ConvBNGelu(128, dim, 1),
            LocalMixBlock(dim, dropout),
            ConvBNGelu(dim, dim, 2),
            LocalMixBlock(dim, dropout),
        )

    def forward(self, x):
        feat = self.net(x)
        seq = feat.mean(dim=2).permute(2, 0, 1)
        pooled = feat.mean(dim=(2, 3))
        return seq, pooled


class BestOCR(nn.Module):
    def __init__(self, num_classes, num_chars, dim=256, layers=8, heads=8, dropout=0.12, max_len=512):
        super().__init__()
        self.backbone = OCRBackbone(dim, dropout)
        self.pos = nn.Parameter(torch.zeros(max_len, 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.ctc_head = nn.Linear(dim, num_classes)
        self.semantic_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_chars))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        seq, pooled = self.backbone(x)
        seq = seq + self.pos[: seq.size(0)]
        seq = self.encoder(seq)
        seq = self.norm(seq)
        logits = self.ctc_head(seq).permute(1, 0, 2).contiguous()
        return logits, self.semantic_head(pooled)


def unwrap(model):
    import torch.nn.parallel as parallel_mod
    if isinstance(model, (nn.DataParallel, parallel_mod.DistributedDataParallel)):
        return model.module
    return model


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(unwrap(model)).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = unwrap(model).state_dict()
        dst = self.module.state_dict()
        for k, v in dst.items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + src[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(src[k])


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + (ca != cb))
        dp = ndp
    return dp[-1]


def expand_chars(chars, pos_type):
    variants = [""]
    for c in chars:
        opts = [c]
        if pos_type == "num":
            opts += NUMERIC_CANDIDATES.get(c, []) + DIGIT_CONFUSIONS.get(c, [])
        elif pos_type == "letter":
            opts += LETTER_CANDIDATES.get(c, []) + NUMERIC_CANDIDATES.get(c, []) + LETTER_CONFUSIONS.get(c, [])
        new = []
        for prefix in variants:
            for o in dict.fromkeys(opts):
                new.append(prefix + o)
        if len(new) > 64:
            new = new[:64]
        variants = new
    return variants


def valid_mid(mid):
    return bool(re.fullmatch(r"[A-ZĐ]{1,3}\d?", mid) or mid in SPECIAL_MIDS or re.fullmatch(r"(MĐ|TĐ)\d?", mid))


def mid_penalty(mid):
    if mid in SPECIAL_MIDS or re.fullmatch(r"(MĐ|TĐ)\d?", mid):
        return 0.0
    if re.fullmatch(r"[A-ZĐ]\d", mid):
        return 0.0
    if re.fullmatch(r"[A-ZĐ]{1,2}", mid):
        if len(mid) == 2 and mid[-1] in NUMERIC_CANDIDATES:
            return 1.25
        return 0.25
    if re.fullmatch(r"[A-ZĐ]{2}\d", mid):
        return 1.0
    if re.fullmatch(r"[A-ZĐ]{3}", mid):
        return 2.5
    if re.fullmatch(r"[A-ZĐ]{3}\d", mid):
        return 1.5
    return 3.0


# Fix kinhnhiem P2: known prefixes for government/special blue plates (Type 5)
# These plates have format "[2-LETTER PREFIX] [DIGITS]" with NO province code
_GOVT_BLUE_PREFIXES = {
    "KT", "KP", "KV", "KD", "KK", "KC", "KA", "KB",
    "AD", "AT", "TC", "TK", "TH", "TT", "TM",
    "VT", "QH", "QA", "QC", "QM", "QB",
    "BC", "BK", "BT", "HN", "HB", "CD",
    "CH", "HC", "PP", "BH",
}


def normalize_plate_prediction(text: str):
    raw = "".join(c for c in text.upper().replace(" ", "") if c in VALID_CHARS)

    # Fix kinhnhiem P2: Early exit for government blue plates (type5_)
    # Format: [2-LETTER] [DIGITS] — no province code
    if len(raw) >= 4:
        prefix2 = raw[:2]
        # Expand possible letter confusions for first 2 chars
        prefix_candidates = expand_chars(prefix2, "letter")
        for p in prefix_candidates:
            if p in _GOVT_BLUE_PREFIXES:
                tail_digits = "".join(c for c in raw[2:] if c.isdigit())
                if len(tail_digits) >= 3:
                    return f"{p} {tail_digits}"

    candidates = []
    for start_idx in (0, 1):
        for tail_len in (5, 4):
            if len(raw) <= start_idx + tail_len + 2:
                continue
            prov_raw = raw[start_idx : start_idx + 2]
            mid_raw = raw[start_idx + 2 : -tail_len]
            tail_raw = raw[-tail_len:]
            for prov in expand_chars(prov_raw, "num"):
                if prov not in PROVINCES:
                    continue
                for tail in expand_chars(tail_raw, "num"):
                    if not tail.isdigit():
                        continue
                    for mid in expand_chars(mid_raw, "letter"):
                        if valid_mid(mid):
                            cand = f"{prov}{mid} {tail}"
                            changes = sum(a != b for a, b in zip(cand.replace(" ", ""), raw[start_idx:])) + \
                                      abs(len(cand.replace(" ", "")) - len(raw[start_idx:])) + start_idx
                            score = changes + mid_penalty(mid) + (1.0 if tail_len == 4 else 0.0)
                            candidates.append((score, changes, cand))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], len(x[2]), x[2]))
        return candidates[0][2]
    if len(raw) > 5:
        return raw[:-5] + " " + raw[-5:]
    return raw


def substitution_pairs(pred, label):
    a, b = pred, label
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            choices = [(dp[i - 1][j] + 1, "del"), (dp[i][j - 1] + 1, "ins"), (dp[i - 1][j - 1] + cost, "eq" if cost == 0 else "sub")]
            dp[i][j], bt[i][j] = min(choices, key=lambda x: x[0])
    i, j, out = n, m, []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in {"eq", "sub"}:
            if op == "sub":
                out.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            i -= 1
        else:
            j -= 1
    return out[::-1]


@dataclass
class Metrics:
    loss: float
    raw_exact: float
    rule_exact: float
    cer: float
    d_stroke_acc: float
    hard_paths: set
    rows: list
    confusions: Counter


def _build_char_weights(codec, device):
    """Fix kinhnhiem P1: Class-weighted loss to counteract '0'-over-prediction bias.

    Analysis showed '0' is over-predicted (30 extra insertions) while '1', '8',
    '2' are under-predicted (87, 78, 71 deletions respectively). We down-weight '0'
    and up-weight the under-predicted digits to balance the gradient signal.
    """
    weights = torch.ones(codec.num_classes, device=device)
    # Up-weight frequently missed digits
    for char, boost in [("1", 1.6), ("8", 1.5), ("2", 1.4), ("3", 1.3), ("7", 1.3), ("9", 1.2)]:
        if char in codec.stoi:
            weights[codec.stoi[char]] = boost
    # Down-weight over-predicted '0'
    if "0" in codec.stoi:
        weights[codec.stoi["0"]] = 0.70
    return weights


def run_epoch(model, loader, codec, device, optimizer=None, scaler=None, scheduler=None, ema=None, amp=False, semantic_weight=0.08):
    train = optimizer is not None
    model.train(train)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    sem_loss_fn = nn.BCEWithLogitsLoss()
    # Fix kinhnhiem P1: build per-character weight for semantic loss (CTC does not
    # support per-token weights natively, so we apply weights via the auxiliary head)
    _char_w = _build_char_weights(codec, device)
    sem_loss_fn_weighted = nn.BCEWithLogitsLoss(weight=_char_w[1:])  # skip blank token
    totals = defaultdict(float)
    hard_paths, rows, confusions = set(), [], Counter()
    for images, labels, paths, targets, target_lengths, sem_targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        sem_targets = sem_targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits, sem_logits = model(images)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).contiguous()
                input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
                ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                # Fix kinhnhiem P1: use weighted loss for auxiliary semantic head
                sem_loss = sem_loss_fn_weighted(sem_logits, sem_targets)
                loss = ctc_loss + semantic_weight * sem_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                if ema:
                    ema.update(model)
                if scheduler:
                    scheduler.step()
        preds = codec.decode(logits.detach())
        bs = len(labels)
        totals["loss"] += loss.item() * bs
        for label, pred, path in zip(labels, preds, paths):
            if train:
                raw_ok = pred == label
                totals["raw"] += int(raw_ok)
                # rule_exact not computed during training (too slow + misleading when
                # mirrored from raw_ok). History CSV will show 0.0 for train_rule_exact.
                if not raw_ok and path != "<synthetic>":
                    hard_paths.add(path)
            else:
                rule = normalize_plate_prediction(pred)
                raw_ok, rule_ok = pred == label, rule == label
                totals["raw"] += int(raw_ok)
                totals["rule"] += int(rule_ok)
                totals["edit"] += edit_distance(rule, label)
                totals["chars"] += max(1, len(label))
                if "Đ" in label:
                    totals["d_total"] += 1
                    totals["d_ok"] += int(rule_ok)
                rows.append((path, label, pred, rule, int(raw_ok), int(rule_ok)))
                if not rule_ok:
                    confusions.update(substitution_pairs(rule, label))
        totals["n"] += bs
    if torch.distributed.is_initialized():
        import torch.distributed as dist_mod
        # Gather all scalar stats across GPUs
        for k in ["loss", "raw", "rule", "edit", "chars", "d_total", "d_ok", "n"]:
            val_tensor = torch.tensor(totals[k], dtype=torch.float32, device=device)
            dist_mod.all_reduce(val_tensor, op=dist_mod.ReduceOp.SUM)
            totals[k] = val_tensor.item()
        
        # Gather hard_paths object
        gathered = [None] * dist_mod.get_world_size()
        dist_mod.all_gather_object(gathered, list(hard_paths))
        hard_paths = set(p for sub in gathered for p in sub if p is not None)

        if not train:
            gathered_rows = [None] * dist_mod.get_world_size()
            dist_mod.all_gather_object(gathered_rows, rows)
            rows = [r for sub in gathered_rows for r in sub if r is not None]
            
            gathered_conf = [None] * dist_mod.get_world_size()
            dist_mod.all_gather_object(gathered_conf, dict(confusions))
            confusions = Counter()
            for c_dict in gathered_conf:
                if c_dict:
                    confusions.update(c_dict)

    n = max(1, totals["n"])
    d_acc = totals["d_ok"] / totals["d_total"] if (not train and totals["d_total"]) else 0.0
    val_edit = totals["edit"] / max(1, totals["chars"]) if not train else 0.0
    return Metrics(totals["loss"] / n, totals["raw"] / n, totals["rule"] / n, val_edit, d_acc, hard_paths, rows, confusions)


class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup:
            lr = self.base_lr * self.step_num / max(1, self.warmup)
        else:
            p = (self.step_num - self.warmup) / max(1, self.total_steps - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * min(1, p)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "warmup": self.warmup,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state):
        self.base_lr = state.get("base_lr", self.base_lr)
        self.warmup = state.get("warmup", self.warmup)
        self.total_steps = state.get("total_steps", self.total_steps)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.step_num = state.get("step_num", self.step_num)


def atomic_torch_save(payload, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_score, cfg, charset, ema=None, epochs_no_improve=0):
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_score": best_score,
        "epochs_no_improve": epochs_no_improve,
        "config": cfg,
        "charset": charset,
    }
    if ema:
        payload["ema_model"] = ema.module.state_dict()
    atomic_torch_save(payload, Path(path))


def write_validation_outputs(out_dir, metrics):
    with (out_dir / "predictions_val.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "prediction", "rule_prediction", "correct", "rule_correct"])
        writer.writerows(metrics.rows)
    with (out_dir / "hard_examples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "prediction", "rule_prediction"])
        for row in metrics.rows:
            if not row[-1]:
                writer.writerow(row[:4])
    with (out_dir / "error_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_read", "should_be", "count"])
        for (src, tgt), count in metrics.confusions.most_common(100):
            writer.writerow([src, tgt, count])


def atomic_zip_best(out_dir: Path, epoch: int):
    """Auto-zip best checkpoint after every improvement.
    Creates /kaggle/working/plate_ocr_best_EXXXXX.zip so files survive session timeout.
    Old best zips are cleaned up to save disk space.
    """
    import zipfile as _zipfile
    important = ["best.pt", "best_ema.pt", "charset.txt", "config_used.json",
                 "history.csv", "predictions_val.csv", "error_report.csv"]
    zip_name = f"plate_ocr_best_E{epoch:04d}.zip"
    zip_path = out_dir.parent / zip_name
    tmp_path = zip_path.with_suffix(".tmp")
    try:
        with _zipfile.ZipFile(tmp_path, "w", compression=_zipfile.ZIP_DEFLATED) as z:
            for name in important:
                p = out_dir / name
                if p.exists():
                    z.write(p, arcname=f"model/{name}")
        tmp_path.replace(zip_path)
        # Remove older best zips to save disk space (keep only latest)
        for old in out_dir.parent.glob("plate_ocr_best_E*.zip"):
            if old != zip_path:
                try:
                    old.unlink()
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] auto-zip failed: {e}", flush=True)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    for key, val in DEFAULT_CONFIG.items():
        if isinstance(val, bool):
            continue
        arg = "--" + key.replace("_", "-")
        parser.add_argument(arg, type=type(val))
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in DEFAULT_CONFIG:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    ddp = "WORLD_SIZE" in os.environ and "RANK" in os.environ
    if ddp:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = (local_rank == 0)

    worker_seed = cfg["seed"] + local_rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.backends.cudnn.benchmark = True

    data_root, out_dir = Path(args.data_root), Path(args.out_dir)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()
        
    charset = (data_root / "charset.txt").read_text(encoding="utf-8").strip("\n")
    if is_master:
        (out_dir / "charset.txt").write_text(charset + "\n", encoding="utf-8")
        (out_dir / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    codec = Codec(charset)

    train_rows, val_rows = read_csv_rows(data_root / "train.csv"), read_csv_rows(data_root / "val.csv")
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.val_limit:
        val_rows = val_rows[: args.val_limit]
    train_sizes = parse_sizes(cfg["multi_sizes"])
    val_size = [(cfg["img_h"], cfg["img_w"])]
    train_ds = PlateDataset(train_rows, codec, train_sizes, True, cfg["synthetic_ratio"], cfg["hard_multiplier"])
    val_ds = PlateDataset(val_rows, codec, val_size, False)
    collate_fn = ParallelCollate(codec)

    if ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        # drop_last=True on val sampler: prevents duplicate samples from padding,
        # which would skew accuracy metrics (padded duplicates counted twice)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=True)
        batch_size_per_gpu = max(1, cfg["batch_size"] // world_size)
    else:
        train_sampler = None
        val_sampler = None
        batch_size_per_gpu = cfg["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    model = BestOCR(codec.num_classes, len(codec.chars), cfg["model_dim"], cfg["layers"], cfg["heads"], cfg["dropout"]).to(device)
    if ddp:
        # CRITICAL: Convert BatchNorm → SyncBatchNorm BEFORE DDP wrapping.
        # Without this, each GPU computes BN stats on batch_size/2 samples independently,
        # causing statistics to diverge between GPUs and degrading val accuracy.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(unwrap(model).parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    scheduler = WarmupCosine(optimizer, cfg["lr"], cfg["warmup_steps"], cfg["epochs"] * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    start_epoch, best_score, epochs_no_improve = 1, 0.0, 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        if ema and "ema_model" in ckpt:
            ema.module.load_state_dict(ckpt["ema_model"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", 0.0)
        loaded_no_improve = ckpt.get("epochs_no_improve", 0)
        loaded_patience = ckpt.get("config", {}).get("patience", cfg["patience"])
        # If checkpoint was saved from a different run (warm-start) or patience changed,
        # reset the no-improve counter so we don't immediately early-stop.
        if loaded_no_improve >= loaded_patience:
            epochs_no_improve = 0
        else:
            epochs_no_improve = loaded_no_improve

    hist = out_dir / "history.csv"
    if is_master and not hist.exists():
        hist.write_text("epoch,train_loss,train_raw_exact,train_rule_exact,train_cer,val_loss,val_raw_exact,val_rule_exact,val_cer,val_d_stroke_acc,hard_train_count,seconds\n", encoding="utf-8")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if ddp and train_sampler:
            train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch, cfg["epochs"])
        t0 = time.time()
        train_m = run_epoch(model, train_loader, codec, device, optimizer, scaler, scheduler, ema, cfg["amp"], cfg["semantic_weight"])
        if train_m.hard_paths:
            train_ds.set_hard_paths(train_m.hard_paths)
        eval_model = ema.module.to(device) if ema else model
        val_m = run_epoch(eval_model, val_loader, codec, device, None, None, None, None, False, cfg["semantic_weight"])
        
        is_best = val_m.rule_exact >= best_score
        if is_best:
            best_score = val_m.rule_exact
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        stop_training = False
        patience = cfg.get("patience", 30)
        if epochs_no_improve >= patience:
            stop_training = True
            
        if ddp:
            stop_tensor = torch.tensor(1.0 if stop_training else 0.0, device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            if stop_tensor.item() > 0.5:
                stop_training = True
                
        if stop_training:
            if is_master:
                print(f"Early stopping triggered at epoch {epoch}! No improvement for {patience} epochs.", flush=True)
            break
            
        if is_master:
            seconds = time.time() - t0
            line = (
                f"{epoch},{train_m.loss:.6f},{train_m.raw_exact:.6f},{train_m.rule_exact:.6f},{train_m.cer:.6f},"
                f"{val_m.loss:.6f},{val_m.raw_exact:.6f},{val_m.rule_exact:.6f},{val_m.cer:.6f},{val_m.d_stroke_acc:.6f},"
                f"{len(train_m.hard_paths)},{seconds:.1f}"
            )
            print(line, flush=True)
            with hist.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_score, cfg, charset, ema, epochs_no_improve)
            if is_best:
                save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_score, cfg, charset, ema, epochs_no_improve)
                if ema:
                    atomic_torch_save({"model": ema.module.state_dict(), "charset": charset, "config": cfg}, out_dir / "best_ema.pt")
                write_validation_outputs(out_dir, val_m)
                atomic_zip_best(out_dir, epoch)


if __name__ == "__main__":
    main()
