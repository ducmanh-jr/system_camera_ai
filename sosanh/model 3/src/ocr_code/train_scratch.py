import argparse
import copy
import csv
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import DataLoader, Dataset


PROVINCES = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
    "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37",
    "38", "39", "40", "41", "43", "47", "48", "49", "50", "51", "52", "53", "54",
    "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67",
    "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
    "81", "82", "83", "84", "85", "86", "88", "89", "90", "92", "93", "94", "95",
    "97", "98", "99",
]
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["Đ"]
SPECIAL_MIDS = ["LD", "MĐ", "TĐ", "HC", "AB", "NG", "QT"]
NUMERIC_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})
LETTER_FIX = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"})


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(r["image_path"], r["label"]) for r in reader]


class Codec:
    def __init__(self, charset: str):
        self.blank = 0
        self.chars = list(charset)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos = {i + 1: c for i, c in enumerate(self.chars)}

    @property
    def num_classes(self):
        return len(self.chars) + 1

    def encode(self, labels):
        targets, lengths = [], []
        for label in labels:
            ids = [self.stoi[c] for c in label]
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(self, logits):
        ids = logits.argmax(-1).transpose(0, 1).cpu().tolist()
        texts = []
        for seq in ids:
            out, prev = [], self.blank
            for idx in seq:
                if idx != self.blank and idx != prev:
                    out.append(self.itos.get(idx, ""))
                prev = idx
            texts.append("".join(out))
        return texts


def parse_sizes(spec: str):
    sizes = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        h, w = item.split("x")
        sizes.append((int(h), int(w)))
    if not sizes:
        raise ValueError("At least one image size is required")
    return sizes


def resize_pad(img: Image.Image, img_h: int, img_w: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(img_w / max(w, 1), img_h / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    canvas.paste(img, ((img_w - nw) // 2, (img_h - nh) // 2))
    return canvas


def pil_to_tensor(img: Image.Image):
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    data = data.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0
    return (data - 0.5) / 0.5


def random_affine(img: Image.Image, difficulty: float):
    w, h = img.size
    shear = random.uniform(-0.08, 0.08) * difficulty
    tx = random.uniform(-0.025, 0.025) * difficulty * w
    ty = random.uniform(-0.035, 0.035) * difficulty * h
    return img.transform((w, h), Image.Transform.AFFINE, (1, shear, tx, 0, 1, ty), Image.Resampling.BILINEAR, fillcolor=(245, 245, 245))


def augment(img: Image.Image, difficulty: float = 1.0):
    difficulty = max(0.0, min(1.0, difficulty))
    if random.random() < 0.45 + 0.25 * difficulty:
        angle = random.uniform(-2 - 3 * difficulty, 2 + 3 * difficulty)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(245, 245, 245))
    if random.random() < 0.20 + 0.30 * difficulty:
        img = random_affine(img, difficulty)
    if random.random() < 0.25 + 0.35 * difficulty:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.4 + 0.9 * difficulty)))
    if random.random() < 0.65:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.45))
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
    if random.random() < 0.60:
        import numpy as np
        arr = np.asarray(img, dtype=np.float32)
        scale = random.uniform(0.1, 5.0 + 10.0 * difficulty) * 1.73
        arr += np.random.uniform(-scale, scale, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if random.random() < 0.35 * difficulty:
        w, h = img.size
        small = img.resize((max(8, int(w * random.uniform(0.55, 0.85))), h), Image.Resampling.BILINEAR)
        img = small.resize((w, h), Image.Resampling.BILINEAR)
    return img


def random_plate_label():
    province = random.choice(PROVINCES)
    kind = random.choices(["car", "motorbike", "special"], weights=[0.52, 0.34, 0.14], k=1)[0]
    if kind == "motorbike":
        mid = random.choice(LETTERS) + str(random.randint(1, 9))
        tail = f"{random.randint(0, 99999):05d}"
    elif kind == "special":
        mid = random.choice(SPECIAL_MIDS)
        tail = f"{random.randint(0, 99999):05d}"
    else:
        mid = random.choice(LETTERS)
        if random.random() < 0.28:
            mid += random.choice(LETTERS)
        tail = f"{random.randint(0, 99999):05d}"
    if random.random() < 0.08:
        tail = str(random.randint(1000, 9999))
    return f"{province}{mid} {tail}"


class SyntheticPlate:
    def __init__(self, sizes, charset):
        self.sizes = sizes
        self.charset = set(charset)
        self.fonts = self._find_fonts()
        self.difficulty = 0.15
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
        fonts = [p for p in candidates if Path(p).exists()]
        return fonts or [None]

    def set_epoch(self, epoch, epochs):
        self.difficulty = min(1.0, 0.10 + 0.90 * epoch / max(1, epochs))

    def render(self, size=None):
        img_h, img_w = size or random.choice(self.sizes)
        label = random_plate_label()
        label = "".join(c for c in label if c in self.charset)
        two_line = random.random() < 0.22
        canvas = self._background(img_h, img_w)
        draw = ImageDraw.Draw(canvas)
        border = random.randint(12, 45)
        draw.rounded_rectangle([1, 1, img_w - 2, img_h - 2], radius=3, outline=(border, border, border), width=random.choice([1, 1, 2]))
        if random.random() < 0.35:
            self._draw_screws(draw, img_h, img_w)
        if two_line and " " in label:
            top, bottom = label.split(" ", 1)
            self._draw_centered(draw, top, img_w, img_h, y_frac=0.16, size_frac=0.42)
            self._draw_centered(draw, bottom, img_w, img_h, y_frac=0.51, size_frac=0.43)
        else:
            self._draw_centered(draw, label, img_w, img_h, y_frac=0.20, size_frac=random.uniform(0.52, 0.70))
        return augment(canvas, self.difficulty), label

    def _background(self, img_h, img_w):
        bg = random.randint(225, 255)
        if random.random() < 0.45:
            phase = random.random() * math.pi
            x = np.arange(img_w, dtype=np.float32)
            sin_wave = bg + 10.0 * np.sin(x / 13.0 + phase)
            shades = np.clip(sin_wave, 210, 255).astype(np.uint8)
            arr = np.tile(shades, (img_h, 1))
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
            canvas = Image.fromarray(arr_rgb)
        else:
            canvas = Image.new("RGB", (img_w, img_h), (bg, bg, bg))
        if random.random() < 0.25:
            tint = random.choice([(235, 240, 255), (245, 255, 245), (255, 248, 235)])
            overlay = Image.new("RGB", (img_w, img_h), tint)
            canvas = Image.blend(canvas, overlay, alpha=random.uniform(0.05, 0.18))
        return canvas

    def _font(self, font_size):
        font_path = random.choice(self.fonts)
        if not font_path:
            return ImageFont.load_default()
        key = (font_path, font_size)
        if key not in self._font_cache:
            self._font_cache[key] = ImageFont.truetype(font_path, font_size)
        return self._font_cache[key]

    def _draw_centered(self, draw, text, img_w, img_h, y_frac, size_frac):
        cache_key = (text, img_h, int(size_frac * 100))
        if cache_key in self._size_cache:
            fs_found = self._size_cache[cache_key]
        else:
            fs_found = 8
            for font_size in range(int(img_h * size_frac), 7, -1):
                font = self._font(font_size)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if tw <= img_w * 0.92 and th <= img_h * 0.72:
                    fs_found = font_size
                    break
            self._size_cache[cache_key] = fs_found
        font = self._font(fs_found)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = int((img_w - tw) / 2 + random.randint(-4, 4))
        y = int(img_h * y_frac + random.randint(-2, 2))
        color = random.randint(0, 35)
        if random.random() < 0.18:
            draw.text((x + 1, y + 1), text, fill=(150, 150, 150), font=font)
        draw.text((x, y), text, fill=(color, color, color), font=font)

    def _draw_screws(self, draw, img_h, img_w):
        for x in (int(img_w * 0.10), int(img_w * 0.90)):
            y = random.randint(int(img_h * 0.20), int(img_h * 0.80))
            r = max(1, img_h // 18)
            shade = random.randint(120, 210)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade, shade), outline=(60, 60, 60))


class PlateDataset(Dataset):
    def __init__(self, rows, codec, sizes, train=False, synthetic_ratio=0):
        self.rows = rows
        self.codec = codec
        self.sizes = sizes
        self.train = train
        self.synthetic_ratio = synthetic_ratio
        self.synthetic = SyntheticPlate(sizes, codec.chars)
        self.epoch = 1
        self.epochs = 1

    def set_epoch(self, epoch, epochs):
        self.epoch = epoch
        self.epochs = epochs
        self.synthetic.set_epoch(epoch, epochs)

    def __len__(self):
        ratio = self.synthetic_ratio if self.train else 0
        return len(self.rows) * (1 + ratio)

    def __getitem__(self, idx):
        use_synth = self.train and idx >= len(self.rows)
        size = random.choice(self.sizes) if self.train else self.sizes[-1]
        img_h, img_w = size
        if use_synth:
            img, label = self.synthetic.render(size=size)
        else:
            path, label = self.rows[idx % len(self.rows)]
            img = Image.open(path)
            img = resize_pad(img, img_h, img_w)
            if self.train:
                difficulty = min(1.0, 0.10 + 0.90 * self.epoch / max(1, self.epochs))
                img = augment(img, difficulty)
        return pil_to_tensor(img), label


class ParallelCollate:
    def __init__(self, codec):
        self.codec = codec

    def __call__(self, batch):
        images, labels = zip(*batch)
        max_h = max(x.shape[1] for x in images)
        max_w = max(x.shape[2] for x in images)
        padded = []
        for x in images:
            pad_h = max_h - x.shape[1]
            pad_w = max_w - x.shape[2]
            padded.append(F.pad(x, (0, pad_w, 0, pad_h), value=1.0))
        targets, target_lengths = self.codec.encode(labels)
        return torch.stack(padded), list(labels), targets, target_lengths


class ConvStem(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, 2, 1), nn.BatchNorm2d(dim), nn.GELU(),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=2)
        return x.permute(2, 0, 1)


class ScratchCTCRecognizer(nn.Module):
    def __init__(self, num_classes, dim=256, layers=8, heads=8, dropout=0.1, max_len=512):
        super().__init__()
        self.stem = ConvStem(dim)
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
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        seq = self.stem(x)
        seq = seq + self.pos[: seq.size(0)]
        seq = self.encoder(seq)
        seq = self.norm(seq)
        return self.head(seq)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


@dataclass
class Metrics:
    loss: float
    exact: float
    exact_rules: float
    cer: float


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + (ca != cb))
        dp = ndp
    return dp[-1]


def normalize_plate_prediction(text: str):
    text = text.upper().replace(" ", "")
    text = "".join(c for c in text if c.isalnum() or c == "Đ")
    if len(text) < 5:
        return text
    candidates = []
    for start_idx in (0, 1):
        for tail_len in (5, 4):
            if len(text) <= start_idx + tail_len + 2:
                continue
            prefix = text[start_idx:-tail_len]
            tail = text[-tail_len:].translate(NUMERIC_FIX)
            province = prefix[:2].translate(NUMERIC_FIX)
            mid = prefix[2:]
            if not province.isdigit() or not tail.isdigit() or not mid:
                continue
            mid_variants = {mid}
            if mid[0].isdigit():
                mid_variants.add(mid[0].translate(LETTER_FIX) + mid[1:])
            if len(mid) >= 2 and mid[1].isdigit() and not mid[1:].isdigit():
                mid_variants.add(mid[0] + mid[1].translate(LETTER_FIX) + mid[2:])
            for m in mid_variants:
                if re.fullmatch(r"[A-ZĐ]{1,3}\d?", m) or re.fullmatch(r"(LD|MĐ|TĐ|HC|AB|NG|QT)", m):
                    cand = f"{province}{m} {tail}"
                    score = abs(len(cand) - len(text) - 1) + start_idx
                    candidates.append((score, cand))
    if candidates:
        return min(candidates, key=lambda x: (x[0], x[1]))[1]
    if len(text) > 5:
        return text[:-5] + " " + text[-5:]
    return text


def run_epoch(model, loader, codec, device, optimizer=None, scaler=None, scheduler=None, amp=False, ema=None):
    train = optimizer is not None
    model.train(train)
    total_loss = total_exact = total_exact_rules = total_edits = total_chars = total = 0
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    for images, labels, targets, target_lengths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=-1)
                input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                if ema:
                    ema.update(model)
                if scheduler:
                    scheduler.step()

        preds = codec.decode(logits.detach())
        bs = len(labels)
        total_loss += loss.item() * bs
        for pred, label in zip(preds, labels):
            if train:
                raw_ok = pred == label
                total_exact += int(raw_ok)
                total_exact_rules += int(raw_ok)
            else:
                pred_rules = normalize_plate_prediction(pred)
                total_exact += int(pred == label)
                total_exact_rules += int(pred_rules == label)
                total_edits += edit_distance(pred_rules, label)
                total_chars += max(1, len(label))
        total += bs
    if torch.distributed.is_initialized():
        import torch.distributed as dist_mod
        # Gather all scalar stats across GPUs
        stats = {
            "total_loss": total_loss,
            "total_exact": total_exact,
            "total_exact_rules": total_exact_rules,
            "total_edits": total_edits,
            "total_chars": total_chars,
            "total": total
        }
        for k, v in stats.items():
            val_tensor = torch.tensor(v, dtype=torch.float32, device=device)
            dist_mod.all_reduce(val_tensor, op=dist_mod.ReduceOp.SUM)
            stats[k] = val_tensor.item()
        
        total_loss = stats["total_loss"]
        total_exact = stats["total_exact"]
        total_exact_rules = stats["total_exact_rules"]
        total_edits = stats["total_edits"]
        total_chars = stats["total_chars"]
        total = stats["total"]

    val_edit = total_edits / max(1, total_chars) if not train else 0.0
    return Metrics(total_loss / total, total_exact / total, total_exact_rules / total, val_edit)


class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup:
            lr = self.base_lr * self.step_num / max(1, self.warmup)
        else:
            p = (self.step_num - self.warmup) / max(1, self.total_steps - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * min(1, p)))
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def unwrap(model):
    import torch.nn.parallel as parallel_mod
    if isinstance(model, (nn.DataParallel, parallel_mod.DistributedDataParallel)):
        return model.module
    return model


def save_checkpoint(path, model, optimizer, epoch, best_exact, args, charset, ema=None, epochs_no_improve=0):
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_exact": best_exact,
        "epochs_no_improve": epochs_no_improve,
        "args": vars(args),
        "charset": charset,
    }
    if ema:
        payload["ema_model"] = ema.module.state_dict()
    torch.save(payload, path)


def write_predictions(path, model, loader, codec, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = codec.decode(logits)
            rows.extend((label, pred, normalize_plate_prediction(pred)) for label, pred in zip(labels, preds))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "prediction", "rule_prediction", "correct", "rule_correct"])
        for label, pred, rule_pred in rows:
            writer.writerow([label, pred, rule_pred, int(label == pred), int(label == rule_pred)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--synthetic-ratio", type=int, default=4)
    parser.add_argument("--img-h", type=int, default=48)
    parser.add_argument("--img-w", type=int, default=192)
    parser.add_argument("--multi-sizes", default="40x160,48x192,56x224")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1800)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

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

    worker_seed = args.seed + local_rank
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    charset = (data_root / "charset.txt").read_text(encoding="utf-8").strip("\n")
    if is_master:
        (out_dir / "charset.txt").write_text(charset + "\n", encoding="utf-8")
    codec = Codec(charset)

    train_rows = read_csv(data_root / "train.csv")
    val_rows = read_csv(data_root / "val.csv")
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.val_limit:
        val_rows = val_rows[: args.val_limit]
    train_sizes = parse_sizes(args.multi_sizes) if args.multi_sizes else [(args.img_h, args.img_w)]
    val_size = [(args.img_h, args.img_w)]
    train_ds = PlateDataset(train_rows, codec, train_sizes, True, args.synthetic_ratio)
    val_ds = PlateDataset(val_rows, codec, val_size, False, 0)
    collate_fn = ParallelCollate(codec)

    if ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
        batch_size_per_gpu = max(1, args.batch_size // world_size)
    else:
        train_sampler = None
        val_sampler = None
        batch_size_per_gpu = args.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
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
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    model = ScratchCTCRecognizer(codec.num_classes, args.model_dim, args.layers, args.heads, args.dropout).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = WarmupCosine(optimizer, args.lr, args.warmup_steps, args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    start_epoch, best_exact, epochs_no_improve = 1, 0.0, 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ema and "ema_model" in ckpt:
            ema.module.load_state_dict(ckpt["ema_model"])
        start_epoch = ckpt["epoch"] + 1
        best_exact = ckpt.get("best_exact", 0.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)

    hist = out_dir / "history.csv"
    if is_master and not hist.exists():
        hist.write_text(
            "epoch,train_loss,train_exact,train_exact_rules,train_cer,val_loss,val_exact,val_exact_rules,val_cer,ema_val_exact_rules,seconds\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp and train_sampler:
            train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch, args.epochs)
        t0 = time.time()
        train_m = run_epoch(model, train_loader, codec, device, optimizer, scaler, scheduler, args.amp, ema)
        eval_model = ema.module if ema else model
        val_m = run_epoch(eval_model, val_loader, codec, device)
        raw_val_m = run_epoch(model, val_loader, codec, device) if ema else val_m
        
        score = val_m.exact_rules
        is_best = score >= best_exact
        if is_best:
            best_exact = score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        stop_training = False
        patience = args.patience
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
                f"{epoch},{train_m.loss:.6f},{train_m.exact:.6f},{train_m.exact_rules:.6f},{train_m.cer:.6f},"
                f"{raw_val_m.loss:.6f},{raw_val_m.exact:.6f},{raw_val_m.exact_rules:.6f},{raw_val_m.cer:.6f},"
                f"{val_m.exact_rules:.6f},{seconds:.1f}"
            )
            print(line, flush=True)
            with hist.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_exact, args, charset, ema, epochs_no_improve)
            if is_best:
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_exact, args, charset, ema, epochs_no_improve)
                write_predictions(out_dir / "predictions_val.csv", eval_model, val_loader, codec, device)


if __name__ == "__main__":
    main()
