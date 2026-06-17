"""
Nhan dien bien so bang ensemble OCR tot nhat trong workspace nay.

Chay 1 anh:
    py recognize_plate.py --image path/to/plate.jpg

Chay ca folder:
    py recognize_plate.py --input-dir ../ocr_dataset/test --output-csv predictions.csv

Chay danh gia neu co label:
    py recognize_plate.py --input-dir ../ocr_dataset/test --labels ../ocr_dataset/test_labels.txt --evaluate
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
OCR_CODE_DIR = ROOT_DIR / "src" / "ocr_code"
if str(OCR_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OCR_CODE_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_plate_text(text: str) -> str:
    text = str(text or "").upper().strip()
    return re.sub(r"[^0-9A-ZĐ]", "", text)


def plate_validity_score(text: str) -> int:
    """
    Cham diem hinh dang bien so Viet Nam.
    Diem cao hon se duoc uu tien trong ensemble.
    """
    plate = normalize_plate_text(text)
    digits = sum(ch.isdigit() for ch in plate)
    letters = len(plate) - digits

    score = 0
    if 7 <= len(plate) <= 10:
        score += 2
    if re.match(r"^[0-9]{2}", plate):
        score += 2
    if re.match(r"^[0-9]{2}[A-ZĐ]", plate):
        score += 2
    if re.match(r"^[0-9]{2}[A-ZĐ][0-9A-ZĐ]?[0-9]{4,5}$", plate):
        score += 4
    if digits >= 6:
        score += 1
    if 1 <= letters <= 3:
        score += 1
    return score


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def character_accuracy(pred: str, gt: str) -> float:
    pred_norm = normalize_plate_text(pred)
    gt_norm = normalize_plate_text(gt)
    max_len = max(len(pred_norm), len(gt_norm))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - levenshtein_distance(pred_norm, gt_norm) / max_len)


def full_plate_correct(pred: str, gt: str) -> bool:
    return normalize_plate_text(pred) == normalize_plate_text(gt)


def normalize_plate_text(text: str) -> str:
    """Ban normalize moi, dung Unicode escape de tranh loi encoding chu D co gach."""
    text = str(text or "").upper().strip()
    return re.sub(r"[^0-9A-Z\u0110]", "", text)


def plate_validity_score(text: str) -> int:
    plate = normalize_plate_text(text)
    digits = sum(ch.isdigit() for ch in plate)
    letters = len(plate) - digits

    score = 0
    if 7 <= len(plate) <= 10:
        score += 2
    if re.match(r"^[0-9]{2}", plate):
        score += 2
    if re.match(r"^[0-9]{2}[A-Z\u0110]", plate):
        score += 2
    if re.match(r"^[0-9]{2}([A-Z\u0110]|[A-Z\u0110][0-9])[0-9]{4,5}$", plate):
        score += 5
    if has_valid_two_letter_series(plate):
        score += 5
    if re.match(r"^[0-9]{2}[A-Z\u0110]{2}[0-9][0-9]{4,5}$", plate):
        score += 5
    if digits >= 6:
        score += 1
    if 1 <= letters <= 3:
        score += 1
    return score


SPECIAL_SERIES = {"LD", "CD", "KT", "MD", "TD", "HC", "NG", "QT"}
VALID_TWO_LETTER_SERIES = SPECIAL_SERIES | {
    "AA", "AB", "AE", "AN", "BD", "BF", "DA", "FA", "FC", "FK",
    "FN", "LA", "NA", "RA", "SA", "TA", "XA",
}
OLD_STYLE_PREFIXES = {
    "AD", "QH", "KP", "QA", "BK", "BT", "TK", "TH",
    "KV", "AT", "TN", "HN", "KK", "TM", "QC",
}


def is_old_style_plate(text: str) -> bool:
    return bool(re.match(r"^[A-Z\u0110]{2}[0-9]{4}$", normalize_plate_text(text)))


def old_style_prefix(text: str) -> str:
    return normalize_plate_text(text)[:2]


def has_valid_two_letter_series(text: str) -> bool:
    plate = normalize_plate_text(text)
    match = re.match(r"^[0-9]{2}([A-Z\u0110]{2})[0-9]?[0-9]{4,5}$", plate)
    return bool(match and match.group(1) in VALID_TWO_LETTER_SERIES)


def is_a_series_confusion(text: str) -> Optional[str]:
    """
    Sua loi hay gap tren bien series dac biet:
    train2 co xu huong doc AA/TA/LA thanh A4/T4/L4.
    Chi ap dung cho A/T/L de tranh lam sai cac bien hop le nhu B4, F4.
    """
    plate = normalize_plate_text(text)
    match = re.match(r"^([0-9]{2})([ATL])4([0-9]{5})$", plate)
    if not match:
        return None
    return match.group(1) + match.group(2) + "A" + match.group(3)


def parse_plate_parts(text: str) -> Tuple[str, str]:
    plate = normalize_plate_text(text)
    patterns = [
        r"^([0-9]{2}[A-Z\u0110])([0-9]{5})$",
        r"^([0-9]{2}[A-Z\u0110])([0-9]{4})$",
        r"^([0-9]{2}[A-Z\u0110][0-9])([0-9]{5})$",
        r"^([0-9]{2}[A-Z\u0110][0-9])([0-9]{4})$",
        r"^([0-9]{2}[A-Z\u0110]{2}[0-9])([0-9]{5})$",
        r"^([0-9]{2}[A-Z\u0110]{2}[0-9])([0-9]{4})$",
        r"^([0-9]{2}[A-Z\u0110]{2})([0-9]{5})$",
        r"^([0-9]{2}[A-Z\u0110]{2})([0-9]{4})$",
    ]
    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            return match.group(1), match.group(2)
    match = re.match(r"^(.*?)([0-9]{3,6})$", plate)
    if match:
        return match.group(1), match.group(2)
    return plate, ""


def has_separator(text: str) -> bool:
    return any(ch in str(text or "") for ch in ".-· ")


def is_special_prefix_extension(train_prefix: str, train1_prefix: str) -> bool:
    return (
        bool(train_prefix)
        and train1_prefix.startswith(train_prefix)
        and len(train1_prefix) > len(train_prefix)
        and train1_prefix[2:] in SPECIAL_SERIES
    )


def is_suspicious_prefix_extension(train_prefix: str, train1_prefix: str) -> bool:
    return (
        bool(train_prefix)
        and train1_prefix.startswith(train_prefix)
        and len(train1_prefix) > len(train_prefix)
        and train1_prefix[2:] not in SPECIAL_SERIES
    )


def prefixes_compatible(train_prefix: str, train1_prefix: str) -> bool:
    return bool(
        train_prefix
        and train1_prefix
        and (train_prefix == train1_prefix or is_special_prefix_extension(train_prefix, train1_prefix))
    )


def patch_missing_tail(train_serial: str, train1_serial: str) -> str:
    if len(train1_serial) <= len(train_serial):
        return train_serial
    if train1_serial.startswith(train_serial):
        return train1_serial
    if len(train_serial) == 4 and len(train1_serial) == 5:
        candidates = [
            train_serial[:idx] + train1_serial[idx] + train_serial[idx:]
            for idx in range(5)
        ]
        candidates.append(train1_serial)
        return max(
            candidates,
            key=lambda item: (
                train1_serial[:3] == train_serial[:3] and item == train1_serial,
                -levenshtein_distance(item, train1_serial),
                item.endswith(train_serial[-2:]),
            ),
        )
    missing = len(train1_serial) - len(train_serial)
    return train_serial + train1_serial[-missing:]


def merge_train2_train1(train2_text: str, train1_text: str) -> str:
    train2_norm = normalize_plate_text(train2_text)
    train1_norm = normalize_plate_text(train1_text)

    if is_old_style_plate(train2_norm) and not is_old_style_plate(train1_norm):
        return train2_norm
    if (
        is_old_style_plate(train2_norm)
        and is_old_style_plate(train1_norm)
        and old_style_prefix(train2_norm) != old_style_prefix(train1_norm)
        and old_style_prefix(train2_norm) in OLD_STYLE_PREFIXES
    ):
        return train2_norm

    train_prefix, train_serial = parse_plate_parts(train2_norm)
    train1_prefix, train1_serial = parse_plate_parts(train1_norm)

    candidates = [train2_norm]
    a_series_fix = is_a_series_confusion(train2_norm)
    if a_series_fix:
        candidates.append(a_series_fix)

    if is_old_style_plate(train2_norm) and is_old_style_plate(train1_norm) and has_separator(train1_text):
        candidates.append(train1_norm)
    elif not (
        plate_validity_score(train2_norm) >= 10
        and is_suspicious_prefix_extension(train_prefix, train1_prefix)
        and train1_serial == train_serial
    ):
        candidates.append(train1_norm)

    if (
        train_prefix == train1_prefix
        and len(train_serial) == 5
        and len(train1_serial) == 5
        and train_serial != train1_serial
        and has_separator(train1_text)
        and (
            train_serial[:2] == train1_serial[:2]
            or train_serial[-2:] == train1_serial[-2:]
            or levenshtein_distance(train_serial, train1_serial) <= 2
        )
    ):
        candidates.append(train_prefix + train1_serial)

    if re.match(r"^[0-9]{2}[A-Z\u0110]", train_prefix) and len(train_serial) < 5 and len(train1_serial) == 5:
        if prefixes_compatible(train_prefix, train1_prefix):
            out_prefix = train1_prefix if is_special_prefix_extension(train_prefix, train1_prefix) else train_prefix
            candidates.append(out_prefix + train1_serial)
        elif not train1_prefix and has_separator(train1_text):
            candidates.append(train_prefix + patch_missing_tail(train_serial, train1_serial))

    if (
        re.match(r"^[0-9]{2}[A-Z\u0110]", train_prefix)
        and is_special_prefix_extension(train_prefix, train1_prefix)
        and len(train1_serial) in (4, 5)
    ):
        candidates.append(train1_prefix + train1_serial)

    return max(
        candidates,
        key=lambda item: (
            is_old_style_plate(item) and item == train1_norm and has_separator(train1_text),
            plate_validity_score(item),
            item != train2_norm and train_prefix == train1_prefix and has_separator(train1_text),
            len(item) in (8, 9),
            len(item),
            item == train2_norm,
        ),
    )


def rgb_to_hsv_arrays(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    mask = diff > 1e-6
    red_hue = ((g - b) / (diff + 1e-6)) % 6
    green_hue = ((b - r) / (diff + 1e-6)) + 2
    blue_hue = ((r - g) / (diff + 1e-6)) + 4
    hue = np.where((mx == r) & mask, red_hue, hue)
    hue = np.where((mx == g) & mask, green_hue, hue)
    hue = np.where((mx == b) & mask, blue_hue, hue)
    hue *= 60.0
    sat = np.where(mx == 0, 0, diff / (mx + 1e-6))
    return hue, sat, mx


def classify_plate_color(image_path: Path) -> Tuple[str, Dict[str, float]]:
    """
    Phan loai mau nen bien so tu anh crop.
    Dung de gan ngu canh rule, khong dung nhu dieu kien tuyet doi.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((180, 72))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hue, sat, val = rgb_to_hsv_arrays(arr)

    yellow = ((hue >= 35) & (hue <= 75) & (sat > 0.25) & (val > 0.35)).mean()
    blue = ((hue >= 170) & (hue <= 260) & (sat > 0.22) & (val > 0.25)).mean()
    red = (((hue <= 18) | (hue >= 340)) & (sat > 0.25) & (val > 0.30)).mean()
    white = ((sat < 0.28) & (val > 0.62)).mean()

    scores = {
        "white": float(white),
        "yellow": float(yellow),
        "blue": float(blue),
        "red": float(red),
    }
    color = max(scores, key=scores.get)
    if scores[color] < 0.08:
        return "unknown", scores
    if scores["white"] > 0.22 and max(scores["yellow"], scores["blue"], scores["red"]) < 0.16:
        return "white", scores
    return color, scores


def should_use_detector_fallback(base: Candidate, detected: Candidate) -> bool:
    base_len = len(base.norm_text)
    detected_len = len(detected.norm_text)
    base_score = plate_validity_score(base.norm_text)
    detected_score = plate_validity_score(detected.norm_text)

    if detected_len < 6:
        return False
    if is_old_style_plate(base.norm_text) and not is_old_style_plate(detected.norm_text):
        return False

    if detected_score > base_score:
        return True
    if base_len < 8 and detected_len >= 8 and detected_score >= base_score:
        return True
    if base_score < 10 and detected_score >= 10:
        return True
    return False


def list_images(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def load_labels(path: Optional[Path]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not path or not path.exists():
        return labels

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row.get("image_path") or row.get("filename") or row.get("image")
                label = row.get("label") or row.get("text") or row.get("plate")
                if image_path and label is not None:
                    labels[Path(image_path).name] = label.strip()
        return labels

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            image_name, label = line.split("\t", 1)
        elif "," in line:
            image_name, label = line.split(",", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            image_name, label = parts
        labels[Path(image_name).name] = label.strip()
    return labels


class TorchCTCPredictor:
    def __init__(self, name: str, checkpoint_path: Path):
        from infer import load_best_model, load_scratch_model  # type: ignore
        from train_best import normalize_plate_prediction, pil_to_tensor, resize_pad  # type: ignore

        from PIL import Image

        self.name = name
        self.Image = Image
        self.normalize_plate_prediction = normalize_plate_prediction
        self.pil_to_tensor = pil_to_tensor
        self.resize_pad = resize_pad
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "config" in ckpt:
            model, codec, img_h, img_w, is_best = load_best_model(ckpt)
        else:
            model, codec, img_h, img_w, is_best = load_scratch_model(ckpt)

        self.model = model.to(self.device).eval()
        self.codec = codec
        self.img_h = img_h
        self.img_w = img_w
        self.use_rule_normalization = is_best

    @torch.no_grad()
    def predict(self, image_path: Path) -> str:
        img = self.Image.open(image_path).convert("RGB")
        x = self.pil_to_tensor(self.resize_pad(img, self.img_h, self.img_w))
        x = x.unsqueeze(0).to(self.device)
        output = self.model(x)
        logits = output[0] if isinstance(output, tuple) else output
        pred = self.codec.decode(logits)[0]
        if self.use_rule_normalization:
            pred = self.normalize_plate_prediction(pred)
        return str(pred)


class PaddleOCRPredictor:
    def __init__(self, name: str, model_dir: Path):
        from paddleocr import PaddleOCR  # type: ignore

        self.name = name
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            det=False,
            rec=True,
            rec_model_dir=str(model_dir),
            use_gpu=torch.cuda.is_available(),
            show_log=False,
        )

    def predict(self, image_path: Path) -> str:
        result = self.ocr.ocr(str(image_path), det=False, cls=False)
        if not result:
            return ""
        first = result[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, tuple) and first:
            return str(first[0])
        return str(first)


@dataclass
class Candidate:
    model: str
    text: str
    norm_text: str
    validity: int
    time_ms: float
    error: str = ""


class PlateOCREnsemble:
    def __init__(self):
        self.predictors = [
            PaddleOCRPredictor("train1_paddleocr", ROOT_DIR / "models" / "train1"),
            TorchCTCPredictor("train2_best_ema", ROOT_DIR / "models" / "train2" / "best_ema.pt"),
        ]
        # Uu tien khi diem bang nhau: train2 la model moi, train1 la model bo tro.
        self.tie_priority = {
            "merged_train2_train1": 3,
            "train2_best_ema": 2,
            "train1_paddleocr": 1,
        }

    def predict_all(self, image_path: Path) -> List[Candidate]:
        candidates: List[Candidate] = []
        for predictor in self.predictors:
            start = time.perf_counter()
            try:
                text = predictor.predict(image_path)
                error = ""
            except Exception as exc:
                text = ""
                error = str(exc)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            candidates.append(
                Candidate(
                    model=predictor.name,
                    text=text,
                    norm_text=normalize_plate_text(text),
                    validity=plate_validity_score(text),
                    time_ms=elapsed_ms,
                    error=error,
                )
            )
        return candidates

    def choose_best(self, candidates: List[Candidate]) -> Candidate:
        by_model = {cand.model: cand for cand in candidates}
        train2 = by_model.get("train2_best_ema")
        train1 = by_model.get("train1_paddleocr")
        ranked_candidates = list(candidates)
        if train2 is not None and train1 is not None:
            merged_text = merge_train2_train1(train2.text, train1.text)
            ranked_candidates.append(
                Candidate(
                    model="merged_train2_train1",
                    text=merged_text,
                    norm_text=normalize_plate_text(merged_text),
                    validity=plate_validity_score(merged_text),
                    time_ms=0.0,
                )
            )

        # Neu co 2 model trung normalized text, tang diem consensus.
        counts: Dict[str, int] = {}
        for cand in ranked_candidates:
            if cand.norm_text:
                counts[cand.norm_text] = counts.get(cand.norm_text, 0) + 1

        def rank(cand: Candidate) -> Tuple[int, int, int, int]:
            consensus_bonus = 3 if counts.get(cand.norm_text, 0) >= 2 else 0
            return (
                cand.validity + consensus_bonus,
                self.tie_priority.get(cand.model, 0),
                len(cand.norm_text),
                -1 if cand.error else 0,
            )

        return max(ranked_candidates, key=rank)

    def predict(self, image_path: Path) -> Tuple[Candidate, List[Candidate]]:
        candidates = self.predict_all(image_path)
        return self.choose_best(candidates), candidates


class PlateDetector:
    """YOLO detector de tim va crop vung bien so truoc khi OCR."""

    def __init__(self, weights_path: Path, conf: float = 0.25, margin: float = 0.08):
        if not weights_path.exists():
            raise FileNotFoundError(f"Khong tim thay detector: {weights_path}")
        from PIL import Image
        from ultralytics import YOLO  # type: ignore

        self.Image = Image
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.margin = margin

    def crop_best_plate(self, image_path: Path, output_dir: Path) -> Tuple[Path, dict]:
        img = self.Image.open(image_path).convert("RGB")
        width, height = img.size
        results = self.model.predict(str(image_path), conf=self.conf, verbose=False)

        best = None
        best_score = -1.0
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = box.xyxy[0].detach().cpu().tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                score = conf * max(area, 1.0)
                if score > best_score:
                    best_score = score
                    best = (x1, y1, x2, y2, conf)

        if best is None:
            return image_path, {
                "detected": False,
                "det_conf": "",
                "crop_path": str(image_path),
                "bbox": "",
            }

        x1, y1, x2, y2, conf = best
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = box_w * self.margin
        pad_y = box_h * self.margin
        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))

        crop = img.crop((left, top, right, bottom))
        output_dir.mkdir(parents=True, exist_ok=True)
        crop_path = output_dir / f"{image_path.stem}_plate{image_path.suffix}"
        crop.save(crop_path)

        return crop_path, {
            "detected": True,
            "det_conf": f"{conf:.4f}",
            "crop_path": str(crop_path),
            "bbox": f"{left},{top},{right},{bottom}",
        }


def write_rows_csv(rows: List[dict], output_csv: Path) -> None:
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    ensemble = PlateOCREnsemble()
    detector = None
    crop_dir = None
    if args.detect_plate or args.detect_fallback:
        detector = PlateDetector(
            weights_path=Path(args.detector),
            conf=args.det_conf,
            margin=args.det_margin,
        )
        crop_dir = Path(args.crop_dir)

    if args.image:
        images = [Path(args.image)]
    else:
        images = list_images(Path(args.input_dir))

    labels = load_labels(Path(args.labels)) if args.labels else {}
    rows: List[dict] = []
    char_scores: List[float] = []
    plate_scores: List[int] = []
    total_times: List[float] = []

    for image_path in images:
        ocr_image_path = image_path
        det_info = {
            "detected": "",
            "det_conf": "",
            "crop_path": "",
            "bbox": "",
        }
        if detector is not None and crop_dir is not None and args.detect_plate:
            ocr_image_path, det_info = detector.crop_best_plate(image_path, crop_dir)

        plate_color, color_scores = classify_plate_color(ocr_image_path)
        best, candidates = ensemble.predict(ocr_image_path)

        fallback_info = {
            "used": False,
            "prediction": "",
            "crop_path": "",
        }
        if detector is not None and crop_dir is not None and args.detect_fallback and not args.detect_plate:
            detected_image_path, fallback_det_info = detector.crop_best_plate(image_path, crop_dir)
            detected_best, detected_candidates = ensemble.predict(detected_image_path)
            if should_use_detector_fallback(best, detected_best):
                best = Candidate(
                    model="detector_fallback_" + detected_best.model,
                    text=detected_best.text,
                    norm_text=detected_best.norm_text,
                    validity=detected_best.validity,
                    time_ms=detected_best.time_ms,
                    error=detected_best.error,
                )
                candidates.extend(detected_candidates)
                ocr_image_path = detected_image_path
                det_info = fallback_det_info
                plate_color, color_scores = classify_plate_color(ocr_image_path)
                fallback_info["used"] = True
            fallback_info["prediction"] = detected_best.text
            fallback_info["crop_path"] = str(detected_image_path)

        total_time = sum(c.time_ms for c in candidates)
        total_times.append(total_time)
        gt = labels.get(image_path.name, "")

        if gt:
            char_scores.append(character_accuracy(best.text, gt))
            plate_scores.append(1 if full_plate_correct(best.text, gt) else 0)

        row = {
            "image": image_path.name,
            "ocr_image": str(ocr_image_path),
            "plate_detected": det_info["detected"],
            "det_conf": det_info["det_conf"],
            "bbox": det_info["bbox"],
            "plate_color": plate_color,
            "detector_fallback_used": fallback_info["used"],
            "detector_fallback_prediction": fallback_info["prediction"],
            "detector_fallback_crop": fallback_info["crop_path"],
            "color_white": f"{color_scores['white']:.4f}",
            "color_yellow": f"{color_scores['yellow']:.4f}",
            "color_blue": f"{color_scores['blue']:.4f}",
            "color_red": f"{color_scores['red']:.4f}",
            "prediction": best.text,
            "prediction_normalized": best.norm_text,
            "selected_model": best.model,
            "validity_score": best.validity,
            "ground_truth": gt,
            "char_acc": character_accuracy(best.text, gt) if gt else "",
            "full_plate_correct": full_plate_correct(best.text, gt) if gt else "",
            "total_time_ms": f"{total_time:.3f}",
        }
        for cand in candidates:
            prefix = cand.model
            row[f"{prefix}_pred"] = cand.text
            row[f"{prefix}_score"] = cand.validity
            row[f"{prefix}_time_ms"] = f"{cand.time_ms:.3f}"
            row[f"{prefix}_error"] = cand.error
        rows.append(row)

        if args.image:
            print(f"Image: {image_path}")
            if detector is not None:
                print(f"Plate crop: {ocr_image_path}")
                print(f"Detector: detected={det_info['detected']} conf={det_info['det_conf']} bbox={det_info['bbox']}")
            if args.detect_fallback:
                print(f"Detector fallback used: {fallback_info['used']} | pred={fallback_info['prediction']}")
            print(f"Plate color: {plate_color} | scores={color_scores}")
            print(f"Prediction: {best.text}")
            print(f"Normalized: {best.norm_text}")
            print(f"Selected model: {best.model}")
            for cand in candidates:
                print(f"- {cand.model}: {cand.text} | score={cand.validity} | {cand.time_ms:.1f} ms")

    if args.output_csv:
        write_rows_csv(rows, Path(args.output_csv))
        print(f"[OK] Saved predictions: {Path(args.output_csv).resolve()}")

    if args.evaluate and labels:
        char_acc = sum(char_scores) / len(char_scores) if char_scores else 0.0
        plate_acc = sum(plate_scores) / len(plate_scores) if plate_scores else 0.0
        avg_time = sum(total_times) / len(total_times) if total_times else 0.0
        print("\n=== ENSEMBLE EVALUATION ===")
        print(f"Images: {len(images)}")
        print(f"Labeled images: {len(char_scores)}")
        print(f"Character Accuracy: {char_acc * 100:.2f}%")
        print(f"Full Plate Accuracy: {plate_acc * 100:.2f}%")
        print(f"Avg total inference time: {avg_time:.2f} ms/image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best OCR ensemble for Vietnamese license plates.")
    parser.add_argument("--image", help="Duong dan 1 anh bien so can nhan dien.")
    parser.add_argument("--input-dir", default="../ocr_dataset/test", help="Thu muc anh can chay.")
    parser.add_argument("--labels", help="File label .txt hoac .csv de danh gia.")
    parser.add_argument("--output-csv", default="ensemble_predictions.csv", help="File CSV luu prediction.")
    parser.add_argument("--evaluate", action="store_true", help="Tinh metric neu co labels.")
    parser.add_argument("--detect-plate", action="store_true", help="Detect va crop bien so truoc khi OCR.")
    parser.add_argument(
        "--detect-fallback",
        action="store_true",
        help="Chay them detector crop nhu phuong an phu; chi dung neu prediction crop tot hon theo rule an toan.",
    )
    parser.add_argument(
        "--detector",
        default=str(ROOT_DIR / "models" / "plate_detector" / "license_plate_detector.pt"),
        help="Duong dan YOLO detector bien so.",
    )
    parser.add_argument("--det-conf", type=float, default=0.25, help="Nguong confidence detector.")
    parser.add_argument("--det-margin", type=float, default=0.08, help="Margin crop quanh bien so.")
    parser.add_argument("--crop-dir", default="detected_plate_crops", help="Thu muc luu anh crop bien so.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

