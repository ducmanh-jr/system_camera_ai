import re
import sys
import uuid
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Postprocess functions (inlined – no dependency on giai_doan_doc_ki_tu)
# ---------------------------------------------------------------------------

LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0', 'I': '1',
    'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7',
}

DIGIT_TO_LETTER = {
    '0': 'D', '1': 'I', '2': 'Z', '5': 'S',
    '8': 'B', '6': 'G', '7': 'T',
}


def _clean_label(label: str) -> str:
    if not label:
        return ""
    label = label.upper()
    return re.sub(r'[^A-Z0-9\u0110]', '', label)


def _force_digit(char: str) -> str:
    return LETTER_TO_DIGIT.get(char, char)


def _force_letter(char: str) -> str:
    return DIGIT_TO_LETTER.get(char, char)


def correct_plate_characters(plate: str) -> str:
    """Applies Vietnamese license plate positional rules to correct OCR mis-transcriptions."""
    plate = _clean_label(plate)
    n = len(plate)
    if n < 4 or n > 11:
        return plate
    chars = list(plate)

    # Military / special plate: 2 letters + 4 digits
    is_military = False
    if n == 6:
        first_two_letters = all(c.isalpha() or c == '\u0110' or c in DIGIT_TO_LETTER for c in chars[:2])
        last_four_digits = all(c.isdigit() or c in LETTER_TO_DIGIT for c in chars[2:])
        if first_two_letters and last_four_digits:
            is_military = True

    if is_military:
        corrected = [
            _force_letter(chars[0]), _force_letter(chars[1]),
            _force_digit(chars[2]), _force_digit(chars[3]),
            _force_digit(chars[4]), _force_digit(chars[5]),
        ]
        return "".join(corrected)

    # Civilian plates: 2 digits province code
    chars[0] = _force_digit(chars[0])
    chars[1] = _force_digit(chars[1])

    if n >= 5 and (chars[2].isalpha() or chars[2] == '\u0110' or chars[2] in DIGIT_TO_LETTER) and (chars[3].isdigit() or chars[3] in LETTER_TO_DIGIT):
        chars[2] = _force_letter(chars[2])
        chars[3] = _force_digit(chars[3])
        for i in range(4, n):
            chars[i] = _force_digit(chars[i])
    elif n >= 6 and (chars[2].isalpha() or chars[2] == '\u0110' or chars[2] in DIGIT_TO_LETTER) and (chars[3].isalpha() or chars[3] == '\u0110' or chars[3] in DIGIT_TO_LETTER):
        chars[2] = _force_letter(chars[2])
        chars[3] = _force_letter(chars[3])
        for i in range(4, n):
            chars[i] = _force_digit(chars[i])
    else:
        chars[2] = _force_letter(chars[2])
        for i in range(3, n):
            chars[i] = _force_digit(chars[i])

    return "".join(chars)


def validate_vietnamese_plate(plate: str) -> bool:
    """Validates if a normalized plate fits standard Vietnamese plate regexes."""
    plate = _clean_label(plate)
    if re.match(r'^[A-Z\u0110]{2}\d{4}$', plate):
        return True
    if re.match(r'^\d{2}[A-Z\u0110]{1,2}\d{4,5}$', plate):
        return True
    if re.match(r'^\d{2}[A-Z\u0110]\d\d{4,5}$', plate):
        return True
    return False


# ---------------------------------------------------------------------------
# OCR Adapter – uses ONLY train_ocr ensemble
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    raw: str
    plate: str
    confidence: float
    valid: bool
    source: str


class VietnamesePlateOCR:
    def __init__(
        self,
        model_dir: Path | None = None,
        dict_path: Path | None = None,
        use_gpu: bool = False,
    ) -> None:
        self.use_gpu = use_gpu
        self.model_dir = ROOT_DIR / "train_ocr"
        self._ensemble: Any | None = None

    def _load(self) -> Any:
        if self._ensemble is not None:
            return self._ensemble

        # Add train_ocr/src and train_ocr/src/ocr_code to system path
        TRAIN_OCR_SRC = ROOT_DIR / "train_ocr" / "src"
        if str(TRAIN_OCR_SRC) not in sys.path:
            sys.path.insert(0, str(TRAIN_OCR_SRC))

        OCR_CODE_DIR = TRAIN_OCR_SRC / "ocr_code"
        if str(OCR_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(OCR_CODE_DIR))

        from pipeline import PlateOCREnsemble
        self._ensemble = PlateOCREnsemble()
        return self._ensemble

    @staticmethod
    def _stitch_if_two_line(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0:
            return image
        if w / h >= 1.4:
            return image
        half_h = h // 2
        if half_h <= 0:
            return image
        top = image[:half_h, :]
        bottom = image[half_h:, :]
        target_h = min(top.shape[0], bottom.shape[0])
        top = cv2.resize(top, (top.shape[1], target_h))
        bottom = cv2.resize(bottom, (bottom.shape[1], target_h))
        return np.concatenate((top, bottom), axis=1)

    def read_image(self, image_path: Path) -> OCRResult:
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            return OCRResult("", "", 0.0, False, "train_ocr")

        crop = self._stitch_if_two_line(image)

        # Save stitched crop to a temp file because the ensemble expects a Path
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"temp_plate_{uuid.uuid4().hex[:8]}.png"

        raw = ""
        conf = 0.0
        try:
            cv2.imwrite(str(temp_path), crop)
            ensemble = self._load()
            best_cand, candidates = ensemble.predict(temp_path)
            raw = best_cand.text
            conf = min(0.99, max(0.1, best_cand.validity / 12.0))
        except Exception:
            raw = ""
            conf = 0.0
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        plate = correct_plate_characters(raw)
        return OCRResult(
            raw=raw,
            plate=plate,
            confidence=round(conf, 4),
            valid=validate_vietnamese_plate(plate),
            source="train_ocr",
        )
