from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.pipeline import (
    Candidate,
    PlateDetector,
    PlateOCREnsemble,
    classify_plate_color,
    normalize_plate_text,
    should_use_detector_fallback,
)


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
CROP_DIR = OUTPUT_DIR / "crops"
RESULT_CSV = OUTPUT_DIR / "results.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_font(size: int = 24):
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate_image(image_path: Path, output_path: Path, text: str, plate_color: str, model_name: str) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    banner_h = max(56, height // 9)
    canvas = Image.new("RGB", (width, height + banner_h), (24, 24, 24))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(max(18, min(34, width // 22)))
    small_font = load_font(max(13, min(20, width // 36)))

    title = f"PLATE: {text}"
    meta = f"color={plate_color} | model={model_name}"
    draw.text((16, height + 8), title, fill=(0, 235, 120), font=font)
    draw.text((16, height + 34), meta, fill=(220, 220, 220), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def list_input_images() -> list[Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    images = list_input_images()
    if not images:
        print(f"No images found in: {INPUT_DIR}")
        print("Put plate/vehicle images into input/, then run: py main.py")
        return

    ensemble = PlateOCREnsemble()
    detector = PlateDetector(ROOT_DIR / "models" / "plate_detector" / "license_plate_detector.pt")

    rows = []
    for image_path in images:
        base_color, _ = classify_plate_color(image_path)
        base_best, base_candidates = ensemble.predict(image_path)

        crop_path, det_info = detector.crop_best_plate(image_path, CROP_DIR)
        crop_best, crop_candidates = ensemble.predict(crop_path)

        final_best = base_best
        final_image_for_color = image_path
        fallback_used = False
        if should_use_detector_fallback(base_best, crop_best):
            final_best = Candidate(
                model="detector_fallback_" + crop_best.model,
                text=crop_best.text,
                norm_text=crop_best.norm_text,
                validity=crop_best.validity,
                time_ms=crop_best.time_ms,
                error=crop_best.error,
            )
            final_image_for_color = crop_path
            fallback_used = True

        plate_color, _ = classify_plate_color(final_image_for_color)
        output_image = OUTPUT_DIR / f"{image_path.stem}_result.jpg"
        annotate_image(image_path, output_image, final_best.norm_text, plate_color, final_best.model)

        rows.append(
            {
                "image": image_path.name,
                "prediction": final_best.norm_text,
                "prediction_raw": final_best.text,
                "plate_color": plate_color,
                "selected_model": final_best.model,
                "detector_used": fallback_used,
                "detected": det_info.get("detected", ""),
                "det_conf": det_info.get("det_conf", ""),
                "crop_path": str(crop_path),
                "output_image": str(output_image),
                "train1_pred": next((c.text for c in base_candidates if c.model == "train1_paddleocr"), ""),
                "train2_pred": next((c.text for c in base_candidates if c.model == "train2_best_ema"), ""),
                "crop_train1_pred": next((c.text for c in crop_candidates if c.model == "train1_paddleocr"), ""),
                "crop_train2_pred": next((c.text for c in crop_candidates if c.model == "train2_best_ema"), ""),
            }
        )
        print(f"{image_path.name} -> {final_best.norm_text}")

    with RESULT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
    print(f"CSV: {RESULT_CSV}")


if __name__ == "__main__":
    main()
