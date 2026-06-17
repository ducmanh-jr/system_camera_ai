# Vietnamese License Plate OCR Ensemble

Production-style package for license plate text recognition.

## Folder Structure

```text
best_ocr_ensemble/
  input/                  # Put images here
  output/                 # Annotated images, crops, results.csv
  models/
    train1/               # PaddleOCR recognition model
    train2/               # PyTorch CTC recognition model
    plate_detector/       # YOLO license-plate detector
  src/
    pipeline.py           # OCR, detector fallback, color/type rules
    ocr_code/             # Local PyTorch OCR architecture code
  reports/                # Final benchmark/report only
  main.py                 # Main entrypoint
  README.md
```

## Run

1. Put images into `input/`.
2. Run:

```powershell
py main.py
```

3. Read results in:

```text
output/results.csv
output/*_result.jpg
```

## Models

- `train1`: PaddleOCR recognition model.
- `train2`: PyTorch CTC OCR model.
- `plate_detector`: YOLO detector used as a safe fallback for difficult crops/full vehicle images.

## Current Best Logic

- `train2` is the main OCR backbone.
- `train1` is used to repair serial/tail mistakes, especially patterns like `123.45`.
- HSV plate-color classification is used as context: `white`, `yellow`, `blue`, `red`, `unknown`.
- Vietnamese plate-type rules are applied for common plates, special series, old-style plates, and `AA/TA/LA` vs `A4/T4/L4` confusion.
- Detector fallback is used only when the detected crop output is structurally better and safe.

## Best Verified Score On Test Set

| Mode | Character Accuracy | Full Plate Accuracy |
|---|---:|---:|
| default OCR merge | 93.04% | 74.82% |
| detector fallback | 93.06% | 75.39% |

