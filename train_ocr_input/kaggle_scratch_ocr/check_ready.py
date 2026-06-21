import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

from train_best import BestOCR, Codec, normalize_plate_prediction, parse_sizes, pil_to_tensor, resize_pad


def read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--multi-sizes", default="40x160,48x192,56x224")
    parser.add_argument("--img-h", type=int, default=48)
    parser.add_argument("--img-w", type=int, default=192)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    print("Checking prepared data:", data_root)
    for name in ["train.csv", "val.csv", "test.csv", "charset.txt"]:
        require((data_root / name).exists(), f"Missing {name}")

    charset = (data_root / "charset.txt").read_text(encoding="utf-8").strip("\n")
    require("Đ" in charset, "Charset must contain Vietnamese Đ")
    require(" " in charset, "Charset should preserve spaces")
    codec = Codec(charset)
    print("Charset:", charset)

    train_rows = read_rows(data_root / "train.csv")
    val_rows = read_rows(data_root / "val.csv")
    test_rows = read_rows(data_root / "test.csv")
    print("Rows:", {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)})
    require(len(train_rows) > 0 and len(val_rows) > 0 and len(test_rows) > 0, "Empty split detected")

    bad = []
    for split, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        for row in rows[: min(200, len(rows))]:
            p = Path(row["image_path"])
            label = row["label"]
            if not p.exists():
                bad.append((split, "missing_image", str(p)))
            unknown = sorted(set(label) - set(charset))
            if unknown:
                bad.append((split, "unknown_chars", label, unknown))
    require(not bad, f"Data check failed: {bad[:5]}")

    sample = train_rows[0]
    img = Image.open(sample["image_path"])
    x = pil_to_tensor(resize_pad(img, args.img_h, args.img_w)).unsqueeze(0)
    model = BestOCR(codec.num_classes, len(codec.chars), args.model_dim, args.layers, args.heads, 0.1)
    model.eval()
    with torch.no_grad():
        logits, sem = model(x)
    require(logits.ndim == 3 and sem.ndim == 2, "Model forward shape invalid")
    require(logits.shape[-1] == codec.num_classes, "CTC class count mismatch")
    require(sem.shape[-1] == len(codec.chars), "Semantic head class count mismatch")
    print("Forward OK:", {"ctc": tuple(logits.shape), "semantic": tuple(sem.shape)})

    decoder_tests = {
        "59AI OO128": "59A1 00128",
        "3OF 782B6": "30F 78286",
        "50TĐ O3O27": "50TĐ 03027",
        "60MĐ1 O16O4": "60MĐ1 01604",
    }
    for raw, expected in decoder_tests.items():
        got = normalize_plate_prediction(raw)
        print("Decoder:", raw, "->", got)
        require(got == expected, f"Decoder expected {expected}, got {got}")

    print("READY_CHECK_PASS")


if __name__ == "__main__":
    main()
