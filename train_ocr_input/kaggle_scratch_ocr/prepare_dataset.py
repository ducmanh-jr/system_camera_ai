import argparse
import csv
import sys
from pathlib import Path


MOJIBAKE_FIXES = {
    "Ä": "Đ",
    "Ä‘": "đ",
    "Ð": "Đ",
}


def normalize_label(text: str, strip_spaces: bool = False) -> str:
    text = text.strip()
    for bad, good in MOJIBAKE_FIXES.items():
        text = text.replace(bad, good)
    text = " ".join(text.split())
    text = text.upper().replace("Đ", "Đ")
    if strip_spaces:
        text = text.replace(" ", "")
    return text


def parse_label_line(line: str):
    if "\t" in line:
        image, label = line.rstrip("\n").split("\t", 1)
    else:
        image, label = line.strip().split(maxsplit=1)
    return image.strip(), label


def convert_split(data_root: Path, out_root: Path, split: str, strip_spaces: bool):
    label_path = data_root / f"{split}_labels.txt"
    rows = []
    chars = set()
    for raw in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not raw.strip():
            continue
        image, label = parse_label_line(raw)
        label = normalize_label(label, strip_spaces=strip_spaces)
        image_path = data_root / split / image
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {split}: {image_path}")
        chars.update(label)
        rows.append((str(image_path), label))

    out_csv = out_root / f"{split}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)
    return rows, chars


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--strip-spaces", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_chars = set()
    summary = []
    for split in ("train", "val", "test"):
        rows, chars = convert_split(data_root, out_root, split, args.strip_spaces)
        all_chars.update(chars)
        summary.append((split, len(rows)))

    charset = "".join(sorted(all_chars))
    (out_root / "charset.txt").write_text(charset + "\n", encoding="utf-8")
    (out_root / "summary.txt").write_text(
        "\n".join([f"{s}: {n}" for s, n in summary] + [f"charset: {charset}"]) + "\n",
        encoding="utf-8",
    )
    print((out_root / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
