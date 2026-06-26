"""Generate a synthetic VN-plate dataset.

Usage:
    python scripts/gen_synthetic.py --out data/synth --num 50000 --font /path/to/font.ttf

Where to get the font:
    - You can use any sans-serif TTF that looks close to VN plate font, e.g.
      'EuroPlate', 'License Plate USA', or the official 'FE-Schrift'-style fonts.
    - For best results, use a real VN plate font (search 'Bien so xe ttf').
    - This generator is rule-based — no pretrained model is involved.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.synthetic import generate_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=50000)
    p.add_argument("--font", required=True, help="Path to TTF font file")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.font):
        raise FileNotFoundError(f"font not found: {args.font}")
    os.makedirs(args.out, exist_ok=True)
    generate_dataset(args.out, args.num, args.font, args.seed)
    print(f"Done: {args.num} samples -> {args.out}")


if __name__ == "__main__":
    main()
