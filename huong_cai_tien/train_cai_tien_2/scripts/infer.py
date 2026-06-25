"""Single-image / folder inference.

Usage:
    python scripts/infer.py --ckpt runs/crnn_base/best.pt --image plate.jpg
    python scripts/infer.py --ckpt runs/crnn_base/best.pt --image-dir folder/
"""
from __future__ import annotations
import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference.predictor import Predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    return p.parse_args()


def main():
    args = parse_args()
    pred = Predictor.from_checkpoint(args.ckpt, decoder=args.decoder)
    if args.image:
        text = pred.predict_image(args.image)
        print(f"{args.image}\t{text}")
    if args.image_dir:
        for path in sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                            glob.glob(os.path.join(args.image_dir, "*.png"))):
            text = pred.predict_image(path)
            print(f"{path}\t{text}")


if __name__ == "__main__":
    main()
