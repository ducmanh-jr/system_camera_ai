"""Export a trained checkpoint to ONNX.

Usage:
    python scripts/export_onnx.py --ckpt runs/crnn_base/best.pt --out runs/crnn_base/model.onnx
"""
from __future__ import annotations
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.vocab import Vocab
from src.models.crnn import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--prefer-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="prefer_ema", action="store_false")
    p.add_argument("--height", type=int, default=None, help="override H")
    p.add_argument("--width", type=int, default=None, help="override W")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    vocab = Vocab(cfg["vocab"]["charset"])
    in_ch = 1 if cfg["data"]["grayscale"] else 3
    model = build_model(cfg["model"], num_classes=vocab.num_classes, in_channels=in_ch).eval()
    state = ckpt.get("ema_state_dict") if args.prefer_ema else None
    if state is None:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)

    H = args.height or cfg["data"]["img_height"]
    W = args.width or cfg["data"]["img_width"]
    dummy = torch.zeros(1, in_ch, H, W)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"},
                       "logits": {0: "seq_len", 1: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {args.out}")

    # also save vocab next to it for runtime decoding
    vocab_txt = args.out + ".vocab.txt"
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for c in vocab.idx2char:
            f.write(c + "\n")
    print(f"Wrote vocab to {vocab_txt}")


if __name__ == "__main__":
    main()
