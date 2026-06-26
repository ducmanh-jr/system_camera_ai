"""Comprehensive evaluation: CER, WER, seq_acc, char confusion matrix, top-N errors.

Usage:
    python v2/scripts/evaluate_v2.py \
        --ckpt runs/crnn_v2/best.pt \
        --data-root /path/ocr_dataset --split test --prefer-ema \
        --beam 10 --save-mismatches runs/crnn_v2/test_mismatches.csv \
        --save-confusion runs/crnn_v2/confusion.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import v2  # noqa: F401
from v2.src.models.crnn_v2 import build_model_v2
from src.data.vocab import Vocab
from src.data.transforms import Preprocessor, PreprocessConfig
from src.data.dataset import PlateDataset
from src.data.collate import make_collate_fn
from src.inference.decode import greedy_decode, beam_search_decode
from src.utils.metrics import compute_metrics


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--prefer-ema", action="store_true")
    p.add_argument("--beam", type=int, default=0,
                   help="Beam size; 0 = greedy")
    p.add_argument("--save-mismatches", default=None)
    p.add_argument("--save-confusion", default=None)
    return p.parse_args()


def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]

    vocab = Vocab(cfg["vocab"]["charset"])
    pre_cfg = PreprocessConfig(
        img_height=cfg["data"]["img_height"],
        img_width=cfg["data"]["img_width"],
        grayscale=cfg["data"]["grayscale"],
        apply_clahe=cfg["data"]["apply_clahe"],
    )
    pre = Preprocessor(pre_cfg)
    ds = PlateDataset(args.data_root, args.split, vocab, pre, augment=None)
    collate = make_collate_fn(vocab)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate,
    )

    in_ch = 1 if cfg["data"]["grayscale"] else 3
    model = build_model_v2(cfg["model"], num_classes=vocab.num_classes, in_channels=in_ch).to(device)
    state = ckpt.get("ema_state_dict") if args.prefer_ema else ckpt.get("model_state_dict")
    state = state or ckpt["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, _t, _tl, texts in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            if args.beam > 0:
                preds = beam_search_decode(log_probs, vocab, beam_size=args.beam)
            else:
                preds = greedy_decode(log_probs, vocab)
            all_preds.extend(preds)
            all_gts.extend(texts)

    metrics = compute_metrics(all_preds, all_gts)
    print(
        f"split={args.split}  N={metrics.num_samples}  "
        f"seq_acc={metrics.seq_acc:.4f}  char_acc={metrics.char_acc:.4f}  "
        f"CER={metrics.cer:.4f}  WER={metrics.wer:.4f}"
    )

    # Mismatches CSV ------------------------------------------------
    if args.save_mismatches:
        with open(args.save_mismatches, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gt", "pred"])
            for p, g in zip(all_preds, all_gts):
                if p != g:
                    w.writerow([g, p])
        print("Saved mismatches to", args.save_mismatches)

    # Confusion CSV -------------------------------------------------
    if args.save_confusion:
        conf = Counter()
        for p, g in zip(all_preds, all_gts):
            for i in range(min(len(p), len(g))):
                if p[i] != g[i]:
                    conf[(g[i], p[i])] += 1
        with open(args.save_confusion, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gt_char", "pred_char", "count"])
            for (g, p), c in conf.most_common():
                w.writerow([g, p, c])
        print("Saved confusion to", args.save_confusion)


if __name__ == "__main__":
    main()
