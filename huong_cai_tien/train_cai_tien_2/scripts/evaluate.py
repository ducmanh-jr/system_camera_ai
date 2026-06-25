"""Evaluate a trained checkpoint on val/test split.

Usage:
    python scripts/evaluate.py --ckpt runs/crnn_base/best.pt --split test
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.vocab import Vocab
from src.data.transforms import Preprocessor, PreprocessConfig
from src.data.dataset import PlateDataset
from src.data.collate import make_collate_fn
from src.models.crnn import build_model
from src.inference.decode import greedy_decode, beam_search_decode
from src.utils.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-root", default=None)
    p.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    p.add_argument("--prefer-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="prefer_ema", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    vocab = Vocab(cfg["vocab"]["charset"])
    pre_cfg = PreprocessConfig(
        img_height=cfg["data"]["img_height"],
        img_width=cfg["data"]["img_width"],
        grayscale=cfg["data"]["grayscale"],
        apply_clahe=cfg["data"]["apply_clahe"],
    )
    pre = Preprocessor(pre_cfg)
    ds = PlateDataset(cfg["data"]["root"], args.split, vocab=vocab, preprocessor=pre)
    if args.limit is not None:
        ds.samples = ds.samples[:args.limit]
    num_workers = cfg["data"]["num_workers"] if args.num_workers is None else args.num_workers
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=num_workers,
                         pin_memory=True, collate_fn=make_collate_fn(vocab))

    model = build_model(cfg["model"], num_classes=vocab.num_classes,
                        in_channels=1 if pre_cfg.grayscale else 3).to(device).eval()
    state = ckpt.get("ema_state_dict") if args.prefer_ema else None
    if state is None:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)

    preds, gts = [], []
    with torch.inference_mode():
        for imgs, targets, target_lengths, texts in loader:
            imgs = imgs.to(device, non_blocking=True)
            log_probs = model.predict_logp(imgs)
            if args.decoder == "beam":
                preds.extend(beam_search_decode(log_probs, vocab, args.beam_size))
            else:
                preds.extend(greedy_decode(log_probs, vocab))
            gts.extend(texts)

    m = compute_metrics(preds, gts)
    print("=" * 60)
    print(f"Eval split={args.split}  decoder={args.decoder}  n={m.num_samples}")
    print(f"  Sequence Accuracy : {m.seq_acc:.4f}")
    print(f"  Character Accuracy: {m.char_acc:.4f}")
    print(f"  CER              : {m.cer:.4f}")
    print(f"  WER              : {m.wer:.4f}")
    # write a small CSV of mismatches for inspection
    out_dir = os.path.dirname(args.ckpt)
    out_path = os.path.join(out_dir, f"eval_{args.split}_mismatches.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("gt,pred\n")
        for p, g in zip(preds, gts):
            if p != g:
                f.write(f"{g},{p}\n")
    print(f"Mismatches written to {out_path}")


if __name__ == "__main__":
    main()
