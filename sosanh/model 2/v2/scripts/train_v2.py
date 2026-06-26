"""Entry point for v2 training.

Kaggle/Colab:
    python v2/scripts/train_v2.py \
        --config v2/configs/crnn_v2.yaml \
        --data-root /kaggle/input/.../ocr_dataset \
        --out-dir /kaggle/working/runs/crnn_v2

DDP:
    torchrun --nproc_per_node=2 v2/scripts/train_v2.py --config v2/configs/crnn_v2.yaml --ddp

Resume:
    python v2/scripts/train_v2.py --config v2/configs/crnn_v2.yaml --resume runs/crnn_v2/last.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist

# Allow running from project root (parent of `v2/`)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)            # access to v1 `src.*`
sys.path.insert(0, os.path.dirname(HERE))   # access to `v2.src.*` via `v2` package
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # `v2.src.*` absolute

# Make `v2` importable as a top-level package
import v2  # type: ignore  # noqa: F401

from v2.src.training.trainer_v2 import TrainerV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--synthetic-dir", default=None)
    p.add_argument("--teacher-ckpt", default=None,
                   help="Enable KD by pointing to a teacher checkpoint (e.g. v1 best.pt)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.synthetic_dir:
        cfg["data"]["synthetic_dir"] = args.synthetic_dir
    if args.out_dir:
        cfg["output"]["dir"] = args.out_dir
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.teacher_ckpt:
        cfg["loss"].setdefault("distill", {})["teacher_ckpt"] = args.teacher_ckpt
    cfg["train"]["ddp"] = bool(args.ddp)

    if args.ddp:
        dist.init_process_group(backend="nccl")

    trainer = TrainerV2(cfg)
    trainer.train()

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
