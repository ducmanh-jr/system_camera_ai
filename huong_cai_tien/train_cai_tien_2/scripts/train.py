"""Train CRNN OCR from scratch on VN license plate dataset.

Single GPU:
    python scripts/train.py --config configs/crnn_base.yaml

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/crnn_base.yaml --ddp

Resume:
    python scripts/train.py --config configs/crnn_base.yaml --resume runs/crnn_base/last.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=None, help="Override data.root in config")
    p.add_argument("--out-dir", default=None, help="Override output.dir in config")
    p.add_argument("--resume", default=None)
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--synthetic-dir", default=None)
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
    cfg["train"]["ddp"] = bool(args.ddp)

    if args.ddp:
        dist.init_process_group(backend="nccl")

    trainer = Trainer(cfg)
    trainer.train()

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
