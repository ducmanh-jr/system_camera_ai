"""Logging helpers (stdlib + tensorboard, optional wandb)."""
from __future__ import annotations
import logging
import os
import sys
from typing import Optional


def setup_logger(name: str = "vn_alpr", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class TBWriter:
    """Thin wrapper so the trainer doesn't import torch.utils.tensorboard unconditionally."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def add_scalar(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_scalar(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_text(*args, **kwargs)

    def close(self):
        if self.writer is not None:
            self.writer.close()
