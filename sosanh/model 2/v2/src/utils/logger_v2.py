"""Logger, TensorBoard writer, JSON-line metrics logger."""
from __future__ import annotations
import os
import json
import logging
import sys
from typing import Any, Dict, Optional


def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class TBWriter:
    """Thin wrapper that silently no-ops if TB is disabled or not installed."""

    def __init__(self, logdir: str, enabled: bool = True):
        self.enabled = enabled
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(logdir, exist_ok=True)
                self.writer = SummaryWriter(logdir)
            except Exception:  # pragma: no cover
                self.enabled = False

    def add_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def add_histogram(self, tag: str, values, step: int):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class JsonLineLogger:
    """Append-mode JSONL — one metrics row per epoch. Easy to parse & plot."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, obj: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
