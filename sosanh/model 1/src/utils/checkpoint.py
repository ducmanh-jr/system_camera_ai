"""Checkpoint save / load helpers."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
import torch


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
