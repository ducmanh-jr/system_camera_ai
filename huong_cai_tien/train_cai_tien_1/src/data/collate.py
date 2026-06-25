"""Collate fn that prepares a batch for CTC loss.

CTCLoss expects:
    log_probs       :  (T, B, C)
    targets         :  1-D concatenation of all label indices (or 2-D padded)
    input_lengths   :  (B,)  -- here all equal to T (we use a fixed width)
    target_lengths  :  (B,)  -- original (un-padded) length of each label
"""
from __future__ import annotations
from typing import List, Tuple
import torch

from .vocab import Vocab


def make_collate_fn(vocab: Vocab):
    def collate(batch: List[Tuple[torch.Tensor, str]]):
        imgs = torch.stack([b[0] for b in batch], dim=0)        # (B, C, H, W)
        texts = [b[1] for b in batch]
        targets, target_lengths = vocab.encode_batch(texts)
        return imgs, targets, target_lengths, texts
    return collate
