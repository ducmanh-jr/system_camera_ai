"""PyTorch Dataset for VN license plate OCR.

Expected on-disk layout (your current dataset matches this):

    <data.root>/
        train/                  *.jpg images
        val/                    *.jpg images
        test/                   *.jpg images
        train_labels.txt        each line: "<filename>\t<plate_text>"
        val_labels.txt
        test_labels.txt
"""
from __future__ import annotations
import os
import logging
from typing import List, Optional, Sequence, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .vocab import Vocab
from .transforms import Preprocessor

log = logging.getLogger(__name__)


def _read_labels_file(path: str) -> List[Tuple[str, str]]:
    """Read a `<filename>\\t<text>` file. Skip blank/bad lines with a warning."""
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if "\t" in line:
                fname, text = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    log.warning("Skipping malformed label line %d in %s: %r", i, path, line)
                    continue
                fname, text = parts
            pairs.append((fname.strip(), text.strip()))
    return pairs


def _filter_by_charset(pairs: Sequence[Tuple[str, str]], vocab: Vocab) -> List[Tuple[str, str]]:
    keep: List[Tuple[str, str]] = []
    bad = 0
    for fname, text in pairs:
        if all(c in vocab.char2idx for c in text):
            keep.append((fname, text))
        else:
            bad += 1
    if bad > 0:
        log.warning("Filtered out %d/%d samples (chars not in vocab)", bad, len(pairs))
    return keep


class PlateDataset(Dataset):
    """One sample = (preprocessed image tensor, label string).

    The CTC label encoding happens in the **collate_fn**, not here, because
    ``torch.nn.CTCLoss`` wants concatenated targets at batch level.

    Args:
        root: dataset root containing the split sub-folders and label files.
        split: 'train' | 'val' | 'test'
        vocab: Vocab object used to filter & later encode labels.
        preprocessor: deterministic preprocessing (resize/CLAHE/normalize).
        augment: optional callable returning a dict {'image': np.ndarray} (Albumentations style).
                 Only used in training.
    """

    def __init__(
        self,
        root: str,
        split: str,
        vocab: Vocab,
        preprocessor: Preprocessor,
        augment=None,
    ):
        self.root = root
        self.split = split
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.augment = augment
        labels_path = os.path.join(root, f"{split}_labels.txt")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        pairs = _read_labels_file(labels_path)
        pairs = _filter_by_charset(pairs, vocab)
        self.image_dir = os.path.join(root, split)
        # also retain only files that physically exist
        self.samples: List[Tuple[str, str]] = []
        miss = 0
        for fname, text in pairs:
            path = os.path.join(self.image_dir, fname)
            if os.path.isfile(path):
                self.samples.append((fname, text))
            else:
                miss += 1
        if miss > 0:
            log.warning("%d label entries had no matching image file in %s", miss, self.image_dir)
        log.info("Loaded split=%s: %d samples (root=%s)", split, len(self.samples), root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname, text = self.samples[idx]
        path = os.path.join(self.image_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise IOError(f"Could not read image: {path}")
        if self.augment is not None:
            img = self.augment(image=img)["image"]
        tensor = self.preprocessor(img)
        return tensor, text


class SyntheticListDataset(Dataset):
    """Dataset of synthetic samples produced by ``scripts/gen_synthetic.py``.

    Expected layout:
        <synthetic_dir>/
            images/  *.png
            labels.txt    "<filename>\\t<plate_text>" lines
    """

    def __init__(self, synthetic_dir: str, vocab: Vocab, preprocessor: Preprocessor, augment=None):
        self.root = synthetic_dir
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.augment = augment
        labels_path = os.path.join(synthetic_dir, "labels.txt")
        pairs = _read_labels_file(labels_path)
        pairs = _filter_by_charset(pairs, vocab)
        self.samples = pairs
        self.image_dir = os.path.join(synthetic_dir, "images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname, text = self.samples[idx]
        img = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Bad synthetic image: {fname}")
        if self.augment is not None:
            img = self.augment(image=img)["image"]
        tensor = self.preprocessor(img)
        return tensor, text


class MixedDataset(Dataset):
    """Train-only mixture: with prob `synthetic_ratio` sample from synthetic, else real."""

    def __init__(self, real: Dataset, synthetic: Dataset, synthetic_ratio: float):
        assert 0.0 <= synthetic_ratio <= 1.0
        self.real = real
        self.synth = synthetic
        self.p = synthetic_ratio
        # length = max(real, synth) keeps each epoch ~ same wallclock time
        self._len = max(len(real), len(synthetic))
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if self._rng.random() < self.p:
            return self.synth[self._rng.integers(0, len(self.synth))]
        return self.real[self._rng.integers(0, len(self.real))]
