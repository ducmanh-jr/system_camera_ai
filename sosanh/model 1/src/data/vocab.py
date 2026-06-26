"""Vocabulary + CTC label encoder/decoder for VN license plate OCR.

Convention:
    - index 0 is reserved for the **CTC blank** token.
    - Characters in `charset` are mapped to indices [1, 2, ..., len(charset)].

The blank token is NOT a real character. CTC's collapse function (squash repeats,
remove blanks) gives the final transcription.
"""
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple
import torch


class Vocab:
    """Character-level vocabulary for CTC-based OCR.

    Args:
        charset: a string containing all valid (distinct) characters. Order matters:
                 it determines the index of each character (after the blank).
                 Example for VN plates: "0123456789ABCDEFGHIJKLMNOPQRSTUVXYZ "
        blank_index: must be 0 for compatibility with ``torch.nn.CTCLoss``'s default.

    Attributes:
        char2idx: dict mapping a character -> int index
        idx2char: list mapping index -> character (idx2char[0] == '<blank>')
        num_classes: |charset| + 1 (extra slot for blank)
    """
    BLANK_TOKEN = "<blank>"

    def __init__(self, charset: str, blank_index: int = 0):
        assert blank_index == 0, "Only blank_index=0 is supported (PyTorch default)."
        # Deduplicate while preserving order
        seen = set()
        clean = []
        for c in charset:
            if c not in seen:
                seen.add(c)
                clean.append(c)
        self.charset: str = "".join(clean)
        self.blank_index = blank_index
        self.idx2char: List[str] = [self.BLANK_TOKEN] + list(self.charset)
        self.char2idx: dict[str, int] = {c: i + 1 for i, c in enumerate(self.charset)}
        self.num_classes: int = len(self.idx2char)  # includes blank

    # ---------- encoding ----------
    def encode(self, text: str) -> List[int]:
        """Map a transcription string -> list of indices (no blanks inserted).

        Unknown characters are silently dropped (raise in strict mode if you prefer).
        For VN plates we keep the strict-drop policy because labels were cleaned upstream.
        """
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def encode_batch(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of transcriptions for ``nn.CTCLoss``.

        Returns:
            targets: 1D LongTensor of concatenated indices (CTCLoss expects this shape).
            target_lengths: 1D LongTensor of original lengths.
        """
        all_ids: List[int] = []
        lengths: List[int] = []
        for t in texts:
            ids = self.encode(t)
            all_ids.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(all_ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    # ---------- decoding ----------
    def decode_indices(self, ids: Iterable[int], collapse: bool = True) -> str:
        """Decode a sequence of int indices back to a string.

        If ``collapse=True`` (default), apply CTC collapse rule:
            (1) merge consecutive identical indices
            (2) drop blanks
        """
        out: List[str] = []
        prev = -1
        for i in ids:
            i = int(i)
            if collapse:
                if i == prev:
                    continue
                prev = i
                if i == self.blank_index:
                    continue
            else:
                if i == self.blank_index:
                    continue
            out.append(self.idx2char[i])
        return "".join(out)

    @classmethod
    def from_config(cls, cfg: dict) -> "Vocab":
        return cls(cfg["charset"])

    def __len__(self) -> int:
        return self.num_classes

    def __repr__(self) -> str:
        return f"Vocab(num_classes={self.num_classes}, charset={self.charset!r})"
