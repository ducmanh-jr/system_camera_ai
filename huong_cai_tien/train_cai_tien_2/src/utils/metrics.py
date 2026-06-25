"""Evaluation metrics for OCR.

  - Sequence Accuracy (a.k.a. exact-match) : fraction of samples with pred == gt
  - Character Accuracy                      : 1 - CER (computed the same way)
  - CER (Character Error Rate)              : Σ edit_distance(pred, gt) / Σ len(gt)
  - WER (Word Error Rate)                   : Σ edit_distance(pred.split(), gt.split()) / Σ len(gt.split())

Edit distance = Levenshtein. We try to use `python-Levenshtein` if available
(C-extension, very fast); otherwise we fall back to a pure-Python DP that handles
both strings and lists (for WER).
"""
from __future__ import annotations
from typing import List, Sequence
from dataclasses import dataclass

try:
    import Levenshtein as _Lev  # python-Levenshtein
    _HAS_LEV = True
except ImportError:  # pragma: no cover
    _HAS_LEV = False


def _editdistance(a: Sequence, b: Sequence) -> int:
    if _HAS_LEV and isinstance(a, str) and isinstance(b, str):
        return _Lev.distance(a, b)
    # Fallback DP — works for lists too
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


@dataclass
class OCRMetrics:
    seq_acc: float
    char_acc: float
    cer: float
    wer: float
    num_samples: int


def compute_metrics(preds: List[str], gts: List[str]) -> OCRMetrics:
    assert len(preds) == len(gts), "preds and gts must have same length"
    if len(preds) == 0:
        return OCRMetrics(0.0, 0.0, 0.0, 0.0, 0)
    exact = 0
    total_char_dist = 0
    total_char_len = 0
    total_word_dist = 0
    total_word_len = 0
    for p, g in zip(preds, gts):
        if p == g:
            exact += 1
        total_char_dist += _editdistance(p, g)
        total_char_len += max(1, len(g))
        p_words = p.split()
        g_words = g.split()
        total_word_dist += _editdistance(p_words, g_words)
        total_word_len += max(1, len(g_words))
    seq_acc = exact / len(preds)
    cer = total_char_dist / total_char_len
    char_acc = max(0.0, 1.0 - cer)
    wer = total_word_dist / total_word_len
    return OCRMetrics(seq_acc=seq_acc, char_acc=char_acc, cer=cer, wer=wer, num_samples=len(preds))
