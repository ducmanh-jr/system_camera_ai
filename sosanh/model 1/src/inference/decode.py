"""CTC decoders: greedy and (optional) prefix beam search.

The beam search implementation here is pure Python and tuned for short sequences
(plate length ≤ 11) — fast enough for online inference. For longer sequences
consider ``ctcdecode`` or ``pyctcdecode``.
"""
from __future__ import annotations
import math
from typing import List, Sequence
import torch

from ..data.vocab import Vocab


def greedy_decode(log_probs: torch.Tensor, vocab: Vocab) -> List[str]:
    """Greedy CTC decode.

    Args:
        log_probs: (T, B, C) — output of model.predict_logp(x)
    Returns:
        list of B decoded strings.
    """
    # argmax per timestep
    preds = log_probs.argmax(dim=-1)  # (T, B)
    preds = preds.transpose(0, 1).cpu().numpy()  # (B, T)
    out: List[str] = []
    for row in preds:
        out.append(vocab.decode_indices(row, collapse=True))
    return out


def beam_search_decode(
    log_probs: torch.Tensor,
    vocab: Vocab,
    beam_size: int = 10,
) -> List[str]:
    """Prefix beam search (token-only, no LM).

    Reference: Hannun et al. 2014 "First-Pass LM..." section 3.1 (numerically
    stable in log-space).

    Args:
        log_probs: (T, B, C)
    Returns:
        list of B decoded strings.
    """
    T, B, C = log_probs.shape
    lp = log_probs.cpu().numpy()
    results: List[str] = []
    blank = vocab.blank_index
    NEG_INF = float("-inf")

    def _logsumexp(a: float, b: float) -> float:
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    for bi in range(B):
        # beams: dict prefix(tuple of indices) -> (p_blank, p_nonblank)
        beams = {(): (0.0, NEG_INF)}  # initial: empty prefix with prob 1 ending in blank
        for t in range(T):
            new_beams: dict = {}
            row = lp[t, bi]
            # only consider top-2*beam_size tokens for speed
            top_k = min(2 * beam_size, C)
            top_ids = row.argsort()[-top_k:][::-1]
            for prefix, (pb, pnb) in beams.items():
                for ci in top_ids:
                    p = float(row[ci])
                    if ci == blank:
                        # extend with blank: collapses
                        new_pb, new_pnb = new_beams.get(prefix, (NEG_INF, NEG_INF))
                        new_pb = _logsumexp(new_pb, _logsumexp(pb, pnb) + p)
                        new_beams[prefix] = (new_pb, new_pnb)
                    else:
                        # same char as last → must come from blank-ending path
                        last = prefix[-1] if prefix else None
                        # case: extending blank ending => append
                        ext = prefix + (int(ci),)
                        npb, npnb = new_beams.get(ext, (NEG_INF, NEG_INF))
                        if last == int(ci):
                            # only from blank-ending: pb
                            npnb = _logsumexp(npnb, pb + p)
                            # the non-blank ending of same char stays in same prefix
                            spb, spnb = new_beams.get(prefix, (NEG_INF, NEG_INF))
                            spnb = _logsumexp(spnb, pnb + p)
                            new_beams[prefix] = (spb, spnb)
                        else:
                            npnb = _logsumexp(npnb, _logsumexp(pb, pnb) + p)
                        new_beams[ext] = (npb, npnb)
            # prune
            scored = sorted(
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_size]
            beams = dict(scored)
        # pick best beam
        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(vocab.idx2char[i] for i in best))
    return results
