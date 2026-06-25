"""CTC loss with optional entropy regularization.

CTC loss formula (Graves et al. 2006):
    L_CTC = - log p(y | x) = - log sum_{pi in B^{-1}(y)} prod_t p(pi_t | x_t)
where:
    pi is an alignment (a sequence of length T over the augmented alphabet C ∪ {blank})
    B is the "collapse" operator: remove repeats then remove blanks.
The forward-backward DP runs in O(T * |y|) per sample.

Entropy regularizer:
    L_ent = - sum_t H(p(. | x_t))    (negative because we want to MINIMIZE entropy)
Adding `entropy_weight * L_ent` to the main loss encourages a sharper posterior
(useful at greedy decode time). Recommended: 0.005–0.02.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCWithEntropy(nn.Module):
    def __init__(
        self,
        blank: int = 0,
        zero_infinity: bool = True,
        reduction: str = "mean",
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction=reduction)
        self.entropy_weight = entropy_weight

    def forward(
        self,
        logits: torch.Tensor,             # (T, B, C)
        targets: torch.Tensor,            # (sum_target_lengths,)
        input_lengths: torch.Tensor,      # (B,)
        target_lengths: torch.Tensor,     # (B,)
    ) -> dict:
        log_probs = F.log_softmax(logits, dim=-1)
        # CTC loss must be computed in float32 for numerical stability
        ctc_loss = self.ctc(log_probs.float(), targets, input_lengths, target_lengths)
        out = {"loss": ctc_loss, "loss_ctc": ctc_loss.detach()}
        if self.entropy_weight > 0:
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1).mean()  # mean over (T, B)
            out["loss"] = ctc_loss + self.entropy_weight * ent
            out["loss_entropy"] = ent.detach()
        return out
