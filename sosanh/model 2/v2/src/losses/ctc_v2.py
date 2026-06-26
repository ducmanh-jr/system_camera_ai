"""CTC v2 — adds label smoothing on the per-timestep posterior, entropy reg,
and an optional Knowledge Distillation term.

The label-smoothed CTC formulation:
    L_total = (1 - eps) * L_ctc + eps * L_uniform_kl
where L_uniform_kl = KL( softmax(logits) || uniform ) per time step.
Reference: "Improved CTC Loss with Label Smoothing" — Shinohara, INTERSPEECH '19.

Knowledge Distillation:
    L_kd = T^2 * KL( softmax(logits_s/T) || softmax(logits_t/T) ) per-timestep,
averaged over (T, B). Lets the small student inherit the teacher's per-frame
class confusion — especially valuable for the digit-pair confusions we saw
in v1 evaluation (0/8, 1/7, 6/8 etc.).
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCv2(nn.Module):
    def __init__(
        self,
        blank: int = 0,
        zero_infinity: bool = True,
        entropy_weight: float = 0.0,
        label_smoothing: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction="mean")
        self.entropy_weight = entropy_weight
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,                  # (T, B, C)
        targets: torch.Tensor,                 # (sum_target_lengths,)
        input_lengths: torch.Tensor,           # (B,)
        target_lengths: torch.Tensor,          # (B,)
    ) -> Dict[str, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=-1)
        ctc_loss = self.ctc(log_probs.float(), targets, input_lengths, target_lengths)
        out: Dict[str, torch.Tensor] = {"loss_ctc": ctc_loss.detach()}
        total = ctc_loss

        # ---- Label smoothing on per-frame posterior (KL vs uniform) -----
        if self.label_smoothing > 0:
            C = logits.size(-1)
            # KL(p || uniform) = -H(p) + log(C)
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1).mean()
            kl_uniform = -ent + torch.log(torch.tensor(float(C), device=logits.device))
            total = (1 - self.label_smoothing) * total + self.label_smoothing * kl_uniform
            out["loss_label_smooth"] = kl_uniform.detach()

        # ---- Entropy regularizer (sharpen posterior) --------------------
        if self.entropy_weight > 0:
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1).mean()
            total = total + self.entropy_weight * ent
            out["loss_entropy"] = ent.detach()

        out["loss"] = total
        return out


class KDCTCLoss(nn.Module):
    """Wraps CTCv2 and adds a per-timestep KD term from a teacher model."""

    def __init__(self, base: CTCv2, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.base = base
        self.alpha = alpha
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base = self.base(student_logits, targets, input_lengths, target_lengths)
        T = self.T
        # KL(student || teacher) at temperature T
        s_logp = F.log_softmax(student_logits / T, dim=-1)
        t_p = F.softmax(teacher_logits / T, dim=-1).detach()
        kl = F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)
        loss = (1 - self.alpha) * base["loss"] + self.alpha * kl
        base["loss"] = loss
        base["loss_kd"] = kl.detach()
        return base
