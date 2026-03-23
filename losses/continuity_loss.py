from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ContinuityLoss(nn.Module):
    def __init__(self, loss_type='bce', pos_weight=None):
        super().__init__()
        self.loss_type = loss_type
        self.pos_weight = pos_weight

    def forward(self, pred, target, valid_mask=None):
        if pred is None or target is None:
            reference = pred if pred is not None else target
            return reference.new_zeros(()) if reference is not None else 0.0
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        if valid_mask is not None:
            loss = loss * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()
