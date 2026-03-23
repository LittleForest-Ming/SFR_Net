from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask=None):
        if target is None:
            return pred.new_zeros(())
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        if valid_mask is not None:
            loss = loss * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()
