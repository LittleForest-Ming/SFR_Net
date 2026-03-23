from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrientationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask=None):
        if target is None:
            zero = pred.new_zeros(())
            return zero, zero
        pred = F.normalize(pred, dim=1, eps=1e-6)
        target = F.normalize(target, dim=1, eps=1e-6)
        cosine = 1.0 - (pred * target).sum(dim=1, keepdim=True)
        if valid_mask is not None:
            cosine = cosine * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
            cos_loss = cosine.sum() / denom
        else:
            cos_loss = cosine.mean()
        norm_reg = pred.norm(dim=1).sub(1.0).abs().mean()
        return cos_loss, norm_reg
