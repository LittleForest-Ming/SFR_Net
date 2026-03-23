from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in {'l1', 'smooth_l1', 'bce'}:
            raise ValueError(f'Unsupported uncertainty loss_type: {loss_type}')

    def _align_mask(self, valid_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape == reference.shape:
            return valid_mask
        if valid_mask.ndim == reference.ndim - 1:
            return valid_mask.unsqueeze(1)
        if valid_mask.ndim == reference.ndim and valid_mask.shape[1] == 1 and reference.shape[1] == 1:
            return valid_mask
        raise ValueError(
            f'valid_mask shape {tuple(valid_mask.shape)} is incompatible with '
            f'prediction shape {tuple(reference.shape)}.'
        )

    def forward(
        self,
        pred: torch.Tensor | None,
        target: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target is None:
            if pred is not None:
                return pred.new_zeros(())
            if valid_mask is not None:
                return valid_mask.new_zeros(())
            return torch.zeros((), dtype=torch.float32)
        if pred is None:
            return target.new_zeros(())
        if pred.shape != target.shape:
            raise ValueError(
                f'Uncertainty prediction shape {tuple(pred.shape)} does not match '
                f'target shape {tuple(target.shape)}.'
            )

        if self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'bce':
            pred = pred.clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(pred, target, reduction='none')
        else:
            loss = (pred - target).abs()

        if valid_mask is not None:
            mask = self._align_mask(valid_mask, loss).to(dtype=loss.dtype, device=loss.device)
            if mask.numel() == 0 or float(mask.sum().item()) <= 0.0:
                return loss.new_zeros(())
            loss = loss * mask
            denom = mask.sum().clamp_min(1.0)
            return loss.sum() / denom

        if loss.numel() == 0:
            return loss.new_zeros(())
        return loss.mean()
