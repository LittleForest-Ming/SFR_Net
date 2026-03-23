from __future__ import annotations

import torch


DEFAULT_THRESHOLD = 0.5


def _to_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().float().cpu() if hasattr(value, 'detach') else torch.as_tensor(value, dtype=torch.float32)


def _align_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.shape == reference.shape:
        return mask
    if mask.ndim == reference.ndim - 1:
        return mask.unsqueeze(1)
    if mask.ndim == reference.ndim and mask.shape[1] == 1 and reference.shape[1] == 1:
        return mask
    raise ValueError(
        f'valid_mask shape {tuple(mask.shape)} is incompatible with reference shape {tuple(reference.shape)}.'
    )


def _prepare_inputs(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    pred_tensor = _to_tensor(pred)
    target_tensor = _to_tensor(target)
    if pred_tensor is None or target_tensor is None:
        return None, None, None
    if pred_tensor.shape != target_tensor.shape:
        raise ValueError(
            f'uncertainty pred shape {tuple(pred_tensor.shape)} does not match '
            f'target shape {tuple(target_tensor.shape)}.'
        )
    mask = None
    if valid_mask is not None:
        mask = _align_mask(_to_tensor(valid_mask), pred_tensor)
        if mask.numel() == 0 or float(mask.sum().item()) <= 0.0:
            return pred_tensor, target_tensor, torch.zeros_like(pred_tensor)
    return pred_tensor, target_tensor, mask


def uncertainty_mae(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> float:
    pred_tensor, target_tensor, mask = _prepare_inputs(pred, target, valid_mask)
    if pred_tensor is None or target_tensor is None:
        return 0.0
    abs_error = (pred_tensor - target_tensor).abs()
    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
        return float((abs_error * mask).sum().item() / denom.item())
    if abs_error.numel() == 0:
        return 0.0
    return float(abs_error.mean().item())


def uncertainty_l1_score(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> float:
    mae = uncertainty_mae(pred, target, valid_mask=valid_mask)
    return max(0.0, 1.0 - mae)


def uncertainty_region_precision(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    pred_tensor, target_tensor, mask = _prepare_inputs(pred, target, valid_mask)
    if pred_tensor is None or target_tensor is None:
        return 0.0
    pred_bin = (pred_tensor >= float(threshold)).float()
    target_bin = (target_tensor >= float(threshold)).float()
    if mask is not None:
        pred_bin = pred_bin * mask
        target_bin = target_bin * mask
    tp = float((pred_bin * target_bin).sum().item())
    fp = float((pred_bin * (1.0 - target_bin)).sum().item())
    return tp / max(tp + fp, 1.0)


def compute_uncertainty_metrics(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute lightweight uncertainty metrics on valid pixels only."""
    if pred is None or target is None:
        return {
            'uncertainty_mae': 0.0,
            'uncertainty_bce_like_score': 0.0,
            'uncertainty_region_precision': 0.0,
        }

    mae = uncertainty_mae(pred, target, valid_mask=valid_mask)
    l1_score = uncertainty_l1_score(pred, target, valid_mask=valid_mask)
    region_precision = uncertainty_region_precision(pred, target, valid_mask=valid_mask, threshold=threshold)
    return {
        'uncertainty_mae': mae,
        'uncertainty_bce_like_score': l1_score,
        'uncertainty_region_precision': region_precision,
    }
