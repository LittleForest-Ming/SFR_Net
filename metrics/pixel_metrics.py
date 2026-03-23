from __future__ import annotations

import torch


def pixel_metrics(pred_center, target_center, threshold: float = 0.5):
    if target_center is None:
        return {'center_mae': 0.0, 'center_precision': 0.0, 'center_recall': 0.0, 'center_f1': 0.0}
    pred_bin = (pred_center >= threshold).float()
    target_bin = (target_center >= threshold).float()
    tp = float((pred_bin * target_bin).sum().item())
    fp = float((pred_bin * (1.0 - target_bin)).sum().item())
    fn = float(((1.0 - pred_bin) * target_bin).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    mae = float((pred_center - target_center).abs().mean().item())
    return {'center_mae': mae, 'center_precision': precision, 'center_recall': recall, 'center_f1': f1}
