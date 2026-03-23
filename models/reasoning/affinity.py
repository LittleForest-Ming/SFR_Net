from __future__ import annotations

import torch


def _safe_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _cfg_value(cfg, path: str, default: float) -> float:
    current = cfg
    for name in path.split('.'):
        if current is None or not hasattr(current, name):
            return float(default)
        current = getattr(current, name)
    try:
        return float(current)
    except (TypeError, ValueError):
        return float(default)


def direction_consistency(current_dir, candidate_dir, move_vec):
    current_dir = _safe_normalize(current_dir)
    candidate_dir = _safe_normalize(candidate_dir)
    move_vec = _safe_normalize(move_vec)
    return 0.5 * ((current_dir * move_vec).sum(dim=-1) + (candidate_dir * move_vec).sum(dim=-1))


def distance_feasibility(distance, step_size, radius):
    normalized = distance / max(step_size + radius, 1e-6)
    return 1.0 - normalized.clamp(min=0.0, max=1.0)


def compute_local_affinity(center, continuity, orientation, structure, uncertainty=None, cfg=None):
    center_weight = _cfg_value(cfg, 'model.affinity_center_weight', 0.30)
    continuity_weight = _cfg_value(cfg, 'model.affinity_continuity_weight', 0.25)
    direction_weight = _cfg_value(cfg, 'model.affinity_direction_weight', 0.15)
    structure_weight = _cfg_value(cfg, 'model.affinity_structure_weight', 0.20)
    uncertainty_weight = _cfg_value(cfg, 'model.affinity_uncertainty_weight', 0.20)

    center_score = center.clamp(0.0, 1.0)
    continuity_score = continuity.clamp(0.0, 1.0) if continuity is not None else torch.zeros_like(center_score)
    direction_score = orientation.norm(dim=1, keepdim=True).clamp(0.0, 1.0)
    structure_score = structure.clamp(0.0, 1.0) if structure is not None else center_score

    positive_support = (
        center_weight * center_score
        + continuity_weight * continuity_score
        + direction_weight * direction_score
        + structure_weight * structure_score
    )

    if uncertainty is None:
        return positive_support.clamp(0.0, 1.0)

    uncertainty_penalty = uncertainty.clamp(0.0, 1.0)
    uncertainty_gate = 1.0 - uncertainty_weight * uncertainty_penalty
    uncertainty_gate = uncertainty_gate.clamp(0.0, 1.0)
    affinity = positive_support * uncertainty_gate
    return affinity.clamp(0.0, 1.0)
