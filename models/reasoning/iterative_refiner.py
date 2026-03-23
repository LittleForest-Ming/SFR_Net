from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .affinity import compute_local_affinity


class IterativeStructureRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        self.register_buffer('smooth_kernel', kernel)

    def _cfg_value(self, path: str, default: float) -> float:
        current = self.cfg
        for name in path.split('.'):
            if current is None or not hasattr(current, name):
                return float(default)
            current = getattr(current, name)
        try:
            return float(current)
        except (TypeError, ValueError):
            return float(default)

    def forward(self, fields: dict) -> dict:
        center = fields['center']
        continuity = fields.get('continuity')
        uncertainty = fields.get('uncertainty')
        if continuity is None:
            return {'structure': center, 'fields': fields, 'debug': {'iterations': [center]}}

        structure = center.clone()
        iterations = [structure]
        num_iters = max(int(self.cfg.model.reasoning_num_iters), 1)
        structure_mix = self._cfg_value('model.refiner_structure_mix', 0.50)
        smooth_mix = self._cfg_value('model.refiner_smooth_mix', 0.25)
        affinity_mix = self._cfg_value('model.refiner_affinity_mix', 0.25)
        uncertainty_strength = self._cfg_value('model.refiner_uncertainty_strength', 1.0)

        total_mix = max(structure_mix + smooth_mix + affinity_mix, 1e-6)
        structure_mix /= total_mix
        smooth_mix /= total_mix
        affinity_mix /= total_mix

        for _ in range(num_iters):
            affinity = compute_local_affinity(
                center,
                continuity,
                fields['orientation'],
                structure,
                uncertainty=uncertainty,
                cfg=self.cfg,
            )
            smoothed = F.conv2d(structure, self.smooth_kernel, padding=1)

            if uncertainty is not None:
                uncertainty_gate = 1.0 - uncertainty_strength * uncertainty.clamp(0.0, 1.0)
                uncertainty_gate = uncertainty_gate.clamp(0.0, 1.0)
                smoothed = smoothed * uncertainty_gate + structure * (1.0 - uncertainty_gate)
                affinity = affinity * uncertainty_gate

            structure = torch.clamp(
                structure_mix * structure + smooth_mix * smoothed + affinity_mix * affinity,
                0.0,
                1.0,
            )
            iterations.append(structure)
        return {'structure': structure, 'fields': {**fields, 'structure': structure}, 'debug': {'iterations': iterations}}
