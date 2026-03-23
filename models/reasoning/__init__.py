from __future__ import annotations

import torch.nn as nn

from .affinity import compute_local_affinity, direction_consistency, distance_feasibility
from .iterative_refiner import IterativeStructureRefiner
from .transformer_refiner import TransformerStructureRefiner


def build_structure_refiner(cfg):
    """Build the structure refiner selected by cfg.model.use_reasoning and cfg.model.reasoning_mode."""
    if not bool(getattr(cfg.model, 'use_reasoning', False)):
        return None

    reasoning_mode = str(getattr(cfg.model, 'reasoning_mode', 'explicit')).lower()
    if reasoning_mode == 'explicit':
        return IterativeStructureRefiner(cfg)
    if reasoning_mode == 'transformer':
        return TransformerStructureRefiner(cfg)
    raise ValueError(
        f'Unsupported reasoning mode: {reasoning_mode}. '
        'Expected one of: explicit, transformer.'
    )


class StructuralReasoner(nn.Module):
    """Backward-compatible wrapper around the selected structure refiner.

    The main model now builds the refiner directly, but this wrapper is kept so any
    older internal imports continue to receive the same refined structure contract.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.refiner = build_structure_refiner(cfg)

    def forward(self, fields: dict) -> dict:
        if self.refiner is None:
            return {'structure': fields['center'], 'fields': fields, 'debug': {'affinity': None, 'iterations': None, 'mode': None}}
        refined = self.refiner(fields)
        return {
            'structure': refined['structure'],
            'fields': refined['fields'],
            'debug': {
                'affinity': None,
                'iterations': refined['debug'].get('iterations'),
                'mode': refined['debug'].get('mode', str(getattr(self.cfg.model, 'reasoning_mode', 'explicit')).lower()),
            },
        }


__all__ = [
    'StructuralReasoner',
    'IterativeStructureRefiner',
    'TransformerStructureRefiner',
    'build_structure_refiner',
    'compute_local_affinity',
    'direction_consistency',
    'distance_feasibility',
]
