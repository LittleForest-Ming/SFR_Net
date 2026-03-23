from __future__ import annotations

import torch.nn as nn

from .center_loss import CenterLoss
from .continuity_loss import ContinuityLoss
from .orientation_loss import OrientationLoss
from .structure_loss import StructureLoss
from .uncertainty_loss import UncertaintyLoss


class SFRCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.center_loss = CenterLoss()
        self.orientation_loss = OrientationLoss()
        self.continuity_loss = ContinuityLoss()
        self.structure_loss = StructureLoss(cfg)
        self.uncertainty_loss = UncertaintyLoss(loss_type=cfg.loss.uncertainty_type)

    def forward(self, outputs, targets):
        center = self.center_loss(outputs['fields']['center'], targets['center'], targets['valid_masks']['center'])
        orientation, orientation_norm = self.orientation_loss(outputs['fields']['orientation'], targets['orientation'], targets['valid_masks']['orientation'])

        zero = center.new_zeros(())

        continuity = zero
        if self.cfg.model.use_continuity:
            continuity = self.continuity_loss(outputs['fields']['continuity'], targets['continuity'], targets['valid_masks']['continuity'])

        structure = zero
        if self.cfg.loss.lambda_structure > 0:
            structure = self.structure_loss(outputs, targets)

        uncertainty = zero
        uncertainty_pred = outputs['fields'].get('uncertainty')
        uncertainty_target = targets.get('uncertainty')
        uncertainty_mask = targets['valid_masks'].get('uncertainty')
        use_uncertainty_loss = (
            bool(self.cfg.model.use_uncertainty)
            and float(self.cfg.loss.lambda_uncertainty) > 0.0
            and uncertainty_pred is not None
            and uncertainty_target is not None
        )
        if use_uncertainty_loss:
            uncertainty = self.uncertainty_loss(uncertainty_pred, uncertainty_target, uncertainty_mask)

        total = (
            self.cfg.loss.lambda_center * center
            + self.cfg.loss.lambda_orientation * orientation
            + self.cfg.loss.lambda_orientation_norm * orientation_norm
            + self.cfg.loss.lambda_continuity * continuity
            + self.cfg.loss.lambda_structure * structure
            + self.cfg.loss.lambda_uncertainty * uncertainty
        )
        return {
            'total': total,
            'items': {
                'center': center,
                'orientation': orientation,
                'orientation_norm': orientation_norm,
                'continuity': continuity,
                'structure': structure,
                'uncertainty': uncertainty,
            },
            'stats': {
                'num_pos_center': int((targets['center'] > 0.5).sum().item()) if targets['center'] is not None else 0,
                'num_pos_orientation': int(targets['valid_masks']['orientation'].sum().item()) if targets['valid_masks']['orientation'] is not None else 0,
                'num_pos_continuity': int(targets['valid_masks']['continuity'].sum().item()) if targets['valid_masks']['continuity'] is not None else 0,
            },
        }
