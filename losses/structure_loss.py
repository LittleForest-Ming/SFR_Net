from __future__ import annotations

import torch
import torch.nn as nn


class StructureLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _zero(self, outputs, targets):
        reference = outputs['fields']['center'] if outputs['fields'].get('center') is not None else targets['center']
        return reference.new_zeros(())

    def forward(self, outputs, targets):
        zero = self._zero(outputs, targets)
        structure = outputs['refined'].get('structure')
        orientation = outputs['fields'].get('orientation')
        continuity = outputs['fields'].get('continuity')
        orientation_mask = targets['valid_masks'].get('orientation')
        inter_row_mask = targets['aux'].get('inter_row_mask')

        direction_smooth = zero
        if self.cfg.loss.use_direction_smooth and structure is not None and orientation is not None and orientation_mask is not None:
            gx = structure[:, :, :, 1:] - structure[:, :, :, :-1]
            gy = structure[:, :, 1:, :] - structure[:, :, :-1, :]
            direction_smooth = gx.abs().mean() + gy.abs().mean()

        continuity_preserve = zero
        if self.cfg.loss.use_continuity_preserve and continuity is not None and structure is not None:
            continuity_preserve = (structure - continuity).abs().mean()

        inter_row_separation = zero
        if self.cfg.loss.use_inter_row_separation and inter_row_mask is not None and structure is not None:
            inter_row_separation = (structure * inter_row_mask).mean()

        return direction_smooth + continuity_preserve + inter_row_separation
