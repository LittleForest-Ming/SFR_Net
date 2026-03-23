from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class FieldHeads(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.use_continuity = bool(cfg.model.use_continuity)
        self.use_uncertainty = bool(cfg.model.use_uncertainty)
        self.center_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.orientation_head = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.continuity_head = nn.Conv2d(in_channels, 1, kernel_size=1) if self.use_continuity else None
        self.uncertainty_head = nn.Conv2d(in_channels, 1, kernel_size=1) if self.use_uncertainty else None

    def forward(self, x):
        center = torch.sigmoid(self.center_head(x))
        orientation = F.normalize(self.orientation_head(x), dim=1, eps=1e-6)

        continuity = None
        if self.continuity_head is not None:
            continuity = torch.sigmoid(self.continuity_head(x))

        uncertainty = None
        if self.uncertainty_head is not None:
            # Full-stage uncertainty is a dense probability map with shape [B, 1, H, W].
            uncertainty = torch.sigmoid(self.uncertainty_head(x))

        return {
            'center': center,
            'orientation': orientation,
            'continuity': continuity,
            'uncertainty': uncertainty,
        }
