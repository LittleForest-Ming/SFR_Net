from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels])

    def forward(self, feats):
        laterals = [conv(feat) for conv, feat in zip(self.lateral, feats)]
        for idx in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[idx], size=laterals[idx - 1].shape[-2:], mode='bilinear', align_corners=False)
            laterals[idx - 1] = laterals[idx - 1] + upsampled
        outputs = [conv(lat) for conv, lat in zip(self.smooth, laterals)]
        return outputs[0]
