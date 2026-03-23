from __future__ import annotations

import torch
import torch.nn.functional as F


class SeedGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, center_map):
        if center_map.ndim == 4:
            center_map = center_map[:, 0]
        threshold = float(self.cfg.infer.seed_threshold)
        kernel = int(self.cfg.infer.seed_nms_kernel)
        if kernel % 2 == 0:
            kernel += 1
        pooled = F.max_pool2d(center_map.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)
        keep = (center_map >= pooled) & (center_map >= threshold)
        if self.cfg.infer.restrict_seed_region:
            h = center_map.shape[-2]
            y_start = h // 2 if self.cfg.infer.seed_region == 'lower_half' else 0
            region_mask = torch.zeros_like(center_map, dtype=torch.bool)
            region_mask[:, y_start:, :] = True
            keep = keep & region_mask
        all_seeds = []
        for batch_idx in range(center_map.shape[0]):
            ys, xs = torch.where(keep[batch_idx])
            seeds = [[float(x.item()), float(y.item())] for y, x in zip(ys, xs)]
            all_seeds.append(seeds)
        return all_seeds
