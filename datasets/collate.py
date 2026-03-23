from __future__ import annotations

import torch


TARGET_KEYS = ('center', 'orientation', 'continuity', 'uncertainty')
MASK_KEYS = ('center', 'orientation', 'continuity', 'uncertainty')
AUX_KEYS = ('row_raster', 'inter_row_mask', 'narrow_band_mask')


def _stack_or_none(values):
    return None if values[0] is None else torch.stack(values, dim=0)


def crop_row_collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = {key: _stack_or_none([item['targets'][key] for item in batch]) for key in TARGET_KEYS}
    targets['valid_masks'] = {key: _stack_or_none([item['targets']['valid_masks'][key] for item in batch]) for key in MASK_KEYS}
    targets['aux'] = {'row_polylines': [item['targets']['aux']['row_polylines'] for item in batch]}
    for key in AUX_KEYS:
        targets['aux'][key] = _stack_or_none([item['targets']['aux'][key] for item in batch])
    return {'image': images, 'targets': targets, 'meta': [item['meta'] for item in batch]}


collate_batch = crop_row_collate_fn
