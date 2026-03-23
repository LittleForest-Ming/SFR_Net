from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .field_label_generator import FieldLabelGenerator
from .io import load_annotation, load_image
from .transforms import build_transforms


class CropRowDataset(Dataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.dataset.root)
        self.annotation_file = self.root / f'{split}.json'
        self.samples = load_annotation(self.annotation_file) if self.annotation_file.exists() else []
        self.label_generator = FieldLabelGenerator(cfg)
        self.transforms = build_transforms(cfg, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = self.root / item['image_path'] if item.get('image_path') else None
        if image_path is not None and image_path.exists():
            image = load_image(image_path)
        else:
            image = np.zeros((self.cfg.dataset.input_height, self.cfg.dataset.input_width, 3), dtype=np.uint8)
        sample = {'image': image, 'row_polylines': item.get('rows', [])}
        sample = self.transforms(sample)
        image_np = np.array(sample['image'], copy=True)
        if image_np.ndim != 3:
            raise ValueError('Expected image to have shape [H, W, C].')
        targets = self.label_generator(image_np.shape[:2], sample['row_polylines'])
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        if self.cfg.dataset.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        return {
            'image': image_tensor,
            'targets': {
                'center': targets['center'],
                'orientation': targets['orientation'],
                'continuity': targets['continuity'],
                'uncertainty': targets['uncertainty'],
                'valid_masks': targets['valid_masks'],
                'aux': {
                    'row_polylines': sample['row_polylines'],
                    'row_raster': targets['aux']['row_raster'],
                    'inter_row_mask': targets['aux']['inter_row_mask'],
                    'narrow_band_mask': targets['aux']['narrow_band_mask'],
                },
            },
            'meta': {
                'image_id': str(item.get('image_id', idx)),
                'orig_size': tuple(item.get('orig_size', image_np.shape[:2])),
                'input_size': tuple(image_np.shape[:2]),
                'path': str(image_path) if image_path is not None else '',
            },
        }
