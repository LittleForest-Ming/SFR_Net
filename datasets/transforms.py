from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Resize:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: dict) -> dict:
        image = Image.fromarray(sample['image'])
        old_w, old_h = image.size
        image = image.resize((self.width, self.height), Image.BILINEAR)
        sx = self.width / max(old_w, 1)
        sy = self.height / max(old_h, 1)
        sample['image'] = np.asarray(image)
        sample['row_polylines'] = [[[x * sx, y * sy] for x, y in row] for row in sample['row_polylines']]
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample
        image = Image.fromarray(sample['image'])
        flipped = ImageOps.mirror(image)
        width = flipped.size[0]
        sample['image'] = np.asarray(flipped)
        sample['row_polylines'] = [[[width - 1 - x, y] for x, y in row] for row in sample['row_polylines']]
        return sample


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.15, contrast: float = 0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample: dict) -> dict:
        image = Image.fromarray(sample['image'])
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        sample['image'] = np.asarray(image)
        return sample


class MildAffine:
    def __init__(self, max_rotation: float = 5.0, max_translate: float = 0.03):
        self.max_rotation = max_rotation
        self.max_translate = max_translate

    @staticmethod
    def _build_forward_matrix(width: int, height: int, angle: float, tx: float, ty: float) -> np.ndarray:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        radians = math.radians(angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        to_center = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        rotate = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        from_center = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        translate = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)
        return translate @ from_center @ rotate @ to_center

    @staticmethod
    def _apply_to_points(rows, matrix: np.ndarray):
        transformed_rows = []
        for row in rows:
            transformed = []
            for x, y in row:
                xy1 = np.array([x, y, 1.0], dtype=np.float32)
                xr, yr, _ = matrix @ xy1
                transformed.append([float(xr), float(yr)])
            transformed_rows.append(transformed)
        return transformed_rows

    def __call__(self, sample: dict) -> dict:
        image = Image.fromarray(sample['image'])
        width, height = image.size
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        tx = random.uniform(-self.max_translate, self.max_translate) * width
        ty = random.uniform(-self.max_translate, self.max_translate) * height
        forward = self._build_forward_matrix(width, height, angle, tx, ty)
        inverse = np.linalg.inv(forward)
        coeffs = inverse[:2, :].reshape(-1).tolist()
        transformed_image = image.transform(
            image.size,
            Image.AFFINE,
            data=coeffs,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        sample['image'] = np.asarray(transformed_image)
        sample['row_polylines'] = self._apply_to_points(sample['row_polylines'], forward)
        return sample


def build_transforms(cfg, split: str):
    transforms = [Resize(cfg.dataset.input_height, cfg.dataset.input_width)]
    if split == cfg.dataset.train_split:
        augment = cfg.train.augment
        if augment.hflip:
            transforms.append(RandomHorizontalFlip())
        if augment.brightness_contrast:
            transforms.append(RandomBrightnessContrast())
        if augment.mild_affine:
            transforms.append(MildAffine())
    return Compose(transforms)
