from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_annotation(path: str | Path) -> list[dict]:
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        data = json.load(handle)
    if isinstance(data, dict) and 'samples' in data:
        data = data['samples']
    if not isinstance(data, list):
        raise TypeError('Annotation file must contain a list of samples or a dict with a samples field.')
    return [validate_sample(item) for item in data]


def validate_sample(sample: dict) -> dict:
    if 'rows' not in sample:
        sample['rows'] = []
    rows = []
    for row in sample['rows']:
        if len(row) < 2:
            continue
        rows.append([[float(x), float(y)] for x, y in row])
    sample['rows'] = rows
    return sample


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    return np.asarray(image)


def save_json(path: str | Path, payload: dict | list) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
