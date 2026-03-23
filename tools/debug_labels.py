from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ''}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sfr_net.datasets.field_label_generator import FieldLabelGenerator
    from sfr_net.datasets.io import load_annotation, load_image
    from sfr_net.utils.config import add_config_args, load_config
    from sfr_net.utils.visualization import overlay_center_heatmap
else:
    from ..datasets.field_label_generator import FieldLabelGenerator
    from ..datasets.io import load_annotation, load_image
    from ..utils.config import add_config_args, load_config
    from ..utils.visualization import overlay_center_heatmap


DEFAULT_CONFIG = 'sfr_net/configs/base.yaml'


def _resolve_annotation_path(annotation_arg: str | None, cfg) -> Path:
    if annotation_arg:
        return Path(annotation_arg)
    return Path(cfg.dataset.root) / f'{cfg.dataset.train_split}.json'


def _resolve_output_dir(output_dir_arg: str | None, cfg) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return Path(cfg.project.output_dir) / 'debug_labels'


def _resolve_image_path(sample: dict, annotation_path: Path, cfg) -> Path:
    image_rel = sample.get('image_path')
    if not image_rel:
        raise ValueError('Sample does not contain image_path.')

    candidates = [
        Path(image_rel),
        annotation_path.parent / image_rel,
        Path(cfg.dataset.root) / image_rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not resolve image_path={image_rel!r} from annotation {annotation_path}.')


def _draw_polylines(image_np: np.ndarray, row_polylines: list[list[list[float]]]) -> np.ndarray:
    image = Image.fromarray(image_np.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for row in row_polylines:
        if len(row) >= 2:
            draw.line([tuple(point) for point in row], fill=(0, 255, 0), width=2)
        for x, y in row:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
    return np.asarray(image)


def _to_gray_rgb(array_2d: np.ndarray, scale: float = 255.0) -> np.ndarray:
    norm = np.clip(array_2d * scale, 0.0, 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def _orientation_valid_vis(valid_mask: np.ndarray) -> np.ndarray:
    mask = (valid_mask > 0).astype(np.uint8) * 255
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    vis[..., 1] = mask
    return vis


def _draw_orientation_vectors(image_np: np.ndarray, orientation: np.ndarray, valid_mask: np.ndarray, step: int = 16, scale: float = 10.0) -> np.ndarray:
    image = Image.fromarray(image_np.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(image)
    h, w = valid_mask.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            if valid_mask[y, x] <= 0:
                continue
            dx = float(orientation[0, y, x]) * scale
            dy = float(orientation[1, y, x]) * scale
            draw.line((x, y, x + dx, y + dy), fill=(255, 255, 0), width=1)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 0))
    return np.asarray(image)


def _save_image(path: Path, image_np: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np.astype(np.uint8)).save(path)


def debug_single_sample(cfg, annotation_path: Path, output_dir: Path, index: int = 0) -> None:
    samples = load_annotation(annotation_path)
    if not samples:
        raise ValueError(f'Annotation file {annotation_path} does not contain any valid samples.')
    if index < 0 or index >= len(samples):
        raise IndexError(f'Sample index {index} is out of range for {annotation_path}; valid range is [0, {len(samples) - 1}].')

    sample = samples[index]
    image_path = _resolve_image_path(sample, annotation_path, cfg)
    image_np = load_image(image_path)
    row_polylines = sample.get('rows', [])

    generator = FieldLabelGenerator(cfg)
    labels = generator(image_np.shape[:2], row_polylines)

    center = labels['center'].numpy()
    orientation = labels['orientation'].numpy()
    center_mask = labels['valid_masks']['center'].numpy()
    orientation_mask = labels['valid_masks']['orientation'].numpy()
    row_raster = labels['aux']['row_raster'].numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = f"{Path(sample.get('image_path', image_path.stem)).stem}_idx{index:03d}"

    _save_image(output_dir / f'{sample_name}_image.png', image_np)
    _save_image(output_dir / f'{sample_name}_rows_overlay.png', _draw_polylines(image_np, row_polylines))
    _save_image(output_dir / f'{sample_name}_center_heatmap.png', overlay_center_heatmap(image_np, center[0]))
    _save_image(output_dir / f'{sample_name}_orientation_valid.png', _orientation_valid_vis(orientation_mask[0]))
    _save_image(output_dir / f'{sample_name}_orientation_vectors.png', _draw_orientation_vectors(image_np, orientation, orientation_mask[0]))
    _save_image(output_dir / f'{sample_name}_row_raster.png', _to_gray_rgb(row_raster[0]))

    center_coverage = float(center_mask.mean())
    orientation_coverage = float(orientation_mask.mean())

    print(f'Annotation: {annotation_path}')
    print(f'Sample index: {index}')
    print(f'Image: {image_path}')
    print(f'center shape/min/max: {tuple(center.shape)} / {center.min():.6f} / {center.max():.6f}')
    print(f'orientation shape/min/max: {tuple(orientation.shape)} / {orientation.min():.6f} / {orientation.max():.6f}')
    print(f'valid mask coverage: center={center_coverage:.6f}, orientation={orientation_coverage:.6f}')
    print(f'Outputs saved to: {output_dir}')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Debug SFR-Net center/orientation label generation for a single annotation file.')
    add_config_args(parser)
    parser.set_defaults(config=DEFAULT_CONFIG)
    parser.add_argument('--annotation', default=None, help='Path to a single annotation JSON file. Defaults to <dataset.root>/<train_split>.json')
    parser.add_argument('--output-dir', default=None, help='Directory to save debug visualizations. Defaults to <project.output_dir>/debug_labels')
    parser.add_argument('--index', type=int, default=0, help='Sample index inside the annotation file. Default: 0')
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.set)
        annotation_path = _resolve_annotation_path(args.annotation, cfg)
        if not annotation_path.exists():
            raise FileNotFoundError(
                f'Annotation file not found: {annotation_path}. '
                f'You can set dataset.root/train_split in {args.config} or pass --annotation explicitly.'
            )
        output_dir = _resolve_output_dir(args.output_dir, cfg)
        debug_single_sample(cfg, annotation_path, output_dir, index=args.index)
    except Exception as exc:
        raise SystemExit(f'[debug_labels] Failed: {exc}') from exc


if __name__ == '__main__':
    main()
