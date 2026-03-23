from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in {None, ''}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sfr_net.datasets.crop_row_dataset import CropRowDataset
    from sfr_net.datasets.io import load_image
    from sfr_net.models.decoder import StructuralDecoder
    from sfr_net.utils.config import add_config_args, load_config
    from sfr_net.utils.geometry import polyline_length
    from sfr_net.utils.visualization import overlay_center_heatmap
else:
    from ..datasets.crop_row_dataset import CropRowDataset
    from ..datasets.io import load_image
    from ..models.decoder import StructuralDecoder
    from ..utils.config import add_config_args, load_config
    from ..utils.geometry import polyline_length
    from ..utils.visualization import overlay_center_heatmap


DEFAULT_CONFIG = 'sfr_net/configs/base.yaml'


def _resolve_output_dir(output_dir_arg: str | None, cfg) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return Path(cfg.project.output_dir) / 'debug_decoder_with_gt'


def _save_image(path: Path, image_np: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np.astype(np.uint8)).save(path)


def _extract_points(row):
    if isinstance(row, dict):
        return row.get('points', [])
    return row


def _draw_rows(image_np: np.ndarray, rows, color=(0, 255, 0), width: int = 2, seeds=None) -> np.ndarray:
    image = Image.fromarray(image_np.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for row in rows:
        points = _extract_points(row)
        if len(points) >= 2:
            draw.line([tuple(point) for point in points], fill=color, width=width)
    for seed in seeds or []:
        x, y = seed
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 0, 0))
    return np.asarray(image)


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


def _make_compare_vis(image_np: np.ndarray, gt_rows, decoded_rows) -> np.ndarray:
    gt_vis = _draw_rows(image_np, gt_rows, color=(0, 255, 0), width=2)
    pred_vis = _draw_rows(image_np, decoded_rows, color=(255, 255, 0), width=2)
    gt_img = Image.fromarray(gt_vis)
    pred_img = Image.fromarray(pred_vis)
    canvas = Image.new('RGB', (gt_img.width * 2, gt_img.height), color=(30, 30, 30))
    canvas.paste(gt_img, (0, 0))
    canvas.paste(pred_img, (gt_img.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), 'GT rows', fill=(255, 255, 255))
    draw.text((gt_img.width + 10, 10), 'Decoded rows', fill=(255, 255, 255))
    return np.asarray(canvas)


def _tensor_to_numpy_image(sample: dict) -> np.ndarray:
    image_path = sample['meta'].get('path', '')
    if image_path and Path(image_path).exists():
        return load_image(image_path)
    image = sample['image'].detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Debug SFR-Net decoder by feeding GT center and GT orientation directly.')
    add_config_args(parser)
    parser.set_defaults(config=DEFAULT_CONFIG)
    parser.add_argument('--split', default='train', help='Dataset split to inspect. Default: train')
    parser.add_argument('--index', type=int, default=0, help='Sample index inside the dataset split. Default: 0')
    parser.add_argument('--output-dir', default=None, help='Directory to save debug visualizations. Defaults to <project.output_dir>/debug_decoder_with_gt')
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.set)
        dataset = CropRowDataset(cfg, args.split)
        if len(dataset) == 0:
            raise ValueError(f'Dataset split {args.split!r} is empty. Please check your annotation file.')
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f'Sample index {args.index} is out of range for split {args.split!r}; valid range is [0, {len(dataset) - 1}].')

        sample = dataset[args.index]
        image_np = _tensor_to_numpy_image(sample)
        gt_rows = sample['targets']['aux']['row_polylines']
        center = sample['targets']['center'].unsqueeze(0)
        orientation = sample['targets']['orientation'].unsqueeze(0)
        continuity = sample['targets']['continuity']
        uncertainty = sample['targets']['uncertainty']
        if continuity is not None:
            continuity = continuity.unsqueeze(0)
        if uncertainty is not None:
            uncertainty = uncertainty.unsqueeze(0)

        fields = {
            'center': center,
            'orientation': orientation,
            'continuity': continuity,
            'uncertainty': uncertainty,
        }

        decoder = StructuralDecoder(cfg)
        decoded = decoder.decode(fields=fields, refined=None, meta=[sample['meta']])

        seeds = decoded['seeds'][0]
        raw_paths = decoded['debug']['candidate_paths'][0]
        final_rows = decoded['rows'][0]

        raw_rows = [
            {
                'points': points,
                'score': 0.0,
                'length': polyline_length(points),
                'source_seed': points[0] if points else None,
            }
            for points in raw_paths if points
        ]

        output_dir = _resolve_output_dir(args.output_dir, cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f'decoder_gt_{args.split}_{args.index:03d}'

        center_np = sample['targets']['center'].numpy()[0]
        orientation_np = sample['targets']['orientation'].numpy()
        orientation_mask = sample['targets']['valid_masks']['orientation'].numpy()[0]
        row_raster = sample['targets']['aux']['row_raster'].numpy()[0] if sample['targets']['aux']['row_raster'] is not None else None

        _save_image(output_dir / f'{stem}_center_heatmap.png', overlay_center_heatmap(image_np, center_np))
        _save_image(output_dir / f'{stem}_orientation_valid.png', _orientation_valid_vis(orientation_mask))
        _save_image(output_dir / f'{stem}_orientation_vectors.png', _draw_orientation_vectors(image_np, orientation_np, orientation_mask))
        _save_image(output_dir / f'{stem}_seed_overlay.png', _draw_rows(image_np, [], seeds=seeds))
        _save_image(output_dir / f'{stem}_raw_trajectory_overlay.png', _draw_rows(image_np, raw_rows, color=(255, 165, 0), width=2, seeds=seeds))
        _save_image(output_dir / f'{stem}_final_trajectory_overlay.png', _draw_rows(image_np, final_rows, color=(0, 255, 255), width=2, seeds=seeds))
        _save_image(output_dir / f'{stem}_gt_vs_decoded.png', _make_compare_vis(image_np, gt_rows, final_rows))
        if row_raster is not None:
            raster_vis = np.stack([row_raster * 255.0, row_raster * 255.0, row_raster * 255.0], axis=-1).clip(0, 255).astype(np.uint8)
            _save_image(output_dir / f'{stem}_row_raster.png', raster_vis)

        print(f'Sample: split={args.split}, index={args.index}')
        print(f'Seed count: {len(seeds)}')
        print(f'Raw trajectory count: {len(raw_rows)}')
        print(f'Final trajectory count: {len(final_rows)}')
        print('Raw trajectory lengths:')
        for idx, row in enumerate(raw_rows):
            print(f'  raw[{idx}]: {row["length"]:.3f}')
        print('Final trajectory lengths:')
        for idx, row in enumerate(final_rows):
            print(f'  final[{idx}]: {row["length"]:.3f}')
        print(f'Filtering summary: before={len(raw_rows)}, after={len(final_rows)}')
        print(f'Outputs saved to: {output_dir}')
    except Exception as exc:
        raise SystemExit(f'[debug_decoder_with_gt] Failed: {exc}') from exc


if __name__ == '__main__':
    main()
