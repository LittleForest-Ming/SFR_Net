from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ''}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sfr_net.datasets.collate import crop_row_collate_fn
    from sfr_net.datasets.crop_row_dataset import CropRowDataset
    from sfr_net.utils.config import add_config_args, load_config
else:
    from ..datasets.collate import crop_row_collate_fn
    from ..datasets.crop_row_dataset import CropRowDataset
    from ..utils.config import add_config_args, load_config


DEFAULT_CONFIG = 'sfr_net/configs/base.yaml'


def _format_tensor_stats(tensor: torch.Tensor) -> str:
    if tensor.numel() == 0:
        return f'shape={tuple(tensor.shape)} dtype={tensor.dtype} empty'
    tensor_float = tensor.detach().float()
    return (
        f'shape={tuple(tensor.shape)} '
        f'dtype={tensor.dtype} '
        f'min={tensor_float.min().item():.6f} '
        f'max={tensor_float.max().item():.6f}'
    )


def _print_tree(name: str, obj: Any, indent: int = 0) -> None:
    prefix = ' ' * indent
    if isinstance(obj, dict):
        print(f'{prefix}{name}: dict')
        for key, value in obj.items():
            _print_tree(str(key), value, indent + 2)
    elif isinstance(obj, list):
        print(f'{prefix}{name}: list(len={len(obj)})')
        if obj:
            first = obj[0]
            if isinstance(first, dict):
                _print_tree('[0]', first, indent + 2)
            elif isinstance(first, list):
                print(f'{prefix}  [0]: list(len={len(first)})')
            else:
                print(f'{prefix}  [0]: {type(first).__name__}')
    elif isinstance(obj, tuple):
        print(f'{prefix}{name}: tuple(len={len(obj)})')
    elif torch.is_tensor(obj):
        print(f'{prefix}{name}: tensor {_format_tensor_stats(obj)}')
    elif obj is None:
        print(f'{prefix}{name}: None')
    else:
        print(f'{prefix}{name}: {type(obj).__name__} = {obj}')


def _print_tensor_report(name: str, obj: Any, path: str = '') -> None:
    current_path = f'{path}.{name}' if path else name
    if isinstance(obj, dict):
        for key, value in obj.items():
            _print_tensor_report(str(key), value, current_path)
    elif isinstance(obj, list):
        if name == 'row_polylines':
            print(f'{current_path}: list(len={len(obj)})')
        elif obj and isinstance(obj[0], dict):
            _print_tensor_report('[0]', obj[0], current_path)
        else:
            print(f'{current_path}: list(len={len(obj)})')
    elif torch.is_tensor(obj):
        print(f'{current_path}: {_format_tensor_stats(obj)}')
    elif obj is None:
        print(f'{current_path}: None')


def _assert_protocol_single(sample: dict) -> None:
    if 'image' not in sample or 'targets' not in sample or 'meta' not in sample:
        raise ValueError('Single sample must contain image, targets, and meta.')
    targets = sample['targets']
    for key in ('center', 'orientation', 'continuity', 'uncertainty', 'valid_masks', 'aux'):
        if key not in targets:
            raise ValueError(f'Single sample targets missing key: {key}')
    if 'row_polylines' not in targets['aux']:
        raise ValueError('Single sample targets.aux missing row_polylines.')
    if not isinstance(targets['aux']['row_polylines'], list):
        raise TypeError('Single sample targets.aux.row_polylines must be a list.')
    if 'center' not in targets['valid_masks'] or 'orientation' not in targets['valid_masks']:
        raise ValueError('Single sample valid_masks missing center/orientation.')


def _assert_protocol_batch(batch: dict) -> None:
    if 'image' not in batch or 'targets' not in batch or 'meta' not in batch:
        raise ValueError('Batch must contain image, targets, and meta.')
    targets = batch['targets']
    for key in ('center', 'orientation', 'continuity', 'uncertainty', 'valid_masks', 'aux'):
        if key not in targets:
            raise ValueError(f'Batch targets missing key: {key}')
    if 'row_polylines' not in targets['aux']:
        raise ValueError('Batch targets.aux missing row_polylines.')
    if not isinstance(targets['aux']['row_polylines'], list):
        raise TypeError('Batch targets.aux.row_polylines must stay a list.')
    if 'center' not in targets['valid_masks'] or 'orientation' not in targets['valid_masks']:
        raise ValueError('Batch valid_masks missing center/orientation.')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Debug SFR-Net CropRowDataset and collate_fn return protocol.')
    add_config_args(parser)
    parser.set_defaults(config=DEFAULT_CONFIG)
    parser.add_argument('--split', default='train', help='Dataset split to inspect. Default: train')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for DataLoader. Default: 2')
    parser.add_argument('--num-samples', type=int, default=2, help='Number of samples to inspect. Default: 2')
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.set)
        dataset = CropRowDataset(cfg, args.split)
        if len(dataset) == 0:
            raise ValueError(f'Dataset split {args.split!r} is empty. Please check your annotation file.')

        inspect_count = min(max(args.num_samples, 1), len(dataset))
        print(f'Dataset split: {args.split}')
        print(f'Dataset size: {len(dataset)}')
        print(f'Inspecting first {inspect_count} sample(s)')
        print('')

        sample = dataset[0]
        _assert_protocol_single(sample)
        print('=== Single Sample Key Tree ===')
        _print_tree('sample', sample)
        print('')
        print('=== Single Sample Tensor Report ===')
        _print_tensor_report('sample', sample)
        print('')
        print('=== Single Sample Protocol Checks ===')
        print(f"targets.continuity is None: {sample['targets']['continuity'] is None}")
        print(f"targets.uncertainty is None: {sample['targets']['uncertainty'] is None}")
        print(f"targets.valid_masks exists: {'valid_masks' in sample['targets']}")
        print(f"targets.aux.row_polylines is list: {isinstance(sample['targets']['aux']['row_polylines'], list)}")
        print('')

        indices = list(range(inspect_count))
        samples = [dataset[idx] for idx in indices]
        batch = crop_row_collate_fn(samples)
        _assert_protocol_batch(batch)
        print('=== Batch Key Tree ===')
        _print_tree('batch', batch)
        print('')
        print('=== Batch Tensor Report ===')
        _print_tensor_report('batch', batch)
        print('')
        print('=== Batch Protocol Checks ===')
        print(f"targets.continuity is None: {batch['targets']['continuity'] is None}")
        print(f"targets.uncertainty is None: {batch['targets']['uncertainty'] is None}")
        print(f"targets.valid_masks exists: {'valid_masks' in batch['targets']}")
        print(f"targets.aux.row_polylines is list: {isinstance(batch['targets']['aux']['row_polylines'], list)}")
        print('')

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=crop_row_collate_fn)
        first_loader_batch = next(iter(loader))
        _assert_protocol_batch(first_loader_batch)
        print('=== DataLoader First Batch Summary ===')
        print(f"image: {_format_tensor_stats(first_loader_batch['image'])}")
        print(f"targets.center: {_format_tensor_stats(first_loader_batch['targets']['center'])}")
        print(f"targets.orientation: {_format_tensor_stats(first_loader_batch['targets']['orientation'])}")
        print(f"targets.continuity is None: {first_loader_batch['targets']['continuity'] is None}")
        print(f"targets.uncertainty is None: {first_loader_batch['targets']['uncertainty'] is None}")
        print(f"targets.valid_masks keys: {list(first_loader_batch['targets']['valid_masks'].keys())}")
        print(f"targets.aux.row_polylines type: {type(first_loader_batch['targets']['aux']['row_polylines']).__name__}")
    except Exception as exc:
        raise SystemExit(f'[debug_dataset] Failed: {exc}') from exc


if __name__ == '__main__':
    main()
