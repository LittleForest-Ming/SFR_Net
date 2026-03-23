from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from sfr_net.datasets.crop_row_dataset import CropRowDataset
    from sfr_net.datasets.io import load_image
    from sfr_net.engine.checkpoint import load_checkpoint
    from sfr_net.models.sfr_net import SFRNet
    from sfr_net.utils.config import add_config_args, load_config
    from sfr_net.utils.visualization import (
        continuity_evidence_panel,
        make_prediction_summary_panel,
        refinement_evidence_panel,
        save_comparison_visualization,
        save_field_visualizations,
        save_image,
        uncertainty_evidence_panel,
    )
else:
    from ..datasets.crop_row_dataset import CropRowDataset
    from ..datasets.io import load_image
    from ..engine.checkpoint import load_checkpoint
    from ..models.sfr_net import SFRNet
    from ..utils.config import add_config_args, load_config
    from ..utils.visualization import (
        continuity_evidence_panel,
        make_prediction_summary_panel,
        refinement_evidence_panel,
        save_comparison_visualization,
        save_field_visualizations,
        save_image,
        uncertainty_evidence_panel,
    )


MODES = (
    'label',
    'pred',
    'decode',
    'summary',
    'continuity_evidence',
    'uncertainty_evidence',
    'refinement_evidence',
)


def _load_model_weights(model: SFRNet, checkpoint_path: str) -> bool:
    if not checkpoint_path:
        return False
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
    model.load_state_dict(state_dict, strict=False)
    return True


def _denormalize_tensor_image(image_tensor: torch.Tensor, cfg) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if cfg.dataset.normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        image = image * std + mean
    image = (image * 255.0).clip(0, 255)
    return image.astype('uint8')


def _resolve_visual_image(sample: dict[str, Any], cfg) -> np.ndarray:
    meta = sample.get('meta', {})
    image_path = meta.get('path', '')
    if image_path and Path(image_path).exists():
        return load_image(image_path)
    return _denormalize_tensor_image(sample['image'], cfg)


def _load_dataset_sample(cfg, split: str, index: int) -> tuple[dict[str, Any], str]:
    dataset = CropRowDataset(cfg, split)
    if len(dataset) == 0:
        raise ValueError(f'Dataset split {split!r} is empty. Please check annotation files under {cfg.dataset.root}.')
    if index < 0 or index >= len(dataset):
        raise IndexError(f'Index {index} out of range for split {split!r} with {len(dataset)} samples.')
    sample = dataset[index]
    stem = f'{split}_{index:04d}'
    return sample, stem


def _load_input_sample(cfg, input_path: Path) -> tuple[dict[str, Any], str]:
    image_np = load_image(input_path)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 255.0
    if cfg.dataset.normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
    sample = {
        'image': image_tensor,
        'targets': {
            'center': None,
            'orientation': None,
            'continuity': None,
            'uncertainty': None,
            'valid_masks': {
                'center': None,
                'orientation': None,
                'continuity': None,
                'uncertainty': None,
            },
            'aux': {
                'row_polylines': [],
                'row_raster': None,
                'inter_row_mask': None,
                'narrow_band_mask': None,
                'continuity_band_mask': None,
                'uncertainty_source_mask': None,
            },
        },
        'meta': {'image_id': input_path.stem, 'path': str(input_path)},
    }
    return sample, input_path.stem


def _prepare_sample(cfg, args) -> tuple[dict[str, Any], np.ndarray, str, bool]:
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f'Input path not found: {input_path}')
        sample, stem = _load_input_sample(cfg, input_path)
        image_np = load_image(input_path)
        return sample, image_np, stem, True

    split = args.split or cfg.dataset.train_split
    sample, stem = _load_dataset_sample(cfg, split, args.index)
    image_np = _resolve_visual_image(sample, cfg)
    return sample, image_np, stem, False


def _run_model(cfg, sample: dict[str, Any], checkpoint_path: str) -> tuple[dict[str, Any], bool]:
    model = SFRNet(cfg)
    loaded = _load_model_weights(model, checkpoint_path)
    model.eval()
    with torch.no_grad():
        outputs = model(sample['image'].unsqueeze(0), sample['targets'], mode='infer')
        outputs['decoded'] = model.decoder.decode(outputs['fields'], refined=outputs['refined'], meta=[sample['meta']])
    return outputs, loaded


def _to_cpu_fields(outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pred_fields = {
        key: None if value is None else value[0].detach().cpu()
        for key, value in outputs['fields'].items()
    }
    pred_refined = {
        'structure': None if outputs['refined'].get('structure') is None else outputs['refined']['structure'][0].detach().cpu(),
    }

    decoded = outputs.get('decoded') or {}
    decoded_debug = decoded.get('debug', {})
    sample_debug = {
        'candidate_paths': decoded_debug.get('candidate_paths', [[]])[0] if decoded_debug.get('candidate_paths') is not None else [],
        'propagation': decoded_debug.get('propagation', [[]])[0] if decoded_debug.get('propagation') is not None else [],
        'seed_source': decoded_debug.get('seed_source', decoded_debug.get('structure_source', 'center')),
        'structure_source': decoded_debug.get('structure_source', 'center'),
        'use_continuity': decoded_debug.get('use_continuity', False),
        'use_uncertainty': decoded_debug.get('use_uncertainty', False),
    }
    pred_decoded = {
        'rows': decoded.get('rows', [[]])[0] if decoded.get('rows') is not None else [],
        'seeds': decoded.get('seeds', [[]])[0] if decoded.get('seeds') is not None else [],
        'scores': decoded.get('scores', [[]])[0] if decoded.get('scores') is not None else [],
        'debug': sample_debug,
    }
    return pred_fields, pred_refined, pred_decoded


def _build_prediction_panel(
    image_np: np.ndarray,
    gt_fields: dict[str, Any],
    pred_fields: dict[str, Any],
    pred_refined: dict[str, Any],
    gt_rows,
    pred_rows,
    pred_seeds,
    raw_trajectories,
    valid_masks: dict[str, Any] | None,
    has_labels: bool,
) -> np.ndarray:
    return make_prediction_summary_panel(
        image_np,
        gt_fields=gt_fields if has_labels else {},
        pred_fields=pred_fields,
        refined_structure=pred_refined.get('structure'),
        gt_rows=gt_rows if has_labels else [],
        pred_rows=pred_rows,
        seeds=pred_seeds,
        raw_trajectories=raw_trajectories,
        final_trajectories=pred_rows,
        valid_masks=valid_masks,
    )


def _decoded_stage(pred_decoded: dict[str, Any]) -> dict[str, Any]:
    return {
        'decoded': pred_decoded,
    }


def _prediction_stage(pred_fields: dict[str, Any], pred_refined: dict[str, Any], pred_decoded: dict[str, Any]) -> dict[str, Any]:
    return {
        'fields': pred_fields,
        'refined': pred_refined,
        'decoded': pred_decoded,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize labels, predictions, decoder states, or summary panels for SFR-Net.')
    parser.add_argument(
        '--mode',
        choices=MODES,
        default='label',
        help='Visualization mode: label / pred / decode / summary / continuity_evidence / uncertainty_evidence / refinement_evidence.',
    )
    parser.add_argument('--checkpoint', default='', help='Optional checkpoint for pred / decode / summary and evidence modes.')
    parser.add_argument('--input', default='', help='Single image path. If omitted, use --split and --index.')
    parser.add_argument('--split', default='', help='Dataset split. Defaults to cfg.dataset.train_split when --input is omitted.')
    parser.add_argument('--index', type=int, default=0, help='Dataset sample index when using --split.')
    parser.add_argument('--output-dir', default='', help='Optional output directory override.')
    return add_config_args(parser)


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.set)
    sample, image_np, stem, from_input = _prepare_sample(cfg, args)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.project.output_dir) / 'visualize'
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = sample['targets']
    gt_fields = {
        'center': targets.get('center'),
        'orientation': targets.get('orientation'),
        'continuity': targets.get('continuity'),
        'uncertainty': targets.get('uncertainty'),
    }
    gt_rows = targets['aux'].get('row_polylines', [])
    gt_decoded = {
        'rows': gt_rows,
        'seeds': [row[0] for row in gt_rows if row],
        'debug': {'candidate_paths': []},
    }

    if args.mode == 'label':
        if from_input:
            raise ValueError('label mode requires dataset-backed labels, so please use --split and --index instead of --input.')
        save_field_visualizations(
            output_dir,
            f'{stem}_label',
            image_np,
            gt_fields,
            refined={'structure': None},
            decoded=gt_decoded,
            valid_masks=targets.get('valid_masks'),
        )
        save_comparison_visualization(
            output_dir / f'{stem}_label_rows.png',
            image_np,
            gt_rows=gt_rows,
            pred_rows=gt_rows,
            seeds=gt_decoded['seeds'],
            title_labels=['gt_rows', 'row_overlay'],
        )
        print({'mode': 'label', 'output_dir': str(output_dir), 'sample': stem})
        return

    outputs, loaded = _run_model(cfg, sample, args.checkpoint)
    pred_fields, pred_refined, pred_decoded = _to_cpu_fields(outputs)
    pred_rows = pred_decoded['rows']
    pred_seeds = pred_decoded['seeds']
    decode_debug = pred_decoded.get('debug', {}) or {}
    raw_trajectories = decode_debug.get('candidate_paths', [])
    has_labels = not from_input

    if args.mode == 'pred':
        save_field_visualizations(
            output_dir,
            f'{stem}_pred',
            image_np,
            pred_fields,
            refined=pred_refined,
            decoded=pred_decoded,
            valid_masks=targets.get('valid_masks'),
        )
        print({'mode': 'pred', 'checkpoint_loaded': loaded, 'output_dir': str(output_dir), 'sample': stem})
        return

    if args.mode == 'decode':
        panel = _build_prediction_panel(
            image_np,
            gt_fields,
            pred_fields,
            pred_refined,
            gt_rows,
            pred_rows,
            pred_seeds,
            raw_trajectories,
            targets.get('valid_masks'),
            has_labels,
        )
        save_image(output_dir / f'{stem}_decode_panel.png', panel)
        print({'mode': 'decode', 'checkpoint_loaded': loaded, 'output_dir': str(output_dir), 'sample': stem})
        return

    if args.mode == 'summary':
        panel = _build_prediction_panel(
            image_np,
            gt_fields,
            pred_fields,
            pred_refined,
            gt_rows,
            pred_rows,
            pred_seeds,
            raw_trajectories,
            targets.get('valid_masks'),
            has_labels,
        )
        save_image(output_dir / f'{stem}_summary_panel.png', panel)
        print({'mode': 'summary', 'checkpoint_loaded': loaded, 'output_dir': str(output_dir), 'sample': stem})
        return

    if args.mode == 'continuity_evidence':
        panel = continuity_evidence_panel(
            image_np,
            gt_fields=gt_fields if has_labels else {},
            pred_fields=pred_fields,
            center_stage=_decoded_stage(pred_decoded),
            continuity_stage=_prediction_stage(pred_fields, pred_refined, pred_decoded),
            columns=3,
        )
        save_image(output_dir / f'{stem}_continuity_evidence_panel.png', panel)
        print({'mode': 'continuity_evidence', 'checkpoint_loaded': loaded, 'output_dir': str(output_dir), 'sample': stem})
        return

    if args.mode == 'uncertainty_evidence':
        panel = uncertainty_evidence_panel(
            image_np,
            pred_fields={**pred_fields, 'structure': pred_refined.get('structure')},
            decoded=pred_decoded,
            columns=3,
        )
        save_image(output_dir / f'{stem}_uncertainty_evidence_panel.png', panel)
        print({'mode': 'uncertainty_evidence', 'checkpoint_loaded': loaded, 'output_dir': str(output_dir), 'sample': stem})
        return

    if args.mode == 'refinement_evidence':
        center_stage = {
            'decoded': {
                **pred_decoded,
                'debug': {**pred_decoded.get('debug', {}), 'candidate_paths': raw_trajectories},
            },
        }
        refined_stage = _prediction_stage(pred_fields, pred_refined, pred_decoded)
        panel = refinement_evidence_panel(
            image_np,
            fields=pred_fields,
            refined=pred_refined,
            center_decoded=center_stage,
            refined_decoded=refined_stage,
            columns=3,
        )
        save_image(output_dir / f'{stem}_refinement_evidence_panel.png', panel)
        print({'mode': 'refinement_evidence', 'checkpoint_loaded': loaded, 'output_dir': str(output_dir), 'sample': stem})
        return

    raise ValueError(f'Unsupported mode: {args.mode}')


if __name__ == '__main__':
    main()
