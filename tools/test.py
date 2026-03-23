from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from sfr_net.datasets.collate import crop_row_collate_fn
    from sfr_net.datasets.crop_row_dataset import CropRowDataset
    from sfr_net.engine.checkpoint import load_checkpoint
    from sfr_net.engine.evaluator import SFREvaluator
    from sfr_net.models.sfr_net import SFRNet
    from sfr_net.utils.config import add_config_args, load_config
    from sfr_net.utils.seed import seed_everything
else:
    from ..datasets.collate import crop_row_collate_fn
    from ..datasets.crop_row_dataset import CropRowDataset
    from ..engine.checkpoint import load_checkpoint
    from ..engine.evaluator import SFREvaluator
    from ..models.sfr_net import SFRNet
    from ..utils.config import add_config_args, load_config
    from ..utils.seed import seed_everything


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate SFR-Net checkpoints for Base/Core/Full configs.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path to evaluate.')
    parser.add_argument('--output-dir', default='', help='Optional directory for evaluation exports.')
    return add_config_args(parser)


def _build_dataloader(cfg) -> DataLoader:
    dataset = CropRowDataset(cfg, cfg.dataset.test_split)
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.project.num_workers),
        collate_fn=crop_row_collate_fn,
    )


def _resolve_device(cfg) -> torch.device:
    use_cuda = torch.cuda.is_available() and getattr(cfg.project, 'device', 'cpu') == 'cuda'
    return torch.device('cuda' if use_cuda else 'cpu')


def _load_model_weights(model: SFRNet, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
    model.load_state_dict(state_dict, strict=False)
    return payload


def _write_samples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(int(cfg.project.seed))

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.project.output_dir) / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_dataloader(cfg)
    device = _resolve_device(cfg)
    model = SFRNet(cfg)
    _load_model_weights(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    evaluator = SFREvaluator(cfg, export_dir=output_dir)
    with torch.no_grad():
        for batch in loader:
            batch['image'] = batch['image'].to(device)
            for key in ('center', 'orientation', 'continuity', 'uncertainty'):
                if batch['targets'][key] is not None:
                    batch['targets'][key] = batch['targets'][key].to(device)
                mask = batch['targets']['valid_masks'][key]
                if mask is not None:
                    batch['targets']['valid_masks'][key] = mask.to(device)
            for key, value in list(batch['targets']['aux'].items()):
                if torch.is_tensor(value):
                    batch['targets']['aux'][key] = value.to(device)

            outputs = model(batch['image'], batch['targets'], mode='infer')
            outputs['decoded'] = model.decoder.decode(outputs['fields'], refined=outputs['refined'], meta=batch['meta'])
            evaluator.update(outputs, batch['targets'], batch['meta'])

    summary = evaluator.summarize()
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    samples_json_path = output_dir / 'samples.json'
    samples_csv_path = output_dir / 'samples.csv'
    samples_json_path.write_text(json.dumps(evaluator.sample_results, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_samples_csv(samples_csv_path, evaluator.sample_results)
    print(summary)


if __name__ == '__main__':
    main()
