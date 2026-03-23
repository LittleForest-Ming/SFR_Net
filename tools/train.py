from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from sfr_net.datasets.collate import crop_row_collate_fn
    from sfr_net.datasets.crop_row_dataset import CropRowDataset
    from sfr_net.engine.trainer import Trainer
    from sfr_net.losses import SFRCriterion
    from sfr_net.models.sfr_net import SFRNet
    from sfr_net.utils.config import add_config_args, load_config
    from sfr_net.utils.seed import seed_everything
else:
    from ..datasets.collate import crop_row_collate_fn
    from ..datasets.crop_row_dataset import CropRowDataset
    from ..engine.trainer import Trainer
    from ..losses import SFRCriterion
    from ..models.sfr_net import SFRNet
    from ..utils.config import add_config_args, load_config
    from ..utils.seed import seed_everything


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train SFR-Net with Base/Core/Full unified configs.')
    parser.add_argument('--resume', default='', help='Optional checkpoint path to resume training from.')
    return add_config_args(parser)


def _build_dataloader(cfg, split: str, shuffle: bool) -> DataLoader:
    dataset = CropRowDataset(cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.project.num_workers),
        collate_fn=crop_row_collate_fn,
    )


def _dump_config(cfg, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(_to_builtin(cfg), ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.set)
    if args.resume:
        cfg.train.resume = args.resume

    seed_everything(int(cfg.project.seed))
    output_dir = Path(cfg.project.output_dir)
    _dump_config(cfg, output_dir)

    train_loader = _build_dataloader(cfg, cfg.dataset.train_split, shuffle=True)
    val_loader = _build_dataloader(cfg, cfg.dataset.val_split, shuffle=False)

    model = SFRNet(cfg)
    criterion = SFRCriterion(cfg)
    trainer = Trainer(cfg, model, criterion, train_loader, val_loader)
    trainer.fit()


if __name__ == '__main__':
    main()
