from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from sfr_net.engine.checkpoint import load_checkpoint
    from sfr_net.engine.inferencer import Inferencer
    from sfr_net.models.sfr_net import SFRNet
    from sfr_net.utils.config import add_config_args, load_config
else:
    from ..engine.checkpoint import load_checkpoint
    from ..engine.inferencer import Inferencer
    from ..models.sfr_net import SFRNet
    from ..utils.config import add_config_args, load_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run SFR-Net inference for one image or a whole directory.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path used for inference.')
    parser.add_argument('--input', required=True, help='Single image path or a directory containing images.')
    parser.add_argument('--output-dir', default='', help='Optional output directory override.')
    parser.add_argument('--limit', type=int, default=0, help='Optional image limit when --input is a directory.')
    parser.add_argument('--no-save-fields', action='store_true', help='Disable field-map visualization export.')
    return add_config_args(parser)


def _load_model_weights(model: SFRNet, checkpoint_path: str) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
    model.load_state_dict(state_dict, strict=False)
    return payload


def _run_single(inferencer: Inferencer, input_path: Path, output_dir: Path, save_fields: bool) -> None:
    result = inferencer.infer_one(input_path, output_dir=output_dir, save_fields=save_fields)
    print(
        {
            'mode': 'single',
            'image_path': result['image_path'],
            'output_dir': result['output_dir'],
            'decoded_rows': len(result['payload'].get('rows', [])),
            'field_stats': result['payload'].get('field_stats', {}),
        }
    )


def _run_directory(inferencer: Inferencer, input_path: Path, output_dir: Path, limit: int, save_fields: bool) -> None:
    results = inferencer.infer_folder(
        input_path,
        output_dir=output_dir,
        limit=limit or None,
        save_fields=save_fields,
    )
    print(
        {
            'mode': 'directory',
            'input_dir': str(input_path),
            'num_images': len(results),
            'output_dir': str(output_dir),
        }
    )


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.set)
    model = SFRNet(cfg)
    _load_model_weights(model, args.checkpoint)
    inferencer = Inferencer(cfg, model, model.decoder)

    save_fields = not args.no_save_fields
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.project.output_dir) / 'infer'
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input path not found: {input_path}')

    if input_path.is_dir():
        _run_directory(inferencer, input_path, output_dir, args.limit, save_fields)
        return
    _run_single(inferencer, input_path, output_dir, save_fields)


if __name__ == '__main__':
    main()
