from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..datasets.io import VALID_EXTENSIONS, load_image, save_json
from ..utils.visualization import save_comparison_visualization, save_field_visualizations, save_prediction_visualization


class Inferencer:
    def __init__(self, cfg, model, decoder, device: str | None = None):
        self.cfg = cfg
        self.model = model
        self.decoder = decoder
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()

    def _cfg_flag(self, path: str, default: bool) -> bool:
        current: Any = self.cfg
        for name in path.split('.'):
            if current is None or not hasattr(current, name):
                return bool(default)
            current = getattr(current, name)
        return bool(current)

    def _prepare_image_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        image_np = image_np.copy()
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if self.cfg.dataset.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        return image_tensor.to(self.device)

    def _to_cpu(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {key: self._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_cpu(item) for item in value]
        if hasattr(value, 'detach'):
            return value.detach().cpu()
        return value

    def _sample_output(self, outputs: dict[str, Any], batch_idx: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        fields = {}
        for key, value in outputs.get('fields', {}).items():
            fields[key] = None if value is None else value[batch_idx]
        refined = outputs.get('refined') or {}
        refined_fields = {
            'structure': None if refined.get('structure') is None else refined['structure'][batch_idx],
            'fields': None if refined.get('fields') is None else refined['fields'],
        }
        decoded = outputs.get('decoded') or {}
        sample_decoded = {
            'rows': decoded.get('rows', [[]])[batch_idx] if decoded.get('rows') is not None else [],
            'scores': decoded.get('scores', [[]])[batch_idx] if decoded.get('scores') is not None else [],
            'seeds': decoded.get('seeds', [[]])[batch_idx] if decoded.get('seeds') is not None else [],
            'debug': {},
        }
        if decoded.get('debug') is not None:
            sample_decoded['debug'] = {
                'candidate_paths': decoded['debug'].get('candidate_paths', [[]])[batch_idx] if decoded['debug'].get('candidate_paths') is not None else [],
                'propagation': decoded['debug'].get('propagation', [[]])[batch_idx] if decoded['debug'].get('propagation') is not None else [],
                'structure_source': decoded['debug'].get('structure_source', 'center'),
                'seed_source': decoded['debug'].get('seed_source', decoded['debug'].get('structure_source', 'center')),
                'continuity_enabled': decoded['debug'].get('continuity_enabled', False),
                'uncertainty_enabled': decoded['debug'].get('uncertainty_enabled', False),
                'use_continuity': decoded['debug'].get('use_continuity', False),
                'use_uncertainty': decoded['debug'].get('use_uncertainty', False),
            }
        return fields, refined_fields, sample_decoded

    def _field_stats(self, fields: dict[str, Any], refined: dict[str, Any]) -> dict[str, float]:
        return {
            'center_max': float(fields['center'].max().item()) if fields.get('center') is not None else 0.0,
            'continuity_max': float(fields['continuity'].max().item()) if fields.get('continuity') is not None else 0.0,
            'uncertainty_max': float(fields['uncertainty'].max().item()) if fields.get('uncertainty') is not None else 0.0,
            'structure_max': float(refined['structure'].max().item()) if refined.get('structure') is not None else 0.0,
        }

    def _serialize_rows(self, decoded: dict[str, Any]) -> list[dict[str, Any]]:
        serialized = []
        for index, row in enumerate(decoded.get('rows', [])):
            if isinstance(row, dict):
                points = row.get('points', [])
                score = float(row.get('score', 0.0))
                length = float(row.get('length', len(points)))
                item = dict(row)
                item['points'] = points
                item['score'] = score
                item['length'] = length
            else:
                points = row
                item = {
                    'points': points,
                    'score': float(decoded.get('scores', [0.0] * len(decoded.get('rows', [])))[index]) if index < len(decoded.get('scores', [])) else 0.0,
                    'length': float(len(points)),
                }
            serialized.append(item)
        return serialized

    def _build_export_payload(self, image_path: Path, meta: dict[str, Any], fields: dict[str, Any], refined: dict[str, Any], decoded: dict[str, Any]) -> dict[str, Any]:
        rows = self._serialize_rows(decoded)
        return {
            'image_path': str(image_path),
            'image_id': meta.get('image_id', image_path.stem),
            'rows': rows,
            'scores': decoded.get('scores', []),
            'seeds': decoded.get('seeds', []),
            'decoder_debug': decoded.get('debug', {}),
            'field_stats': self._field_stats(fields, refined),
        }

    def _save_field_arrays(self, sample_dir: Path, stem: str, fields: dict[str, Any], refined: dict[str, Any]) -> None:
        if not self._cfg_flag('infer.save_field_arrays', False):
            return
        arrays: dict[str, np.ndarray] = {}
        for key in ('center', 'continuity', 'uncertainty', 'orientation'):
            value = fields.get(key)
            if value is not None:
                arrays[key] = value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
        if refined.get('structure') is not None:
            arrays['refined_structure'] = refined['structure'].numpy() if hasattr(refined['structure'], 'numpy') else np.asarray(refined['structure'])
        for name, array in arrays.items():
            np.save(sample_dir / f'{stem}_{name}.npy', array)

    def _export_sample(
        self,
        image_np: np.ndarray,
        image_path: Path,
        meta: dict[str, Any],
        fields: dict[str, Any],
        refined: dict[str, Any],
        decoded: dict[str, Any],
        output_dir: str | Path | None = None,
        save_fields: bool = True,
    ) -> dict[str, Any]:
        export_root = Path(output_dir) if output_dir is not None else Path(self.cfg.project.output_dir) / 'infer'
        export_root.mkdir(parents=True, exist_ok=True)
        stem = Path(meta.get('image_id', image_path.stem)).stem or image_path.stem
        sample_dir = export_root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        payload = self._build_export_payload(image_path, meta, fields, refined, decoded)
        if self._cfg_flag('infer.save_rows', True):
            save_json(sample_dir / f'{stem}_rows.json', payload)

        overlay_rows = decoded.get('rows', [])
        overlay_seeds = decoded.get('seeds', [])
        if self._cfg_flag('infer.save_vis', True):
            save_prediction_visualization(
                sample_dir / f'{stem}_overlay.png',
                image_np,
                fields.get('center'),
                overlay_rows,
                overlay_seeds,
                continuity_map=fields.get('continuity'),
                structure_map=refined.get('structure'),
                uncertainty_map=fields.get('uncertainty'),
            )
            save_comparison_visualization(
                sample_dir / f'{stem}_decoded_rows.png',
                image_np,
                gt_rows=[],
                pred_rows=overlay_rows,
                seeds=overlay_seeds,
                title_labels=['input', 'decoded'],
            )
        if save_fields and self._cfg_flag('infer.save_fields', True):
            save_field_visualizations(sample_dir, stem, image_np, fields, refined=refined, decoded=decoded)
        self._save_field_arrays(sample_dir, stem, fields, refined)
        return {
            'image_path': str(image_path),
            'output_dir': str(sample_dir),
            'payload': payload,
        }

    def infer_batch(self, batch):
        images = batch['image'].to(self.device)
        targets = batch.get('targets')
        with torch.no_grad():
            outputs = self.model(images, targets, mode='infer')
            decoded = self.decoder.decode(outputs['fields'], refined=outputs['refined'], meta=batch.get('meta'))
            outputs['decoded'] = decoded
        outputs = self._to_cpu(outputs)

        meta_list = batch.get('meta') or []
        exported = []
        for batch_idx, meta in enumerate(meta_list):
            fields, refined, sample_decoded = self._sample_output(outputs, batch_idx)
            image_tensor = batch['image'][batch_idx].detach().cpu()
            image_np = image_tensor.permute(1, 2, 0).numpy()
            if self.cfg.dataset.normalize:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                image_np = image_np * std + mean
            image_np = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
            image_path = Path(meta.get('path', meta.get('image_id', f'batch_{batch_idx:04d}.png')))
            exported.append(
                self._export_sample(
                    image_np,
                    image_path,
                    meta,
                    fields,
                    refined,
                    sample_decoded,
                    output_dir=Path(self.cfg.project.output_dir) / 'infer_batch',
                    save_fields=self._cfg_flag('infer.save_fields', True),
                )
            )
        outputs['exports'] = exported
        return outputs

    def infer_one(self, image_path, output_dir: str | Path | None = None, save_fields: bool = True):
        image_path = Path(image_path)
        image_np = load_image(image_path)
        image_tensor = self._prepare_image_tensor(image_np)
        with torch.no_grad():
            outputs = self.model(image_tensor, mode='infer')
            decoded = self.decoder.decode(outputs['fields'], refined=outputs['refined'], meta=[{'path': str(image_path), 'image_id': image_path.stem}])
            outputs['decoded'] = decoded
        outputs = self._to_cpu(outputs)
        fields, refined, sample_decoded = self._sample_output(outputs, 0)
        export = self._export_sample(
            image_np,
            image_path,
            {'path': str(image_path), 'image_id': image_path.stem},
            fields,
            refined,
            sample_decoded,
            output_dir=output_dir,
            save_fields=save_fields,
        )
        return {
            'image_path': str(image_path),
            'output_dir': export['output_dir'],
            'payload': export['payload'],
            'outputs': outputs,
        }

    def infer_folder(self, input_dir, output_dir: str | Path | None = None, limit: int | None = None, save_fields: bool = True):
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f'Input directory not found: {input_dir}')
        image_paths = [path for path in sorted(input_dir.rglob('*')) if path.suffix.lower() in VALID_EXTENSIONS]
        if limit is not None:
            image_paths = image_paths[:limit]
        export_root = Path(output_dir) if output_dir is not None else Path(self.cfg.project.output_dir) / 'infer'
        export_root.mkdir(parents=True, exist_ok=True)
        results = []
        for image_path in image_paths:
            results.append(self.infer_one(image_path, output_dir=export_root, save_fields=save_fields))
        save_json(export_root / 'infer_summary.json', [{'image_path': item['image_path'], 'output_dir': item['output_dir']} for item in results])
        return results
