from __future__ import annotations

from pathlib import Path
from typing import Any

from ..datasets.io import save_json
from ..metrics import compute_continuity_metrics, compute_uncertainty_metrics, pixel_metrics, row_metrics


class SFREvaluator:
    def __init__(self, cfg, export_dir: str | Path | None = None):
        self.cfg = cfg
        self.export_dir = Path(export_dir) if export_dir is not None else None
        self.reset()

    def reset(self) -> None:
        self.pixel_center_records: list[dict[str, float]] = []
        self.continuity_records: list[dict[str, float]] = []
        self.uncertainty_records: list[dict[str, float]] = []
        self.row_records: list[dict[str, float]] = []
        self.sample_results: list[dict[str, Any]] = []

    def _to_cpu(self, value):
        if value is None:
            return None
        return value.detach().cpu() if hasattr(value, 'detach') else value

    def _update_center_metrics(self, outputs, targets) -> None:
        center_metrics = pixel_metrics(
            self._to_cpu(outputs['fields'].get('center')),
            self._to_cpu(targets.get('center')),
        )
        self.pixel_center_records.append(center_metrics)

    def _update_continuity_metrics(self, outputs, targets) -> None:
        continuity_pred = outputs['fields'].get('continuity')
        continuity_target = targets.get('continuity')
        continuity_mask = targets['valid_masks'].get('continuity')
        continuity_metrics = compute_continuity_metrics(
            self._to_cpu(continuity_pred),
            self._to_cpu(continuity_target),
            self._to_cpu(continuity_mask),
        )
        self.continuity_records.append(continuity_metrics)

    def _update_uncertainty_metrics(self, outputs, targets) -> None:
        uncertainty_pred = outputs['fields'].get('uncertainty')
        uncertainty_target = targets.get('uncertainty')
        uncertainty_mask = targets['valid_masks'].get('uncertainty')
        uncertainty_metrics = compute_uncertainty_metrics(
            self._to_cpu(uncertainty_pred),
            self._to_cpu(uncertainty_target),
            self._to_cpu(uncertainty_mask),
        )
        self.uncertainty_records.append(uncertainty_metrics)

    def _update_row_metrics(self, outputs, targets, meta) -> None:
        decoded_rows = outputs.get('decoded', {}).get('rows') or [[] for _ in meta]
        gt_rows = targets['aux'].get('row_polylines', [[] for _ in meta])
        for sample_meta, pred, gt in zip(meta, decoded_rows, gt_rows):
            row_record = row_metrics(pred, gt)
            self.row_records.append(row_record)
            self.sample_results.append({
                'image_id': sample_meta.get('image_id', ''),
                'path': sample_meta.get('path', ''),
                **row_record,
            })

    def update(self, outputs, targets, meta) -> None:
        self._update_center_metrics(outputs, targets)
        self._update_continuity_metrics(outputs, targets)
        self._update_uncertainty_metrics(outputs, targets)
        self._update_row_metrics(outputs, targets, meta)

    def summarize(self) -> dict[str, float]:
        def average(records: list[dict[str, float]], key: str) -> float:
            if not records:
                return 0.0
            return sum(float(record.get(key, 0.0)) for record in records) / len(records)

        summary = {
            'pixel_f1_center': average(self.pixel_center_records, 'center_f1'),
            'pixel_f1_continuity': average(self.continuity_records, 'continuity_f1'),
            'continuity_iou': average(self.continuity_records, 'continuity_iou'),
            'continuity_mae': average(self.continuity_records, 'continuity_mae'),
            'uncertainty_mae': average(self.uncertainty_records, 'uncertainty_mae'),
            'uncertainty_bce_like_score': average(self.uncertainty_records, 'uncertainty_bce_like_score'),
            'uncertainty_region_precision': average(self.uncertainty_records, 'uncertainty_region_precision'),
            'row_precision': average(self.row_records, 'row_precision'),
            'row_recall': average(self.row_records, 'row_recall'),
            'row_f1': average(self.row_records, 'row_f1'),
            'avg_centerline_distance': average(self.row_records, 'avg_centerline_distance'),
            'false_split': average(self.row_records, 'false_split'),
            'false_merge': average(self.row_records, 'false_merge'),
            'decoded_row_count': average(self.row_records, 'predicted_row_count'),
            'gt_row_count': average(self.row_records, 'gt_row_count'),
        }
        if self.export_dir is not None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            save_json(self.export_dir / 'summary.json', summary)
            save_json(self.export_dir / 'samples.json', self.sample_results)
        return summary
