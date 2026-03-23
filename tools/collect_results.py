from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


SUMMARY_CANDIDATES = (
    'results.json',
    'summary.json',
    'test_summary.json',
    'infer_summary.json',
)
METRICS_CANDIDATES = ('metrics.csv',)
CONFIG_CANDIDATES = ('config_dump.yaml', 'config.yaml', 'config.json')
PRIMARY_METRIC_KEYS = (
    'row_f1',
    'row_precision',
    'row_recall',
    'pixel_f1_center',
    'pixel_f1_continuity',
    'continuity_f1',
    'continuity_iou',
    'uncertainty_mae',
    'avg_centerline_distance',
)
CHECKPOINT_KEYS = (
    'checkpoint',
    'checkpoint_path',
    'best_checkpoint',
    'last_checkpoint',
    'resume',
)


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open('r', encoding='utf-8-sig') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8-sig')
    except OSError:
        return ''


def _convert_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ''
    lowered = text.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    if lowered in {'null', 'none'}:
        return None
    try:
        if any(token in text for token in ('.', 'e', 'E')):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _read_simple_yaml(path: Path) -> dict[str, Any] | None:
    text = _read_text(path)
    if not text:
        return None

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip(' '))
        stripped = line.strip()
        if ':' not in stripped:
            continue
        key, value = stripped.split(':', 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _convert_scalar(value)
    return root


def _read_config(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() == '.json':
        payload = _read_json(path)
        return payload if isinstance(payload, dict) else None
    if path.suffix.lower() in {'.yaml', '.yml'}:
        return _read_simple_yaml(path)
    return None


def _normalize_pattern(pattern: str | None) -> re.Pattern[str] | None:
    if not pattern:
        return None
    return re.compile(pattern)


def _matches_pattern(path: Path, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(path.name) or pattern.search(str(path)))


def _safe_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _infer_mode(config: dict[str, Any] | None, metrics: dict[str, Any]) -> str:
    profile = _safe_get(config, 'model', 'profile')
    if isinstance(profile, str) and profile:
        return profile.lower()

    use_uncertainty = _safe_get(config, 'model', 'use_uncertainty')
    use_reasoning = _safe_get(config, 'model', 'use_reasoning')
    use_continuity = _safe_get(config, 'model', 'use_continuity')
    if use_uncertainty:
        return 'full'
    if use_reasoning or use_continuity:
        return 'core'

    if any(key in metrics for key in ('uncertainty_mae', 'uncertainty_bce_like_score')):
        return 'full'
    if any(key in metrics for key in ('pixel_f1_continuity', 'continuity_f1', 'continuity_iou')):
        return 'core'
    return 'base'


def _pick_first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _extract_metrics_from_json(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float, str, bool)):
            metrics[key] = value
    return metrics


def _extract_metrics_from_csv(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    last_row = rows[-1]
    metrics: dict[str, Any] = {}
    for key, value in last_row.items():
        if value is None:
            continue
        converted = _convert_scalar(value)
        if isinstance(converted, (int, float, str, bool)) or converted is None:
            metrics[key] = converted
    return metrics


def _extract_checkpoint_info(config: dict[str, Any] | None, metrics: dict[str, Any], run_dir: Path) -> str:
    for key in CHECKPOINT_KEYS:
        value = metrics.get(key)
        if isinstance(value, str) and value:
            return value
    train_resume = _safe_get(config, 'train', 'resume')
    if isinstance(train_resume, str) and train_resume:
        return train_resume
    for candidate in ('best.pt', 'last.pt'):
        checkpoint_path = run_dir / candidate
        if checkpoint_path.exists():
            return str(checkpoint_path)
    return ''


def _build_row(run_dir: Path, metrics: dict[str, Any], config: dict[str, Any] | None, files: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        'experiment_name': run_dir.name,
        'experiment_dir': str(run_dir),
        'mode': _infer_mode(config, metrics),
        'checkpoint_path': _extract_checkpoint_info(config, metrics, run_dir),
        'results_file': files.get('results_file', ''),
        'metrics_file': files.get('metrics_file', ''),
        'config_file': files.get('config_file', ''),
    }

    for key in PRIMARY_METRIC_KEYS:
        row[key] = metrics.get(key, '')

    row['profile'] = _safe_get(config, 'model', 'profile') or ''
    row['backbone'] = _safe_get(config, 'model', 'backbone') or ''
    row['output_dir'] = _safe_get(config, 'project', 'output_dir') or ''
    return row


def _find_runs(root: Path, pattern: re.Pattern[str] | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.rglob('*') if path.is_dir()):
        if not _matches_pattern(directory, pattern):
            continue

        summary_path = _pick_first_existing(directory, SUMMARY_CANDIDATES)
        metrics_path = _pick_first_existing(directory, METRICS_CANDIDATES)
        config_path = _pick_first_existing(directory, CONFIG_CANDIDATES)
        if summary_path is None and metrics_path is None:
            continue

        summary_payload = _read_json(summary_path) if summary_path is not None else None
        csv_rows = _read_csv_rows(metrics_path) if metrics_path is not None else []
        config_payload = _read_config(config_path) if config_path is not None else None

        metrics: dict[str, Any] = {}
        metrics.update(_extract_metrics_from_json(summary_payload if isinstance(summary_payload, dict) else None))
        metrics.update({key: value for key, value in _extract_metrics_from_csv(csv_rows).items() if key not in metrics or metrics[key] in ('', None)})

        files = {
            'results_file': str(summary_path) if summary_path is not None else '',
            'metrics_file': str(metrics_path) if metrics_path is not None else '',
            'config_file': str(config_path) if config_path is not None else '',
        }
        row = _build_row(directory, metrics, config_payload, files)
        runs.append(
            {
                'experiment_name': directory.name,
                'experiment_dir': str(directory),
                'files': files,
                'metrics': metrics,
                'config': config_payload,
                'row': row,
            }
        )
    return runs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open('w', encoding='utf-8', newline='') as handle:
            handle.write('')
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Collect SFR-Net experiment results into one summary table.')
    parser.add_argument('--root_dir', '--root', dest='root_dir', default='outputs', help='Root directory to scan recursively.')
    parser.add_argument('--pattern', default='', help='Optional regex pattern used to filter experiment directories.')
    parser.add_argument('--output_file', '--output', dest='output_file', default='outputs/collected_results.json', help='Output file path. Supports .json and .csv.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f'Root directory not found: {root_dir}')

    pattern = _normalize_pattern(args.pattern)
    runs = _find_runs(root_dir, pattern)
    flat_rows = [run['row'] for run in runs]
    payload = {
        'root_dir': str(root_dir),
        'pattern': args.pattern,
        'num_experiments': len(runs),
        'experiments': runs,
        'table': flat_rows,
    }

    output_path = Path(args.output_file)
    suffix = output_path.suffix.lower()
    if suffix == '.csv':
        _write_csv(output_path, flat_rows)
    else:
        _write_json(output_path, payload)

    print({'root_dir': str(root_dir), 'num_experiments': len(runs), 'output_file': str(output_path)})


if __name__ == '__main__':
    main()
