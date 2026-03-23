from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PRIMARY_METRICS = (
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
STRUCTURE_COLUMNS = (
    'experiment_name',
    'mode',
    'use_continuity',
    'use_uncertainty',
    'use_reasoning',
    'row_f1',
    'pixel_f1_center',
    'pixel_f1_continuity',
    'uncertainty_mae',
)
LOSS_COLUMNS = (
    'experiment_name',
    'mode',
    'lambda_center',
    'lambda_orientation',
    'lambda_continuity',
    'lambda_structure',
    'lambda_uncertainty',
    'row_f1',
    'pixel_f1_center',
    'uncertainty_mae',
)
PARAMETER_CANDIDATES = (
    'reasoning_num_iters',
    'step_size',
    'candidate_radius',
    'stop_continuity_threshold',
    'stop_uncertainty_threshold',
    'center_sigma',
    'orientation_band_width',
)


def _read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return [{key: _convert_scalar(value) for key, value in row.items()} for row in reader]


def _convert_scalar(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
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


def _safe_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _infer_mode(row: dict[str, Any], config: dict[str, Any] | None) -> str:
    mode = row.get('mode')
    if isinstance(mode, str) and mode:
        return mode.lower()
    profile = _safe_get(config, 'model', 'profile')
    if isinstance(profile, str) and profile:
        return profile.lower()
    if _safe_get(config, 'model', 'use_uncertainty'):
        return 'full'
    if _safe_get(config, 'model', 'use_reasoning') or _safe_get(config, 'model', 'use_continuity'):
        return 'core'
    return 'base'


def _round_if_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'Y' if value else 'N'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _build_row_from_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    base_row = dict(experiment.get('row') or {})
    config = experiment.get('config') if isinstance(experiment.get('config'), dict) else {}

    row = dict(base_row)
    row['experiment_name'] = row.get('experiment_name') or experiment.get('experiment_name', '')
    row['experiment_dir'] = row.get('experiment_dir') or experiment.get('experiment_dir', '')
    row['mode'] = _infer_mode(row, config)

    row['use_continuity'] = bool(_safe_get(config, 'model', 'use_continuity'))
    row['use_uncertainty'] = bool(_safe_get(config, 'model', 'use_uncertainty'))
    row['use_reasoning'] = bool(_safe_get(config, 'model', 'use_reasoning'))

    row['lambda_center'] = _safe_get(config, 'loss', 'lambda_center')
    row['lambda_orientation'] = _safe_get(config, 'loss', 'lambda_orientation')
    row['lambda_continuity'] = _safe_get(config, 'loss', 'lambda_continuity')
    row['lambda_structure'] = _safe_get(config, 'loss', 'lambda_structure')
    row['lambda_uncertainty'] = _safe_get(config, 'loss', 'lambda_uncertainty')

    row['reasoning_num_iters'] = _safe_get(config, 'model', 'reasoning_num_iters')
    row['step_size'] = _safe_get(config, 'infer', 'step_size')
    row['candidate_radius'] = _safe_get(config, 'infer', 'candidate_radius')
    row['stop_continuity_threshold'] = _safe_get(config, 'infer', 'stop_continuity_threshold')
    row['stop_uncertainty_threshold'] = _safe_get(config, 'infer', 'stop_uncertainty_threshold')
    row['center_sigma'] = _safe_get(config, 'dataset', 'labels', 'center_sigma')
    row['orientation_band_width'] = _safe_get(config, 'dataset', 'labels', 'orientation_band_width')

    for key in list(row.keys()):
        row[key] = _round_if_float(row[key])
    return row


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == '.csv':
        rows = _read_csv(path)
        for row in rows:
            row.setdefault('mode', str(row.get('mode', '')).lower())
        return rows

    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get('experiments'), list):
        return [_build_row_from_experiment(experiment) for experiment in payload['experiments']]
    if isinstance(payload, dict) and isinstance(payload.get('table'), list):
        return [{key: _round_if_float(value) for key, value in row.items()} for row in payload['table']]
    if isinstance(payload, list):
        return [{key: _round_if_float(value) for key, value in row.items()} for row in payload if isinstance(row, dict)]
    raise ValueError(f'Unsupported input file structure: {path}')


def _available_columns(rows: list[dict[str, Any]], preferred: tuple[str, ...]) -> list[str]:
    available = []
    for column in preferred:
        if any(column in row and row[column] not in ('', None) for row in rows):
            available.append(column)
    return available


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get('mode', '')), -float(row.get('row_f1', 0.0) or 0.0), str(row.get('experiment_name', ''))))


def _group_rows(rows: list[dict[str, Any]], group_by: str | None) -> dict[str, list[dict[str, Any]]]:
    if not group_by:
        return {'all': rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_by, 'unknown') or 'unknown')
        grouped.setdefault(key, []).append(row)
    return grouped


def _make_markdown_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [f'## {title}', '']
    if not rows or not columns:
        lines.append('No matching rows.')
        return '\n'.join(lines)
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(_format_value(row.get(column, '')) for column in columns) + ' |')
    return '\n'.join(lines)


def _escape_latex(value: Any) -> str:
    text = _format_value(value)
    replacements = {
        '\\': '\\textbackslash{}',
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _make_latex_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [f'% {title}']
    if not rows or not columns:
        lines.append('% No matching rows.')
        return '\n'.join(lines)
    alignment = 'l' + 'c' * max(len(columns) - 1, 0)
    lines.extend([
        f'\\begin{{tabular}}{{{alignment}}}',
        '\\hline',
        ' & '.join(_escape_latex(column) for column in columns) + ' \\\\',
        '\\hline',
    ])
    for row in rows:
        lines.append(' & '.join(_escape_latex(row.get(column, '')) for column in columns) + ' \\\\')
    lines.extend(['\\hline', '\\end{tabular}'])
    return '\n'.join(lines)


def _parameter_columns(rows: list[dict[str, Any]]) -> list[str]:
    dynamic = _available_columns(rows, PARAMETER_CANDIDATES)
    metric_columns = _available_columns(rows, ('row_f1', 'pixel_f1_center', 'uncertainty_mae'))
    columns = ['experiment_name', 'mode'] + dynamic + metric_columns
    deduped: list[str] = []
    for column in columns:
        if column not in deduped:
            deduped.append(column)
    return deduped


def _build_tables(rows: list[dict[str, Any]], group_by: str | None) -> dict[str, dict[str, str]]:
    sorted_rows = _sort_rows(rows)
    grouped = _group_rows(sorted_rows, group_by)
    outputs: dict[str, dict[str, str]] = {}

    structure_columns = _available_columns(sorted_rows, STRUCTURE_COLUMNS)
    loss_columns = _available_columns(sorted_rows, LOSS_COLUMNS)
    parameter_columns = _parameter_columns(sorted_rows)

    sections = {
        'structure_ablation': structure_columns,
        'loss_ablation': loss_columns,
        'parameter_sensitivity': parameter_columns,
    }

    for section_name, columns in sections.items():
        markdown_parts: list[str] = []
        latex_parts: list[str] = []
        pretty_title = section_name.replace('_', ' ').title()
        for group_name, group_rows in grouped.items():
            title = pretty_title if group_name == 'all' else f'{pretty_title} ({group_by}={group_name})'
            markdown_parts.append(_make_markdown_table(title, group_rows, columns))
            latex_parts.append(_make_latex_table(title, group_rows, columns))
        outputs[section_name] = {
            'markdown': '\n\n'.join(markdown_parts).strip() + '\n',
            'latex': '\n\n'.join(latex_parts).strip() + '\n',
        }
    return outputs


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate SFR-Net ablation tables from collected results.')
    parser.add_argument('--input_file', default='outputs/collected_results.json', help='Input results file from collect_results.py (.json or .csv).')
    parser.add_argument('--output_dir', default='outputs/ablation_tables', help='Directory used to save generated tables.')
    parser.add_argument('--format', choices=['markdown', 'latex', 'both'], default='both', help='Output format.')
    parser.add_argument('--group_by', default='', help='Optional column used to split rows into grouped sub-tables.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    rows = _load_rows(input_path)
    tables = _build_tables(rows, args.group_by or None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[str] = []
    for section_name, contents in tables.items():
        if args.format in {'markdown', 'both'}:
            markdown_path = output_dir / f'{section_name}.md'
            _write_text(markdown_path, contents['markdown'])
            written_files.append(str(markdown_path))
        if args.format in {'latex', 'both'}:
            latex_path = output_dir / f'{section_name}.tex'
            _write_text(latex_path, contents['latex'])
            written_files.append(str(latex_path))

    print({'input_file': str(input_path), 'num_rows': len(rows), 'output_dir': str(output_dir), 'files': written_files})


if __name__ == '__main__':
    main()
