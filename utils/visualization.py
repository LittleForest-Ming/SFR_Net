from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_BG_COLOR = (18, 18, 18)
DEFAULT_LABEL_COLOR = (255, 255, 255)


def _extract_points(row: Any) -> list[list[float]]:
    if isinstance(row, dict):
        return row.get('points', [])
    return row


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_image(image_np: Any) -> np.ndarray:
    image = _to_numpy(image_np)
    if image is None:
        raise ValueError('image_np must not be None.')
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3:
        raise ValueError('image_np must have shape [H, W, C] or [C, H, W].')
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _extract_single_channel(field_map: Any) -> np.ndarray | None:
    field = _to_numpy(field_map)
    if field is None:
        return None
    while field.ndim > 2:
        field = field[0]
    return np.asarray(field, dtype=np.float32)


def _to_heatmap(field_map: Any, mode: str = 'red') -> np.ndarray:
    field = _extract_single_channel(field_map)
    if field is None:
        raise ValueError('field_map must not be None.')
    field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    field = np.clip(field, 0.0, 1.0)
    heat = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.uint8)
    if mode == 'green':
        heat[..., 1] = (field * 255).astype(np.uint8)
        heat[..., 0] = (field * 64).astype(np.uint8)
    elif mode == 'cyan':
        heat[..., 1] = (field * 255).astype(np.uint8)
        heat[..., 2] = (field * 255).astype(np.uint8)
    elif mode == 'blue':
        heat[..., 2] = (field * 255).astype(np.uint8)
        heat[..., 1] = (field * 80).astype(np.uint8)
    elif mode == 'magenta':
        heat[..., 0] = (field * 255).astype(np.uint8)
        heat[..., 2] = (field * 255).astype(np.uint8)
    elif mode == 'orange':
        heat[..., 0] = (field * 255).astype(np.uint8)
        heat[..., 1] = (field * 160).astype(np.uint8)
    else:
        heat[..., 0] = (field * 255).astype(np.uint8)
        heat[..., 1] = (field * 64).astype(np.uint8)
    return heat


def draw_rows(
    image_np: Any,
    rows,
    seeds=None,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
    seed_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    image = Image.fromarray(_normalize_image(image_np)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for row in rows or []:
        points = [tuple(map(float, point)) for point in _extract_points(row)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
    for seed in seeds or []:
        x, y = seed
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=seed_color)
    return np.asarray(image)


def draw_seeds(
    image_np: Any,
    seeds,
    color: tuple[int, int, int] = (255, 64, 64),
    radius: int = 3,
) -> np.ndarray:
    image = Image.fromarray(_normalize_image(image_np)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for seed in seeds or []:
        x, y = seed
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return np.asarray(image)


def draw_trajectories(
    image_np: Any,
    trajectories,
    color: tuple[int, int, int] = (255, 128, 0),
    width: int = 2,
    seeds=None,
    seed_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    return draw_rows(image_np, trajectories, seeds=seeds, color=color, width=width, seed_color=seed_color)


def overlay_heatmap(image_np: Any, field_map: Any, mode: str = 'red', alpha: float = 0.3) -> np.ndarray:
    base = _normalize_image(image_np).astype(np.float32)
    heat = _to_heatmap(field_map, mode=mode).astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * heat
    return blended.clip(0, 255).astype(np.uint8)


def overlay_center_heatmap(image_np: Any, center_map: Any) -> np.ndarray:
    return overlay_heatmap(image_np, center_map, mode='red', alpha=0.32)


def overlay_continuity_heatmap(image_np: Any, continuity_map: Any) -> np.ndarray:
    return overlay_heatmap(image_np, continuity_map, mode='green', alpha=0.30)


def overlay_structure_heatmap(image_np: Any, structure_map: Any) -> np.ndarray:
    return overlay_heatmap(image_np, structure_map, mode='cyan', alpha=0.30)


def overlay_uncertainty_heatmap(image_np: Any, uncertainty_map: Any) -> np.ndarray:
    return overlay_heatmap(image_np, uncertainty_map, mode='magenta', alpha=0.30)


def overlay_field_bundle(
    image_np: Any,
    center_map: Any = None,
    continuity_map: Any = None,
    structure_map: Any = None,
    uncertainty_map: Any = None,
) -> np.ndarray:
    canvas = _normalize_image(image_np)
    if center_map is not None:
        canvas = overlay_center_heatmap(canvas, center_map)
    if continuity_map is not None:
        canvas = overlay_continuity_heatmap(canvas, continuity_map)
    if structure_map is not None:
        canvas = overlay_structure_heatmap(canvas, structure_map)
    if uncertainty_map is not None:
        canvas = overlay_uncertainty_heatmap(canvas, uncertainty_map)
    return canvas


def draw_orientation_vectors(
    image_np: Any,
    orientation_map: Any,
    valid_mask: Any = None,
    step: int = 16,
    scale: float = 10.0,
) -> np.ndarray:
    image = Image.fromarray(_normalize_image(image_np)).convert('RGB')
    draw = ImageDraw.Draw(image)
    orientation = _to_numpy(orientation_map)
    if orientation is None or orientation.ndim != 3 or orientation.shape[0] != 2:
        raise ValueError('orientation_map must have shape [2, H, W].')
    h, w = orientation.shape[1:]
    mask = np.ones((h, w), dtype=np.float32) if valid_mask is None else _extract_single_channel(valid_mask)
    if mask is None:
        mask = np.ones((h, w), dtype=np.float32)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask[y, x] <= 0:
                continue
            dx = float(orientation[0, y, x]) * scale
            dy = float(orientation[1, y, x]) * scale
            draw.line((x, y, x + dx, y + dy), fill=(255, 255, 0), width=1)
    return np.asarray(image)


def _blank_tile(height: int, width: int, color: tuple[int, int, int] = DEFAULT_BG_COLOR) -> np.ndarray:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    tile[..., 0] = color[0]
    tile[..., 1] = color[1]
    tile[..., 2] = color[2]
    return tile


def _append_panel(panels: list[Any], labels: list[str], image: Any, label: str) -> None:
    panels.append(image)
    labels.append(label)


def _stage_fields(stage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(stage, dict):
        return {}
    return stage.get('fields', stage)


def _stage_refined(stage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(stage, dict):
        return {}
    refined = stage.get('refined')
    if isinstance(refined, dict):
        return refined
    if 'structure' in stage:
        return {'structure': stage.get('structure')}
    return {}


def _stage_decoded(stage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(stage, dict):
        return {}
    decoded = stage.get('decoded')
    if isinstance(decoded, dict):
        return decoded
    return stage


def make_summary_panel(
    images,
    labels: list[str] | None = None,
    columns: int = 3,
    bg_color: tuple[int, int, int] = DEFAULT_BG_COLOR,
    allow_none: bool = True,
) -> np.ndarray:
    if not images:
        raise ValueError('images must not be empty.')

    normalized: list[np.ndarray] = []
    for image in images:
        if image is None and allow_none:
            normalized.append(None)
        else:
            normalized.append(_normalize_image(image))

    valid = [image for image in normalized if image is not None]
    if not valid:
        raise ValueError('At least one image must be non-empty.')

    tile_h = max(image.shape[0] for image in valid)
    tile_w = max(image.shape[1] for image in valid)
    rows = (len(normalized) + columns - 1) // columns
    panel = Image.new('RGB', (columns * tile_w, rows * tile_h), color=bg_color)
    draw = ImageDraw.Draw(panel)

    for idx, image_np in enumerate(normalized):
        if image_np is None:
            image = Image.fromarray(_blank_tile(tile_h, tile_w, color=bg_color))
        else:
            image = Image.fromarray(image_np)
            if image.size != (tile_w, tile_h):
                image = ImageOps.pad(image, (tile_w, tile_h), color=bg_color)
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
        panel.paste(image, (x, y))

        if labels and idx < len(labels):
            label = labels[idx]
            label_x = x + 8
            label_y = y + 8
            box_w = min(tile_w - 12, max(60, 8 * len(label) + 12))
            draw.rectangle((label_x - 4, label_y - 4, label_x + box_w, label_y + 20), fill=(0, 0, 0))
            draw.text((label_x, label_y), label, fill=DEFAULT_LABEL_COLOR)

    return np.asarray(panel)


def field_summary_panel(
    image_np: Any,
    gt_fields: dict[str, Any] | None = None,
    pred_fields: dict[str, Any] | None = None,
    valid_masks: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    image = _normalize_image(image_np)
    gt_fields = gt_fields or {}
    pred_fields = pred_fields or {}
    valid_masks = valid_masks or {}
    panels: list[Any] = []
    labels: list[str] = []

    _append_panel(panels, labels, image, 'input')
    if gt_fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, gt_fields['center']), 'center_gt')
    if pred_fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, pred_fields['center']), 'center_pred')
    if gt_fields.get('continuity') is not None:
        _append_panel(panels, labels, overlay_continuity_heatmap(image, gt_fields['continuity']), 'continuity_gt')
    if pred_fields.get('continuity') is not None:
        _append_panel(panels, labels, overlay_continuity_heatmap(image, pred_fields['continuity']), 'continuity_pred')
    if gt_fields.get('uncertainty') is not None:
        _append_panel(panels, labels, overlay_uncertainty_heatmap(image, gt_fields['uncertainty']), 'uncertainty_gt')
    if pred_fields.get('uncertainty') is not None:
        _append_panel(panels, labels, overlay_uncertainty_heatmap(image, pred_fields['uncertainty']), 'uncertainty_pred')
    if pred_fields.get('orientation') is not None:
        _append_panel(
            panels,
            labels,
            draw_orientation_vectors(image, pred_fields['orientation'], valid_masks.get('orientation')),
            'orientation_pred',
        )
    if gt_fields.get('orientation') is not None:
        _append_panel(
            panels,
            labels,
            draw_orientation_vectors(image, gt_fields['orientation'], valid_masks.get('orientation')),
            'orientation_gt',
        )
    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)


def refinement_panel(
    image_np: Any,
    fields: dict[str, Any] | None = None,
    refined: dict[str, Any] | None = None,
    valid_masks: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    image = _normalize_image(image_np)
    fields = fields or {}
    refined = refined or {}
    valid_masks = valid_masks or {}
    panels: list[Any] = []
    labels: list[str] = []

    _append_panel(panels, labels, image, 'input')
    if fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, fields['center']), 'center')
    if fields.get('continuity') is not None:
        _append_panel(panels, labels, overlay_continuity_heatmap(image, fields['continuity']), 'continuity')
    if fields.get('uncertainty') is not None:
        _append_panel(panels, labels, overlay_uncertainty_heatmap(image, fields['uncertainty']), 'uncertainty')
    if refined.get('structure') is not None:
        _append_panel(panels, labels, overlay_structure_heatmap(image, refined['structure']), 'refined_structure')
    if fields.get('orientation') is not None:
        _append_panel(
            panels,
            labels,
            draw_orientation_vectors(image, fields['orientation'], valid_masks.get('orientation')),
            'orientation',
        )
    if fields or refined:
        _append_panel(
            panels,
            labels,
            overlay_field_bundle(
                image,
                center_map=fields.get('center'),
                continuity_map=fields.get('continuity'),
                structure_map=refined.get('structure'),
                uncertainty_map=fields.get('uncertainty'),
            ),
            'field_bundle',
        )
    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)


def decoder_panel(
    image_np: Any,
    rows=None,
    seeds=None,
    raw_trajectories=None,
    final_trajectories=None,
    fields: dict[str, Any] | None = None,
    refined: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    image = _normalize_image(image_np)
    fields = fields or {}
    refined = refined or {}
    final_rows = final_trajectories or rows or []
    panels: list[Any] = []
    labels: list[str] = []

    _append_panel(panels, labels, image, 'input')
    _append_panel(panels, labels, draw_seeds(image, seeds or []), 'seeds')
    if raw_trajectories is not None:
        _append_panel(panels, labels, draw_trajectories(image, raw_trajectories or [], color=(80, 180, 255), width=2, seeds=seeds), 'raw_trajectories')
    _append_panel(panels, labels, draw_trajectories(image, final_rows, color=(255, 128, 0), width=2, seeds=seeds), 'final_trajectories')
    if rows is not None:
        _append_panel(panels, labels, draw_rows(image, rows or [], seeds=seeds, color=(255, 128, 0), width=2), 'decoded_rows')
    if fields or refined:
        _append_panel(
            panels,
            labels,
            overlay_field_bundle(
                image,
                center_map=fields.get('center'),
                continuity_map=fields.get('continuity'),
                structure_map=refined.get('structure'),
                uncertainty_map=fields.get('uncertainty'),
            ),
            'decode_support',
        )
    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)


def base_core_full_comparison_panel(
    image_np: Any,
    base: dict[str, Any] | None = None,
    core: dict[str, Any] | None = None,
    full: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    image = _normalize_image(image_np)
    panels: list[Any] = []
    labels: list[str] = []
    stages = [
        ('base', base),
        ('core', core),
        ('full', full),
    ]

    _append_panel(panels, labels, image, 'input')
    for stage_name, stage in stages:
        if stage is None:
            continue
        fields = _stage_fields(stage)
        refined = _stage_refined(stage)
        decoded = _stage_decoded(stage)
        seeds = decoded.get('seeds', []) if isinstance(decoded, dict) else []
        rows = decoded.get('rows', []) if isinstance(decoded, dict) else []
        _append_panel(
            panels,
            labels,
            overlay_field_bundle(
                image,
                center_map=fields.get('center'),
                continuity_map=fields.get('continuity'),
                structure_map=refined.get('structure'),
                uncertainty_map=fields.get('uncertainty'),
            ),
            f'{stage_name}_fields',
        )
        _append_panel(
            panels,
            labels,
            draw_rows(image, rows, seeds=seeds, color=(255, 128, 0), width=2),
            f'{stage_name}_rows',
        )
    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)


def make_prediction_summary_panel(
    image_np: Any,
    gt_fields: dict[str, Any] | None = None,
    pred_fields: dict[str, Any] | None = None,
    refined_structure: Any = None,
    gt_rows=None,
    pred_rows=None,
    seeds=None,
    raw_trajectories=None,
    final_trajectories=None,
    valid_masks: dict[str, Any] | None = None,
) -> np.ndarray:
    image = _normalize_image(image_np)
    gt_fields = gt_fields or {}
    pred_fields = pred_fields or {}
    valid_masks = valid_masks or {}

    panels = [
        image,
        draw_rows(image, gt_rows or [], color=(0, 255, 0), width=2),
        draw_rows(image, pred_rows or [], seeds=seeds, color=(255, 128, 0), width=2),
        None if gt_fields.get('center') is None else overlay_center_heatmap(image, gt_fields.get('center')),
        None if pred_fields.get('center') is None else overlay_center_heatmap(image, pred_fields.get('center')),
        None if gt_fields.get('continuity') is None else overlay_continuity_heatmap(image, gt_fields.get('continuity')),
        None if pred_fields.get('continuity') is None else overlay_continuity_heatmap(image, pred_fields.get('continuity')),
        None if gt_fields.get('uncertainty') is None else overlay_uncertainty_heatmap(image, gt_fields.get('uncertainty')),
        None if pred_fields.get('uncertainty') is None else overlay_uncertainty_heatmap(image, pred_fields.get('uncertainty')),
        None if refined_structure is None else overlay_structure_heatmap(image, refined_structure),
        draw_seeds(image, seeds or []),
        draw_trajectories(image, raw_trajectories or [], color=(80, 180, 255), width=2, seeds=seeds),
        draw_trajectories(image, final_trajectories or pred_rows or [], color=(255, 128, 0), width=2, seeds=seeds),
    ]
    labels = [
        'input',
        'gt_rows',
        'pred_rows',
        'center_gt',
        'center_pred',
        'continuity_gt',
        'continuity_pred',
        'uncertainty_gt',
        'uncertainty_pred',
        'refined_structure',
        'seeds',
        'raw_trajectories',
        'final_trajectories',
    ]

    if pred_fields.get('orientation') is not None:
        panels.append(draw_orientation_vectors(image, pred_fields['orientation'], valid_masks.get('orientation')))
        labels.append('orientation_pred')
    if gt_fields.get('orientation') is not None:
        panels.append(draw_orientation_vectors(image, gt_fields['orientation'], valid_masks.get('orientation')))
        labels.append('orientation_gt')

    return make_summary_panel(panels, labels=labels, columns=3, allow_none=True)


def save_prediction_visualization(
    path,
    image_np,
    center_map,
    rows,
    seeds=None,
    continuity_map=None,
    structure_map=None,
    uncertainty_map=None,
):
    overlay = overlay_field_bundle(
        image_np,
        center_map=center_map,
        continuity_map=continuity_map,
        structure_map=structure_map,
        uncertainty_map=uncertainty_map,
    )
    overlay = draw_rows(overlay, rows, seeds)
    save_image(path, overlay)


def save_field_visualizations(output_dir, stem, image_np, fields, refined=None, decoded=None, valid_masks=None):
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    image = _normalize_image(image_np)
    panels = [image]
    labels = ['image']

    center_map = fields.get('center') if fields is not None else None
    continuity_map = fields.get('continuity') if fields is not None else None
    uncertainty_map = fields.get('uncertainty') if fields is not None else None
    orientation_map = fields.get('orientation') if fields is not None else None
    structure_map = None if refined is None else refined.get('structure')

    if center_map is not None:
        center_overlay = overlay_center_heatmap(image, center_map)
        save_image(target_dir / f'{stem}_center.png', center_overlay)
        panels.append(center_overlay)
        labels.append('center')
    if continuity_map is not None:
        continuity_overlay = overlay_continuity_heatmap(image, continuity_map)
        save_image(target_dir / f'{stem}_continuity.png', continuity_overlay)
        panels.append(continuity_overlay)
        labels.append('continuity')
    if structure_map is not None:
        structure_overlay = overlay_structure_heatmap(image, structure_map)
        save_image(target_dir / f'{stem}_structure.png', structure_overlay)
        panels.append(structure_overlay)
        labels.append('structure')
    if uncertainty_map is not None:
        uncertainty_overlay = overlay_uncertainty_heatmap(image, uncertainty_map)
        save_image(target_dir / f'{stem}_uncertainty.png', uncertainty_overlay)
        panels.append(uncertainty_overlay)
        labels.append('uncertainty')
    if orientation_map is not None:
        orientation_overlay = draw_orientation_vectors(
            image,
            orientation_map,
            None if valid_masks is None else valid_masks.get('orientation'),
        )
        save_image(target_dir / f'{stem}_orientation.png', orientation_overlay)
        panels.append(orientation_overlay)
        labels.append('orientation')
    if decoded is not None:
        rows = decoded.get('rows', []) if isinstance(decoded, dict) else decoded
        seeds = decoded.get('seeds') if isinstance(decoded, dict) else None
        raw_paths = decoded.get('debug', {}).get('candidate_paths', []) if isinstance(decoded, dict) else []
        row_overlay = draw_rows(image, rows, seeds=seeds)
        raw_overlay = draw_trajectories(image, raw_paths, color=(80, 180, 255), width=2, seeds=seeds)
        seeds_overlay = draw_seeds(image, seeds)
        save_image(target_dir / f'{stem}_rows.png', row_overlay)
        save_image(target_dir / f'{stem}_raw_trajectories.png', raw_overlay)
        save_image(target_dir / f'{stem}_seeds.png', seeds_overlay)
        panels.extend([row_overlay, raw_overlay, seeds_overlay])
        labels.extend(['decoded', 'raw_trajectories', 'seeds'])
    summary = make_summary_panel(panels, labels=labels, columns=min(3, max(1, len(panels))), allow_none=True)
    save_image(target_dir / f'{stem}_summary.png', summary)
    return {
        'summary': str(target_dir / f'{stem}_summary.png'),
        'count': len(panels),
    }


def save_comparison_visualization(path, image_np, gt_rows=None, pred_rows=None, seeds=None, title_labels=None):
    gt_overlay = draw_rows(image_np, gt_rows or [], color=(0, 255, 0), width=2)
    pred_overlay = draw_rows(image_np, pred_rows or [], seeds=seeds, color=(255, 128, 0), width=2)
    labels = title_labels or ['gt_rows', 'pred_rows']
    panel = make_summary_panel([gt_overlay, pred_overlay], labels=labels, columns=2)
    save_image(path, panel)


def save_summary_panel(path, images, labels=None, columns: int = 3) -> None:
    panel = make_summary_panel(images, labels=labels, columns=columns, allow_none=True)
    save_image(path, panel)


def save_image(path, image_np):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_normalize_image(image_np)).save(target)


def describe_rows(rows: list[dict]) -> str:
    lengths = [len(_extract_points(row)) for row in rows or []]
    return f'num_rows={len(rows or [])}, point_lengths={lengths}'


def continuity_evidence_panel(
    image_np: Any,
    gt_fields: dict[str, Any] | None = None,
    pred_fields: dict[str, Any] | None = None,
    center_stage: dict[str, Any] | None = None,
    continuity_stage: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    """Create a continuity-focused panel that contrasts center support with continuity support.

    The panel is intended as mechanism evidence rather than a generic summary view:
    it places center/continuity heatmaps next to decode outputs driven by different
    support assumptions when available.
    """
    image = _normalize_image(image_np)
    gt_fields = gt_fields or {}
    pred_fields = pred_fields or {}
    panels: list[Any] = []
    labels: list[str] = []

    _append_panel(panels, labels, image, 'input')
    if gt_fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, gt_fields['center']), 'center_gt')
    if gt_fields.get('continuity') is not None:
        _append_panel(panels, labels, overlay_continuity_heatmap(image, gt_fields['continuity']), 'continuity_gt')
    if pred_fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, pred_fields['center']), 'center_pred')
    if pred_fields.get('continuity') is not None:
        _append_panel(panels, labels, overlay_continuity_heatmap(image, pred_fields['continuity']), 'continuity_pred')

    center_decoded = _stage_decoded(center_stage)
    center_rows = center_decoded.get('rows', []) if isinstance(center_decoded, dict) else []
    center_seeds = center_decoded.get('seeds', []) if isinstance(center_decoded, dict) else []
    center_raw = center_decoded.get('debug', {}).get('candidate_paths', []) if isinstance(center_decoded, dict) else []
    if center_seeds or center_rows or center_raw:
        _append_panel(panels, labels, draw_seeds(image, center_seeds), 'center_seeds')
        _append_panel(panels, labels, draw_trajectories(image, center_raw, color=(80, 180, 255), width=2, seeds=center_seeds), 'center_raw')
        _append_panel(panels, labels, draw_rows(image, center_rows, seeds=center_seeds, color=(255, 160, 0), width=2), 'center_decode')

    continuity_decoded = _stage_decoded(continuity_stage)
    continuity_rows = continuity_decoded.get('rows', []) if isinstance(continuity_decoded, dict) else []
    continuity_seeds = continuity_decoded.get('seeds', []) if isinstance(continuity_decoded, dict) else []
    continuity_raw = continuity_decoded.get('debug', {}).get('candidate_paths', []) if isinstance(continuity_decoded, dict) else []
    if continuity_seeds or continuity_rows or continuity_raw:
        _append_panel(panels, labels, draw_seeds(image, continuity_seeds), 'support_seeds')
        _append_panel(panels, labels, draw_trajectories(image, continuity_raw, color=(80, 180, 255), width=2, seeds=continuity_seeds), 'support_raw')
        _append_panel(panels, labels, draw_rows(image, continuity_rows, seeds=continuity_seeds, color=(0, 220, 180), width=2), 'support_decode')

    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)



def uncertainty_evidence_panel(
    image_np: Any,
    pred_fields: dict[str, Any] | None = None,
    decoded: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    """Create a panel showing how uncertainty relates to decoding decisions.

    This function is designed to strengthen the uncertainty evidence chain by placing
    the uncertainty map next to raw/final decoding states and the combined support map.
    """
    image = _normalize_image(image_np)
    pred_fields = pred_fields or {}
    decoded = decoded or {}
    panels: list[Any] = []
    labels: list[str] = []

    rows = decoded.get('rows', []) if isinstance(decoded, dict) else []
    seeds = decoded.get('seeds', []) if isinstance(decoded, dict) else []
    raw_paths = decoded.get('debug', {}).get('candidate_paths', []) if isinstance(decoded, dict) else []

    _append_panel(panels, labels, image, 'input')
    if pred_fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, pred_fields['center']), 'center_pred')
    if pred_fields.get('continuity') is not None:
        _append_panel(panels, labels, overlay_continuity_heatmap(image, pred_fields['continuity']), 'continuity_pred')
    if pred_fields.get('uncertainty') is not None:
        _append_panel(panels, labels, overlay_uncertainty_heatmap(image, pred_fields['uncertainty']), 'uncertainty_pred')
    if seeds:
        _append_panel(panels, labels, draw_seeds(image, seeds), 'seeds')
    if raw_paths:
        _append_panel(panels, labels, draw_trajectories(image, raw_paths, color=(80, 180, 255), width=2, seeds=seeds), 'raw_trajectories')
    if rows:
        _append_panel(panels, labels, draw_rows(image, rows, seeds=seeds, color=(255, 128, 0), width=2), 'final_trajectories')
    if pred_fields:
        _append_panel(
            panels,
            labels,
            overlay_field_bundle(
                image,
                center_map=pred_fields.get('center'),
                continuity_map=pred_fields.get('continuity'),
                structure_map=pred_fields.get('structure'),
                uncertainty_map=pred_fields.get('uncertainty'),
            ),
            'support_vs_risk',
        )

    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)



def refinement_evidence_panel(
    image_np: Any,
    fields: dict[str, Any] | None = None,
    refined: dict[str, Any] | None = None,
    center_decoded: dict[str, Any] | None = None,
    refined_decoded: dict[str, Any] | None = None,
    columns: int = 3,
) -> np.ndarray:
    """Create a panel that highlights the independent value of refined structure.

    The intent is to compare center-driven decoding against refined-structure-driven
    decoding without changing the broader visualization stack.
    """
    image = _normalize_image(image_np)
    fields = fields or {}
    refined = refined or {}
    panels: list[Any] = []
    labels: list[str] = []

    _append_panel(panels, labels, image, 'input')
    if fields.get('center') is not None:
        _append_panel(panels, labels, overlay_center_heatmap(image, fields['center']), 'center_pred')
    if refined.get('structure') is not None:
        _append_panel(panels, labels, overlay_structure_heatmap(image, refined['structure']), 'refined_structure')
    if fields or refined:
        _append_panel(
            panels,
            labels,
            overlay_field_bundle(
                image,
                center_map=fields.get('center'),
                continuity_map=fields.get('continuity'),
                structure_map=refined.get('structure'),
                uncertainty_map=fields.get('uncertainty'),
            ),
            'field_bundle',
        )

    center_stage = _stage_decoded(center_decoded)
    center_rows = center_stage.get('rows', []) if isinstance(center_stage, dict) else []
    center_seeds = center_stage.get('seeds', []) if isinstance(center_stage, dict) else []
    if center_seeds or center_rows:
        _append_panel(panels, labels, draw_seeds(image, center_seeds), 'center_seeds')
        _append_panel(panels, labels, draw_rows(image, center_rows, seeds=center_seeds, color=(255, 160, 0), width=2), 'center_decode')

    refined_stage = _stage_decoded(refined_decoded)
    refined_rows = refined_stage.get('rows', []) if isinstance(refined_stage, dict) else []
    refined_seeds = refined_stage.get('seeds', []) if isinstance(refined_stage, dict) else []
    if refined_seeds or refined_rows:
        _append_panel(panels, labels, draw_seeds(image, refined_seeds), 'refined_seeds')
        _append_panel(panels, labels, draw_rows(image, refined_rows, seeds=refined_seeds, color=(0, 220, 180), width=2), 'refined_decode')

    return make_summary_panel(panels, labels=labels, columns=columns, allow_none=True)
