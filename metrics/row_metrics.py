from __future__ import annotations

import math
from typing import Any


DEFAULT_NUM_SAMPLES = 24
DEFAULT_MATCH_THRESHOLD = 0.25
DEFAULT_DISTANCE_THRESHOLD = 24.0
DEFAULT_MIN_POINTS = 2


def _extract_points(row: Any) -> list[list[float]]:
    if isinstance(row, dict):
        return row.get('points', [])
    return row


def _polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _sample_points(points: list[list[float]], num_samples: int = DEFAULT_NUM_SAMPLES) -> list[list[float]]:
    if not points:
        return []
    if len(points) <= num_samples:
        return points
    indices = [int(round(i * (len(points) - 1) / max(num_samples - 1, 1))) for i in range(num_samples)]
    return [points[idx] for idx in indices]


def _mean_min_distance(points_a: list[list[float]], points_b: list[list[float]]) -> float:
    if not points_a or not points_b:
        return float('inf')
    total = 0.0
    for ax, ay in points_a:
        best = min(math.hypot(ax - bx, ay - by) for bx, by in points_b)
        total += best
    return total / len(points_a)


def _symmetric_centerline_distance(points_a: list[list[float]], points_b: list[list[float]], num_samples: int = DEFAULT_NUM_SAMPLES) -> float:
    sample_a = _sample_points(points_a, num_samples=num_samples)
    sample_b = _sample_points(points_b, num_samples=num_samples)
    if not sample_a or not sample_b:
        return float('inf')
    dist_ab = _mean_min_distance(sample_a, sample_b)
    dist_ba = _mean_min_distance(sample_b, sample_a)
    return 0.5 * (dist_ab + dist_ba)


def _direction_similarity(points_a: list[list[float]], points_b: list[list[float]]) -> float:
    if len(points_a) < 2 or len(points_b) < 2:
        return 0.0
    ax0, ay0 = points_a[0]
    ax1, ay1 = points_a[-1]
    bx0, by0 = points_b[0]
    bx1, by1 = points_b[-1]
    adx, ady = ax1 - ax0, ay1 - ay0
    bdx, bdy = bx1 - bx0, by1 - by0
    an = math.hypot(adx, ady)
    bn = math.hypot(bdx, bdy)
    if an < 1e-6 or bn < 1e-6:
        return 0.0
    return abs((adx * bdx + ady * bdy) / (an * bn))


def _trajectory_match_score(
    pred_points: list[list[float]],
    gt_points: list[list[float]],
    num_samples: int = DEFAULT_NUM_SAMPLES,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> tuple[float, float]:
    pred_sample = _sample_points(pred_points, num_samples=num_samples)
    gt_sample = _sample_points(gt_points, num_samples=num_samples)
    if not pred_sample or not gt_sample:
        return 0.0, float('inf')

    centerline_distance = _symmetric_centerline_distance(pred_sample, gt_sample, num_samples=num_samples)
    if not math.isfinite(centerline_distance):
        return 0.0, float('inf')

    distance_term = max(0.0, 1.0 - centerline_distance / max(distance_threshold, 1e-6))
    direction_term = _direction_similarity(pred_sample, gt_sample)
    pred_length = _polyline_length(pred_points)
    gt_length = _polyline_length(gt_points)
    length_ratio = min(pred_length, gt_length) / max(pred_length, gt_length, 1e-6)
    score = 0.55 * distance_term + 0.25 * direction_term + 0.20 * length_ratio
    return score, centerline_distance


def _collect_candidate_matches(
    pred_rows: list[list[list[float]]],
    gt_rows: list[list[list[float]]],
    match_threshold: float,
    distance_threshold: float,
    num_samples: int,
) -> list[dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []
    for pred_idx, pred_points in enumerate(pred_rows):
        for gt_idx, gt_points in enumerate(gt_rows):
            score, centerline_distance = _trajectory_match_score(
                pred_points,
                gt_points,
                num_samples=num_samples,
                distance_threshold=distance_threshold,
            )
            if score < match_threshold or centerline_distance > distance_threshold:
                continue
            candidates.append(
                {
                    'pred_idx': pred_idx,
                    'gt_idx': gt_idx,
                    'score': score,
                    'centerline_distance': centerline_distance,
                }
            )
    candidates.sort(key=lambda item: (-float(item['score']), float(item['centerline_distance'])))
    return candidates


def row_metrics(
    pred_rows,
    gt_rows,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    min_points: int = DEFAULT_MIN_POINTS,
) -> dict[str, float]:
    pred_points_list = [points for points in (_extract_points(row) for row in pred_rows) if len(points) >= min_points]
    gt_points_list = [points for points in (_extract_points(row) for row in gt_rows) if len(points) >= min_points]

    pred_count = len(pred_points_list)
    gt_count = len(gt_points_list)
    if pred_count == 0 and gt_count == 0:
        return {
            'predicted_row_count': 0,
            'decoded_row_count': 0,
            'gt_row_count': 0,
            'matched_row_count': 0,
            'row_precision': 0.0,
            'row_recall': 0.0,
            'row_f1': 0.0,
            'mean_match_score': 0.0,
            'avg_centerline_distance': 0.0,
            'false_split': 0.0,
            'false_merge': 0.0,
        }

    candidates = _collect_candidate_matches(
        pred_points_list,
        gt_points_list,
        match_threshold=match_threshold,
        distance_threshold=distance_threshold,
        num_samples=num_samples,
    )

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, float | int]] = []
    for candidate in candidates:
        pred_idx = int(candidate['pred_idx'])
        gt_idx = int(candidate['gt_idx'])
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append(candidate)

    matched = len(matches)
    precision = matched / max(pred_count, 1)
    recall = matched / max(gt_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    mean_match_score = sum(float(item['score']) for item in matches) / max(matched, 1)
    avg_centerline_distance = sum(float(item['centerline_distance']) for item in matches) / max(matched, 1)

    unmatched_pred = pred_count - matched
    unmatched_gt = gt_count - matched
    false_split = max(float(unmatched_pred), 0.0)
    false_merge = max(float(unmatched_gt), 0.0)

    return {
        'predicted_row_count': pred_count,
        'decoded_row_count': pred_count,
        'gt_row_count': gt_count,
        'matched_row_count': matched,
        'row_precision': precision,
        'row_recall': recall,
        'row_f1': f1,
        'mean_match_score': mean_match_score,
        'avg_centerline_distance': avg_centerline_distance,
        'false_split': false_split,
        'false_merge': false_merge,
    }
