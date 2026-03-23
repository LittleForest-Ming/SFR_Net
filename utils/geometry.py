from __future__ import annotations

import math

import numpy as np


def polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def interpolate_polyline(points: list[list[float]], step: float = 1.0) -> list[list[float]]:
    if len(points) < 2:
        return points[:]
    dense: list[list[float]] = [list(points[0])]
    for start, end in zip(points[:-1], points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        num_steps = max(int(dist / max(step, 1e-6)), 1)
        for idx in range(1, num_steps + 1):
            alpha = idx / num_steps
            dense.append([start[0] + alpha * dx, start[1] + alpha * dy])
    return dense


def clip_points(points: list[list[float]], h: int, w: int) -> list[list[float]]:
    clipped = []
    last_point = None
    for x, y in points:
        current = [float(min(max(x, 0.0), w - 1.0)), float(min(max(y, 0.0), h - 1.0))]
        if last_point is None or current[0] != last_point[0] or current[1] != last_point[1]:
            clipped.append(current)
            last_point = current
    return clipped


def rasterize_polyline(points: list[list[float]], h: int, w: int) -> np.ndarray:
    raster = np.zeros((h, w), dtype=np.float32)
    for x, y in interpolate_polyline(points, step=0.5):
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            raster[iy, ix] = 1.0
    return raster


def compute_tangent(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return []
    tangents: list[list[float]] = []
    last_valid = [1.0, 0.0]
    for idx in range(len(points)):
        tangent = None
        for radius in range(1, len(points)):
            left_idx = max(idx - radius, 0)
            right_idx = min(idx + radius, len(points) - 1)
            prev_pt = points[left_idx]
            next_pt = points[right_idx]
            dx = next_pt[0] - prev_pt[0]
            dy = next_pt[1] - prev_pt[1]
            norm = math.hypot(dx, dy)
            if norm > 0:
                tangent = [dx / norm, dy / norm]
                break
        if tangent is None:
            tangent = last_valid
        else:
            last_valid = tangent
        tangents.append(tangent)
    return tangents
