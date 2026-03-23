from __future__ import annotations

import math


class TrajectoryPostProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _direction_signature(points):
        if len(points) < 2:
            return (0.0, 0.0)
        x0, y0 = points[0]
        x1, y1 = points[-1]
        norm = math.hypot(x1 - x0, y1 - y0)
        if norm < 1e-6:
            return (0.0, 0.0)
        return ((x1 - x0) / norm, (y1 - y0) / norm)

    def __call__(self, trajectories):
        filtered = []
        min_points = int(self.cfg.infer.min_row_points)
        seen = []
        for item in trajectories:
            points = item['points']
            if len(points) < min_points:
                continue
            if item.get('avg_structure', item.get('score', 0.0)) < float(self.cfg.infer.stop_score_threshold):
                continue
            if item.get('avg_continuity', 1.0) < float(self.cfg.infer.stop_continuity_threshold):
                continue
            if item.get('avg_uncertainty', 0.0) > float(self.cfg.infer.stop_uncertainty_threshold):
                continue
            direction = self._direction_signature(points)
            key = tuple((int(round(x)), int(round(y))) for x, y in points[: min(5, len(points))])
            duplicate = False
            for existing_key, existing_dir in seen:
                dir_similarity = direction[0] * existing_dir[0] + direction[1] * existing_dir[1]
                if key == existing_key and dir_similarity > 0.9:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append((key, direction))
            filtered.append(item)
        return filtered
