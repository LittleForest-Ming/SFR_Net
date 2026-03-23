from __future__ import annotations

from ...utils.geometry import polyline_length
from .postprocess import TrajectoryPostProcessor
from .seed_generator import SeedGenerator
from .trajectory_propagation import TrajectoryPropagator


class StructuralDecoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed_generator = SeedGenerator(cfg)
        self.propagator = TrajectoryPropagator(cfg)
        self.postprocess = TrajectoryPostProcessor(cfg)

    def _select_structure_source(self, fields, refined):
        seed_from = getattr(self.cfg.infer, 'seed_from', 'center')
        refined_structure = None if refined is None else refined.get('structure')
        if seed_from == 'refined_structure' and refined_structure is not None:
            return refined_structure, 'refined_structure'
        return fields['center'], 'center'

    def decode(self, fields, refined=None, meta=None):
        structure_map, structure_source = self._select_structure_source(fields, refined)
        orientation_map = fields['orientation']
        continuity_map = fields.get('continuity')
        uncertainty_map = fields.get('uncertainty')
        all_seeds = self.seed_generator(structure_map)
        batch_rows = []
        batch_scores = []
        debug_paths = []
        propagation_debug = []
        for batch_idx, seeds in enumerate(all_seeds):
            support_map = structure_map[batch_idx, 0]
            local_continuity = None if continuity_map is None else continuity_map[batch_idx, 0]
            local_uncertainty = None if uncertainty_map is None else uncertainty_map[batch_idx, 0]
            trajectories = self.propagator(
                support_map,
                orientation_map[batch_idx],
                seeds,
                local_continuity,
                local_uncertainty,
            )
            propagation_debug.append(self.propagator.last_debug)
            rows = []
            scores = []
            for seed, points in zip(seeds, trajectories):
                if not points:
                    continue
                support_scores = [
                    support_map[int(round(y)), int(round(x))].item()
                    for x, y in points
                    if 0 <= int(round(y)) < support_map.shape[-2] and 0 <= int(round(x)) < support_map.shape[-1]
                ]
                continuity_scores = [
                    local_continuity[int(round(y)), int(round(x))].item()
                    for x, y in points
                    if local_continuity is not None and 0 <= int(round(y)) < support_map.shape[-2] and 0 <= int(round(x)) < support_map.shape[-1]
                ]
                uncertainty_scores = [
                    local_uncertainty[int(round(y)), int(round(x))].item()
                    for x, y in points
                    if local_uncertainty is not None and 0 <= int(round(y)) < support_map.shape[-2] and 0 <= int(round(x)) < support_map.shape[-1]
                ]
                mean_support = sum(support_scores) / max(len(support_scores), 1)
                mean_continuity = sum(continuity_scores) / max(len(continuity_scores), 1) if continuity_scores else 0.0
                mean_uncertainty = sum(uncertainty_scores) / max(len(uncertainty_scores), 1) if uncertainty_scores else 0.0
                score = float(0.60 * mean_support + 0.25 * mean_continuity - 0.15 * mean_uncertainty)
                rows.append({
                    'points': points,
                    'score': score,
                    'length': polyline_length(points),
                    'source_seed': seed,
                    'avg_center': float(mean_support),
                    'avg_structure': float(mean_support),
                    'avg_continuity': float(mean_continuity),
                    'avg_uncertainty': float(mean_uncertainty),
                })
                scores.append(score)
            rows = self.postprocess(rows)
            batch_rows.append(rows)
            batch_scores.append([row['score'] for row in rows])
            debug_paths.append(trajectories)
        return {
            'rows': batch_rows,
            'scores': batch_scores,
            'seeds': all_seeds,
            'debug': {
                'seed_maps': structure_map,
                'candidate_paths': debug_paths,
                'propagation': propagation_debug,
                'structure_source': structure_source,
                'seed_source': structure_source,
                'continuity_enabled': continuity_map is not None,
                'uncertainty_enabled': uncertainty_map is not None,
                'use_continuity': continuity_map is not None,
                'use_uncertainty': uncertainty_map is not None,
            },
        }
