from __future__ import annotations

import math

import torch


class TrajectoryPropagator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_debug = None

    def _cfg_value(self, path: str, default: float) -> float:
        current = self.cfg
        for name in path.split('.'):
            if current is None or not hasattr(current, name):
                return float(default)
            current = getattr(current, name)
        try:
            return float(current)
        except (TypeError, ValueError):
            return float(default)

    def _score_weights(self) -> dict[str, float]:
        weights = {
            'structure': self._cfg_value('infer.structure_score_weight', 0.45),
            'continuity': self._cfg_value('infer.continuity_score_weight', 0.20),
            'direction': self._cfg_value('infer.direction_score_weight', 0.20),
            'distance': self._cfg_value('infer.distance_penalty_weight', 0.15),
            'uncertainty': self._cfg_value('infer.uncertainty_penalty_weight', 0.20),
        }
        return weights

    @staticmethod
    def _make_stop_event(event: str, point, **extra) -> dict:
        payload = {
            'event': event,
            'point': point,
        }
        payload.update(extra)
        return payload

    def _candidate_score(
        self,
        local_structure: float,
        continuity_score: float,
        dir_score: float,
        distance_penalty: float,
        uncertainty_penalty: float,
    ) -> tuple[float, dict[str, float]]:
        weights = self._score_weights()
        weighted_structure = weights['structure'] * local_structure
        weighted_continuity = weights['continuity'] * continuity_score
        weighted_direction = weights['direction'] * dir_score
        weighted_distance_penalty = weights['distance'] * distance_penalty
        weighted_uncertainty_penalty = weights['uncertainty'] * uncertainty_penalty
        score = (
            weighted_structure
            + weighted_continuity
            + weighted_direction
            - weighted_distance_penalty
            - weighted_uncertainty_penalty
        )
        return score, {
            'structure': local_structure,
            'continuity': continuity_score,
            'direction': dir_score,
            'distance_penalty': distance_penalty,
            'uncertainty_penalty': uncertainty_penalty,
            'weighted_structure': weighted_structure,
            'weighted_continuity': weighted_continuity,
            'weighted_direction': weighted_direction,
            'weighted_distance_penalty': weighted_distance_penalty,
            'weighted_uncertainty_penalty': weighted_uncertainty_penalty,
            'score': score,
        }

    def _find_directional_fallback(self, structure_map, orientation_map, current, direction_vector, radius, continuity_map=None, uncertainty_map=None):
        h, w = structure_map.shape
        best = None
        best_score = -1.0
        cx, cy = current
        dx, dy = direction_vector
        weights = self._score_weights()
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                nx = int(round(cx + ox))
                ny = int(round(cy + oy))
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                step_x = nx - cx
                step_y = ny - cy
                if step_x == 0 and step_y == 0:
                    continue
                if step_x * dx + step_y * dy <= 0:
                    continue
                local_dir = orientation_map[:, ny, nx]
                local_norm = torch.linalg.norm(local_dir).item()
                if local_norm < 1e-6:
                    continue
                local_score = weights['structure'] * float(structure_map[ny, nx].item())
                if continuity_map is not None:
                    local_score += weights['continuity'] * float(continuity_map[ny, nx].item())
                if uncertainty_map is not None:
                    local_score -= weights['uncertainty'] * float(uncertainty_map[ny, nx].item())
                if local_score > best_score:
                    best_score = local_score
                    best = [float(nx), float(ny)]
        return best

    def _trace_single(self, structure_map, orientation_map, seed, direction_sign, continuity_map=None, uncertainty_map=None):
        step_size = float(self.cfg.infer.step_size)
        radius = int(self.cfg.infer.candidate_radius)
        max_steps = int(self.cfg.infer.max_steps)
        score_thresh = float(self.cfg.infer.stop_score_threshold)
        stop_continuity_thresh = float(self.cfg.infer.stop_continuity_threshold)
        stop_uncertainty_thresh = float(self.cfg.infer.stop_uncertainty_threshold)
        h, w = structure_map.shape
        points = [seed]
        debug_steps = []
        current = [seed[0], seed[1]]
        missed_steps = 0
        max_missed_steps = int(self._cfg_value('infer.max_missed_steps', 1))
        stop_reason = self._make_stop_event('stop_reached_max_steps', point=current, num_points=len(points))
        for step_idx in range(max_steps):
            ix = int(round(current[0]))
            iy = int(round(current[1]))
            if not (0 <= ix < w and 0 <= iy < h):
                stop_reason = self._make_stop_event('out_of_bounds', point=current, step_idx=step_idx)
                debug_steps.append(stop_reason)
                break
            direction = orientation_map[:, iy, ix].clone()
            norm = torch.linalg.norm(direction).item()
            if norm < 1e-6:
                if len(points) >= 2:
                    prev = points[-2]
                    dx = current[0] - prev[0]
                    dy = current[1] - prev[1]
                    vec_norm = math.hypot(dx, dy)
                    if vec_norm > 1e-6:
                        dx /= vec_norm
                        dy /= vec_norm
                    else:
                        stop_reason = self._make_stop_event('invalid_direction', point=current, step_idx=step_idx)
                        debug_steps.append(stop_reason)
                        break
                else:
                    stop_reason = self._make_stop_event('missing_initial_direction', point=current, step_idx=step_idx)
                    debug_steps.append(stop_reason)
                    break
            else:
                direction = direction / max(norm, 1e-6)
                dx = float(direction[0].item()) * direction_sign
                dy = float(direction[1].item()) * direction_sign
            proposal = [current[0] + dx * step_size, current[1] + dy * step_size]
            best = None
            best_score = score_thresh
            best_debug = None
            candidate_debug = []
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    nx = int(round(proposal[0] + ox))
                    ny = int(round(proposal[1] + oy))
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    rel_x = nx - current[0]
                    rel_y = ny - current[1]
                    forwardness = rel_x * dx + rel_y * dy
                    if forwardness <= 0:
                        continue
                    local_structure = float(structure_map[ny, nx].item())
                    local_dir = orientation_map[:, ny, nx]
                    local_norm = torch.linalg.norm(local_dir).item()
                    if local_norm < 1e-6:
                        continue
                    local_dir = local_dir / max(local_norm, 1e-6)
                    dir_score = max(float(local_dir[0].item()) * dx + float(local_dir[1].item()) * dy, 0.0)
                    distance_penalty = math.hypot(nx - proposal[0], ny - proposal[1]) / max(radius, 1)
                    continuity_score = 0.0
                    if continuity_map is not None:
                        continuity_score = float(continuity_map[ny, nx].item())
                        if continuity_score < stop_continuity_thresh:
                            candidate_debug.append({
                                'point': [float(nx), float(ny)],
                                'proposal': [float(proposal[0]), float(proposal[1])],
                                'structure': local_structure,
                                'continuity': continuity_score,
                                'direction': dir_score,
                                'distance_penalty': distance_penalty,
                                'uncertainty_penalty': 0.0,
                                'score': None,
                                'selected': False,
                                'rejected_by': 'low_continuity',
                            })
                            continue
                    uncertainty_penalty = 0.0
                    if uncertainty_map is not None:
                        uncertainty_penalty = float(uncertainty_map[ny, nx].item())
                        if uncertainty_penalty > stop_uncertainty_thresh:
                            candidate_debug.append({
                                'point': [float(nx), float(ny)],
                                'proposal': [float(proposal[0]), float(proposal[1])],
                                'structure': local_structure,
                                'continuity': continuity_score,
                                'direction': dir_score,
                                'distance_penalty': distance_penalty,
                                'uncertainty_penalty': uncertainty_penalty,
                                'score': None,
                                'selected': False,
                                'rejected_by': 'high_uncertainty',
                            })
                            continue
                    score, score_debug = self._candidate_score(
                        local_structure,
                        continuity_score,
                        dir_score,
                        distance_penalty,
                        uncertainty_penalty,
                    )
                    candidate_item = {
                        'point': [float(nx), float(ny)],
                        'proposal': [float(proposal[0]), float(proposal[1])],
                        **score_debug,
                        'selected': False,
                        'rejected_by': '',
                    }
                    candidate_debug.append(candidate_item)
                    if score > best_score:
                        best_score = score
                        best = [float(nx), float(ny)]
                        best_debug = {
                            'event': 'select_candidate',
                            'step_idx': step_idx,
                            'point': best,
                            'proposal': [float(proposal[0]), float(proposal[1])],
                            **score_debug,
                        }
            if best is None:
                fallback = self._find_directional_fallback(structure_map, orientation_map, current, (dx, dy), radius, continuity_map=continuity_map, uncertainty_map=uncertainty_map)
                if fallback is None:
                    missed_steps += 1
                    miss_event = self._make_stop_event('miss', point=current, step_idx=step_idx, missed_steps=missed_steps)
                    debug_steps.append(miss_event)
                    if missed_steps > max_missed_steps:
                        stop_reason = self._make_stop_event('stop_after_miss_limit', point=current, step_idx=step_idx, missed_steps=missed_steps)
                        debug_steps.append(stop_reason)
                        break
                    continue
                best = fallback
                best_debug = {
                    'event': 'fallback',
                    'step_idx': step_idx,
                    'point': best,
                    'proposal': [float(proposal[0]), float(proposal[1])],
                }
            missed_steps = 0
            for item in candidate_debug:
                item['selected'] = item.get('point') == best
            if best_debug is not None:
                best_debug['candidate_count'] = len(candidate_debug)
                best_debug['candidates'] = candidate_debug
            bix = int(round(best[0]))
            biy = int(round(best[1]))
            if continuity_map is not None and continuity_map[biy, bix].item() < stop_continuity_thresh:
                stop_reason = self._make_stop_event(
                    'stop_low_continuity',
                    point=best,
                    step_idx=step_idx,
                    continuity=float(continuity_map[biy, bix].item()),
                )
                debug_steps.append(stop_reason)
                break
            if uncertainty_map is not None and uncertainty_map[biy, bix].item() > stop_uncertainty_thresh:
                stop_reason = self._make_stop_event(
                    'stop_high_uncertainty',
                    point=best,
                    step_idx=step_idx,
                    uncertainty=float(uncertainty_map[biy, bix].item()),
                )
                debug_steps.append(stop_reason)
                break
            if math.hypot(best[0] - current[0], best[1] - current[1]) < 0.5:
                stop_reason = self._make_stop_event('stop_too_close', point=best, step_idx=step_idx)
                debug_steps.append(stop_reason)
                break
            current = best
            points.append(best)
            stop_reason = self._make_stop_event('stop_reached_max_steps', point=current, step_idx=step_idx, num_points=len(points))
            if best_debug is not None:
                debug_steps.append(best_debug)
        return points, debug_steps, stop_reason

    def __call__(self, center_map, orientation_map, seeds, continuity_map=None, uncertainty_map=None):
        trajectories = []
        debug_info = []
        for seed in seeds:
            forward, forward_debug, forward_stop = self._trace_single(center_map, orientation_map, seed, 1.0, continuity_map=continuity_map, uncertainty_map=uncertainty_map)
            backward, backward_debug, backward_stop = self._trace_single(center_map, orientation_map, seed, -1.0, continuity_map=continuity_map, uncertainty_map=uncertainty_map)
            backward = list(reversed(backward[:-1])) if len(backward) > 1 else []
            full = backward + forward
            trajectories.append(full)
            debug_info.append({
                'seed': seed,
                'forward': forward_debug,
                'backward': backward_debug,
                'forward_stop_reason': forward_stop,
                'backward_stop_reason': backward_stop,
                'trajectory_stop_reason': {
                    'forward': forward_stop.get('event', ''),
                    'backward': backward_stop.get('event', ''),
                },
                'uncertainty_enabled': uncertainty_map is not None,
                'stop_uncertainty_threshold': float(self.cfg.infer.stop_uncertainty_threshold),
            })
        self.last_debug = debug_info
        return trajectories
