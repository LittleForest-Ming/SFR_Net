from __future__ import annotations

import math

import numpy as np
import torch

from ..utils.geometry import clip_points, compute_tangent, interpolate_polyline, rasterize_polyline


class FieldLabelGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_center_field(self, h: int, w: int, row_polylines: list[list[list[float]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = float(self.cfg.dataset.labels.center_sigma)
        center = np.zeros((h, w), dtype=np.float32)
        row_raster = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for row in row_polylines:
            dense = clip_points(interpolate_polyline(row, step=1.0), h, w)
            if len(dense) < 2:
                continue
            row_raster = np.maximum(row_raster, rasterize_polyline(dense, h, w))
            for x, y in dense:
                dist2 = (xx - x) ** 2 + (yy - y) ** 2
                center = np.maximum(center, np.exp(-dist2 / (2 * sigma * sigma)))
        valid_mask = np.ones((h, w), dtype=np.float32)
        return torch.from_numpy(center[None]), torch.from_numpy(valid_mask[None]), torch.from_numpy(row_raster[None])

    def build_orientation_field(self, h: int, w: int, row_polylines: list[list[list[float]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        band = int(self.cfg.dataset.labels.orientation_band_width)
        orientation = np.zeros((2, h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        narrow_band = np.zeros((h, w), dtype=np.float32)
        best_dist2 = np.full((h, w), np.inf, dtype=np.float32)
        for row in row_polylines:
            dense = clip_points(interpolate_polyline(row, step=1.0), h, w)
            if len(dense) < 2:
                continue
            tangents = compute_tangent(dense)
            for (x, y), tangent in zip(dense, tangents):
                ix = int(round(x))
                iy = int(round(y))
                for oy in range(-band, band + 1):
                    for ox in range(-band, band + 1):
                        nx = ix + ox
                        ny = iy + oy
                        dist2 = float(ox * ox + oy * oy)
                        if 0 <= nx < w and 0 <= ny < h and dist2 <= band * band:
                            if dist2 < best_dist2[ny, nx]:
                                orientation[:, ny, nx] = tangent
                                best_dist2[ny, nx] = dist2
                            mask[ny, nx] = 1.0
                            narrow_band[ny, nx] = 1.0
        valid = mask > 0
        if np.any(valid):
            norms = np.linalg.norm(orientation, axis=0)
            norms = np.where(norms > 1e-6, norms, 1.0)
            orientation[:, valid] = orientation[:, valid] / norms[valid]
        return torch.from_numpy(orientation), torch.from_numpy(mask[None]), torch.from_numpy(narrow_band[None])

    def build_inter_row_mask(self, row_raster: torch.Tensor) -> torch.Tensor | None:
        if row_raster is None:
            return None
        raster = row_raster[0].numpy()
        if np.count_nonzero(raster) == 0:
            return torch.zeros_like(row_raster)
        suppress = np.zeros_like(raster, dtype=np.float32)
        for shift in (-6, -3, 3, 6):
            suppress = np.maximum(suppress, np.roll(raster, shift=shift, axis=1))
        suppress = np.clip(suppress - raster, 0.0, 1.0)
        return torch.from_numpy(suppress[None])

    def _local_mean(self, value: np.ndarray, radius: int = 1) -> np.ndarray:
        if radius <= 0:
            return value.astype(np.float32, copy=False)
        padded = np.pad(value, ((radius, radius), (radius, radius)), mode='edge')
        result = np.zeros_like(value, dtype=np.float32)
        kernel_size = float((2 * radius + 1) ** 2)
        for oy in range(2 * radius + 1):
            for ox in range(2 * radius + 1):
                result += padded[oy:oy + value.shape[0], ox:ox + value.shape[1]]
        return result / kernel_size

    def _directional_band_response(
        self,
        h: int,
        w: int,
        row_polylines: list[list[list[float]]],
        band: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        continuity = np.zeros((h, w), dtype=np.float32)
        row_raster = np.zeros((h, w), dtype=np.float32)
        lateral_sigma = max(float(band) * 0.45, 1.0)
        longitudinal_sigma = max(float(band) * 1.35, lateral_sigma + 0.5)
        longitudinal_limit = max(int(round(band * 2.2)), band + 2)

        for row in row_polylines:
            dense = clip_points(interpolate_polyline(row, step=1.0), h, w)
            if len(dense) < 2:
                continue
            row_raster = np.maximum(row_raster, rasterize_polyline(dense, h, w))
            tangents = compute_tangent(dense)
            for index, ((x, y), tangent) in enumerate(zip(dense, tangents)):
                tx, ty = float(tangent[0]), float(tangent[1])
                norm = math.hypot(tx, ty)
                if norm < 1e-6:
                    continue
                tx /= norm
                ty /= norm
                nx = -ty
                ny = tx

                ix = int(round(x))
                iy = int(round(y))
                local_radius = max(longitudinal_limit, band)
                x0 = max(ix - local_radius, 0)
                x1 = min(ix + local_radius + 1, w)
                y0 = max(iy - local_radius, 0)
                y1 = min(iy + local_radius + 1, h)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                rel_x = xx.astype(np.float32) - float(x)
                rel_y = yy.astype(np.float32) - float(y)

                along = rel_x * tx + rel_y * ty
                across = rel_x * nx + rel_y * ny
                inside = (np.abs(across) <= float(band)) & (np.abs(along) <= float(longitudinal_limit))
                if not np.any(inside):
                    continue

                local = np.exp(
                    -0.5 * (
                        (along / longitudinal_sigma) ** 2
                        + (across / lateral_sigma) ** 2
                    )
                ).astype(np.float32)
                local[~inside] = 0.0

                # Slightly downweight uncertain segment ends so continuity becomes a support band
                # for stable propagation rather than just a wider center response.
                if index == 0 or index == len(dense) - 1:
                    local *= 0.85

                continuity[y0:y1, x0:x1] = np.maximum(continuity[y0:y1, x0:x1], local)
        return continuity, row_raster

    def build_continuity_field(self, h: int, w: int, row_polylines: list[list[list[float]]]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.cfg.dataset.labels.generate_continuity:
            return None, None, None, None

        band = int(self.cfg.dataset.labels.continuity_band_width)
        continuity, row_raster = self._directional_band_response(h, w, row_polylines, band)
        continuity_mask = np.zeros((h, w), dtype=np.float32)
        continuity_band = np.zeros((h, w), dtype=np.float32)

        inter_row_mask = self.build_inter_row_mask(torch.from_numpy(row_raster[None]))
        inter_row_np = np.zeros((h, w), dtype=np.float32) if inter_row_mask is None else inter_row_mask[0].numpy().astype(np.float32)

        # Keep continuity closer to a drivable/extendable support band by suppressing inter-row ambiguity,
        # large unsupported spread, and excessively flat halo around the rows.
        continuity = np.clip(continuity - 0.35 * inter_row_np, 0.0, 1.0)
        local_smooth = self._local_mean(continuity, radius=1)
        continuity = np.clip(0.75 * continuity + 0.25 * np.minimum(continuity, local_smooth + 0.15), 0.0, 1.0)

        continuity_mask[continuity > 0.05] = 1.0
        continuity_band[continuity > 0.08] = 1.0
        return (
            torch.from_numpy(continuity[None]).float(),
            torch.from_numpy(continuity_mask[None]).float(),
            torch.from_numpy(continuity_band[None]).float(),
            inter_row_mask.float() if inter_row_mask is not None else None,
        )

    def _orientation_instability(self, orientation_np: np.ndarray, valid_mask_np: np.ndarray) -> np.ndarray:
        if orientation_np.shape[0] != 2:
            raise ValueError('orientation_np must have shape [2, H, W].')
        instability = np.zeros_like(valid_mask_np, dtype=np.float32)
        if np.count_nonzero(valid_mask_np) == 0:
            return instability
        valid = valid_mask_np > 0
        orientation_x = orientation_np[0]
        orientation_y = orientation_np[1]
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                neighbor_x = np.roll(orientation_x, shift=(oy, ox), axis=(0, 1))
                neighbor_y = np.roll(orientation_y, shift=(oy, ox), axis=(0, 1))
                neighbor_valid = np.roll(valid_mask_np, shift=(oy, ox), axis=(0, 1)) > 0
                overlap = valid & neighbor_valid
                if not np.any(overlap):
                    continue
                dot = orientation_x * neighbor_x + orientation_y * neighbor_y
                disagreement = np.clip(1.0 - np.abs(dot), 0.0, 1.0)
                instability = np.maximum(instability, disagreement * overlap.astype(np.float32))
        instability *= valid.astype(np.float32)
        return np.clip(instability, 0.0, 1.0)

    def _boundary_uncertainty(self, support_mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(support_mask) == 0:
            return np.zeros_like(support_mask, dtype=np.float32)
        blurred = self._local_mean(support_mask.astype(np.float32), radius=1)
        boundary = np.clip(blurred - support_mask.astype(np.float32), 0.0, 1.0)
        return boundary

    def build_uncertainty_field(
        self,
        h: int,
        w: int,
        center: torch.Tensor,
        orientation: torch.Tensor,
        continuity: torch.Tensor | None,
        row_raster: torch.Tensor,
        narrow_band: torch.Tensor,
        continuity_band: torch.Tensor | None,
        inter_row_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.cfg.dataset.labels.generate_uncertainty:
            return None, None, None

        center_np = center[0].numpy().astype(np.float32)
        orientation_np = orientation.numpy().astype(np.float32)
        row_np = row_raster[0].numpy().astype(np.float32)
        narrow_band_np = narrow_band[0].numpy().astype(np.float32)
        continuity_np = None if continuity is None else continuity[0].numpy().astype(np.float32)
        continuity_band_np = None if continuity_band is None else continuity_band[0].numpy().astype(np.float32)
        inter_row_np = np.zeros((h, w), dtype=np.float32) if inter_row_mask is None else inter_row_mask[0].numpy().astype(np.float32)

        if np.count_nonzero(center_np) == 0 and np.count_nonzero(row_np) == 0:
            zero = torch.zeros((1, h, w), dtype=torch.float32)
            return zero, torch.ones((1, h, w), dtype=torch.float32), zero.clone()

        orientation_mag = np.linalg.norm(orientation_np, axis=0)
        orientation_valid = np.clip(np.maximum(narrow_band_np, row_np), 0.0, 1.0)
        low_direction_conf = np.clip(1.0 - orientation_mag, 0.0, 1.0) * orientation_valid
        orientation_instability = self._orientation_instability(orientation_np, orientation_valid)

        center_response = np.clip(center_np, 0.0, 1.0)
        continuity_support = np.zeros((h, w), dtype=np.float32) if continuity_np is None else continuity_np.copy()
        if continuity_band_np is not None:
            continuity_support *= np.clip(np.maximum(continuity_band_np, row_np), 0.0, 1.0)

        support_region = np.maximum(center_response, 0.75 * orientation_valid)
        weak_continuity = support_region * np.clip(1.0 - continuity_support, 0.0, 1.0)

        center_local_mean = self._local_mean(center_response, radius=1)
        continuity_local_mean = self._local_mean(continuity_support, radius=1)
        support_transition = np.clip(np.abs(center_response - continuity_support), 0.0, 1.0)
        center_instability = np.clip(np.abs(center_response - center_local_mean) * support_region, 0.0, 1.0)
        continuity_instability = np.clip(np.abs(continuity_support - continuity_local_mean) * np.maximum(continuity_support, row_np), 0.0, 1.0)
        boundary_uncertainty = self._boundary_uncertainty(np.maximum(row_np, continuity_support))

        source_mask = np.clip(
            np.maximum.reduce([
                inter_row_np,
                weak_continuity,
                orientation_instability,
                boundary_uncertainty,
                support_transition,
            ]),
            0.0,
            1.0,
        )

        uncertainty = (
            0.30 * inter_row_np
            + 0.24 * weak_continuity
            + 0.18 * orientation_instability
            + 0.10 * low_direction_conf
            + 0.08 * support_transition
            + 0.06 * continuity_instability
            + 0.04 * center_instability
        )
        uncertainty = np.maximum(uncertainty, 0.20 * boundary_uncertainty * np.maximum(support_region, continuity_support))
        uncertainty = np.clip(uncertainty * np.maximum(source_mask, 0.15), 0.0, 1.0)
        valid_mask = np.ones((h, w), dtype=np.float32)
        return (
            torch.from_numpy(uncertainty[None]).float(),
            torch.from_numpy(valid_mask[None]).float(),
            torch.from_numpy(source_mask[None]).float(),
        )

    def build_valid_masks(self, center_mask: torch.Tensor, orientation_mask: torch.Tensor, continuity_mask, uncertainty_mask) -> dict:
        return {
            'center': center_mask,
            'orientation': orientation_mask,
            'continuity': continuity_mask,
            'uncertainty': uncertainty_mask,
        }

    def __call__(self, image_shape, row_polylines) -> dict:
        height, width = image_shape
        center, center_mask, row_raster = self.build_center_field(height, width, row_polylines)
        orientation, orientation_mask, narrow_band = self.build_orientation_field(height, width, row_polylines)
        continuity, continuity_mask, continuity_band_mask, inter_row_mask = self.build_continuity_field(height, width, row_polylines)
        uncertainty, uncertainty_mask, uncertainty_source_mask = self.build_uncertainty_field(
            height,
            width,
            center,
            orientation,
            continuity,
            row_raster,
            narrow_band,
            continuity_band_mask,
            inter_row_mask,
        )
        valid_masks = self.build_valid_masks(center_mask, orientation_mask, continuity_mask, uncertainty_mask)
        return {
            'center': center.float(),
            'orientation': orientation.float(),
            'continuity': continuity,
            'uncertainty': uncertainty,
            'valid_masks': valid_masks,
            'aux': {
                'row_polylines': row_polylines,
                'row_raster': row_raster.float(),
                'inter_row_mask': inter_row_mask,
                'narrow_band_mask': narrow_band.float(),
                'continuity_band_mask': continuity_band_mask,
                'uncertainty_source_mask': uncertainty_source_mask,
            },
        }
