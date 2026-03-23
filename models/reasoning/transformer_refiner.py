from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerStructureRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        hidden_dim = self._cfg_value('model.context_channels', 128, cast=int)
        num_layers = self._cfg_value('reasoning.transformer_num_layers', 2, cast=int)
        num_heads = self._cfg_value('reasoning.transformer_num_heads', 4, cast=int)
        mlp_ratio = self._cfg_value('reasoning.transformer_mlp_ratio', 2.0, cast=float)
        dropout = self._cfg_value('reasoning.transformer_dropout', 0.0, cast=float)
        self.output_scale = 1.0

        input_channels = 5  # center(1) + orientation(2) + continuity(1) + uncertainty(1)
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1, bias=False)
        self.input_norm = nn.BatchNorm2d(hidden_dim)
        self.input_act = nn.ReLU(inplace=True)

        mlp_dim = max(int(round(hidden_dim * mlp_ratio)), hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def _resolve_path(self, path: str):
        current = self.cfg
        for name in path.split('.'):
            if current is None or not hasattr(current, name):
                return None
            current = getattr(current, name)
        return current

    def _cfg_value(self, paths: str | tuple[str, ...], default, cast):
        candidate_paths = (paths,) if isinstance(paths, str) else paths
        for path in candidate_paths:
            value = self._resolve_path(path)
            if value is None:
                continue
            try:
                return cast(value)
            except (TypeError, ValueError):
                continue
        return cast(default)

    def _build_position_encoding(self, height: int, width: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build a lightweight 2D sinusoidal position encoding with shape [1, H*W, C]."""
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        tokens = height * width
        position = torch.zeros((tokens, channels), device=device, dtype=dtype)
        quarter_channels = max(channels // 4, 1)
        scales = torch.exp(
            torch.linspace(0.0, math.log(10000.0), steps=quarter_channels, device=device, dtype=dtype)
        )

        x_flat = xx.reshape(-1, 1)
        y_flat = yy.reshape(-1, 1)
        encoded = torch.cat(
            [
                torch.sin(x_flat / scales),
                torch.cos(x_flat / scales),
                torch.sin(y_flat / scales),
                torch.cos(y_flat / scales),
            ],
            dim=1,
        )
        encoded = encoded[:, :channels]
        position[:, : encoded.shape[1]] = encoded
        return position.unsqueeze(0)

    def _compose_inputs(self, fields: dict) -> torch.Tensor:
        center = fields['center']
        orientation = fields['orientation']
        continuity = fields.get('continuity')
        uncertainty = fields.get('uncertainty')

        if continuity is None:
            continuity = torch.zeros_like(center)
        if uncertainty is None:
            uncertainty = torch.zeros_like(center)

        return torch.cat([center, orientation, continuity, uncertainty], dim=1)

    def forward(self, fields: dict) -> dict:
        center = fields['center']
        features = self._compose_inputs(fields)
        hidden = self.input_act(self.input_norm(self.input_proj(features)))

        batch_size, hidden_dim, height, width = hidden.shape
        tokens = hidden.flatten(2).transpose(1, 2)
        position = self._build_position_encoding(height, width, hidden_dim, hidden.device, hidden.dtype)
        refined_tokens = self.encoder(tokens + position)
        refined_tokens = self.token_norm(refined_tokens)
        refined_map = refined_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, height, width)

        structure_logits = self.output_proj(refined_map)
        structure = torch.sigmoid(structure_logits) * float(self.output_scale)
        structure = structure.clamp(0.0, 1.0)

        debug = {
            'mode': 'transformer',
            'token_shape': [int(batch_size), int(height * width), int(hidden_dim)],
            'input_channels': int(features.shape[1]),
            'continuity_enabled': fields.get('continuity') is not None,
            'uncertainty_enabled': fields.get('uncertainty') is not None,
        }
        return {
            'structure': structure,
            'fields': {**fields, 'structure': structure},
            'debug': debug,
        }

