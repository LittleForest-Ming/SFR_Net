from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerContext(nn.Module):
    def __init__(self, cfg, channels: int):
        super().__init__()
        self.cfg = cfg
        self.channels = int(channels)

        num_layers = self._cfg_value('context.transformer_num_layers', 2, cast=int)
        num_heads = self._cfg_value('context.transformer_num_heads', 4, cast=int)
        mlp_ratio = self._cfg_value('context.transformer_mlp_ratio', 2.0, cast=float)
        dropout = self._cfg_value('context.transformer_dropout', 0.0, cast=float)

        mlp_dim = max(int(round(self.channels * mlp_ratio)), self.channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.channels)
        self.output_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

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

    def _build_position_encoding(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build a lightweight 2D sinusoidal position encoding with shape [1, H*W, C]."""
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        tokens = height * width
        position = torch.zeros((tokens, self.channels), device=device, dtype=dtype)
        half_channels = self.channels // 2
        quarter_channels = max(half_channels // 2, 1)

        if quarter_channels > 0:
            scales = torch.exp(
                torch.linspace(0.0, math.log(10000.0), steps=quarter_channels, device=device, dtype=dtype)
            )
            x_flat = xx.reshape(-1, 1)
            y_flat = yy.reshape(-1, 1)

            x_sin = torch.sin(x_flat / scales)
            x_cos = torch.cos(x_flat / scales)
            y_sin = torch.sin(y_flat / scales)
            y_cos = torch.cos(y_flat / scales)

            chunks = [x_sin, x_cos, y_sin, y_cos]
            encoded = torch.cat(chunks, dim=1)
            encoded = encoded[:, : self.channels]
            position[:, : encoded.shape[1]] = encoded

        return position.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f'TransformerContext expected {self.channels} channels, but got {channels}.'
            )

        tokens = x.flatten(2).transpose(1, 2)
        position = self._build_position_encoding(height, width, x.device, x.dtype)
        encoded = self.encoder(tokens + position)
        encoded = self.norm(encoded)
        encoded = encoded.transpose(1, 2).reshape(batch_size, channels, height, width)

        # Keep the module shape-preserving and lightweight while allowing a final local remix.
        return self.output_proj(encoded)

