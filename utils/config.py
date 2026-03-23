from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return self._wrap(value)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            return ConfigNode({k: cls._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def to_namespace(cfg: dict[str, Any]) -> ConfigNode:
    return ConfigNode._wrap(cfg)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f'Config at {path} must be a mapping.')
    return data


def _parse_cli_overrides(overrides: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if '=' not in item:
            continue
        key, raw_value = item.split('=', 1)
        value = yaml.safe_load(raw_value)
        cursor = parsed
        parts = key.split('.')
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return parsed


def validate_config(cfg: ConfigNode) -> None:
    if cfg.model.use_continuity is False and cfg.loss.lambda_continuity != 0.0:
        raise ValueError('loss.lambda_continuity must be 0 when continuity is disabled.')
    if cfg.model.use_uncertainty is False and cfg.loss.lambda_uncertainty != 0.0:
        raise ValueError('loss.lambda_uncertainty must be 0 when uncertainty is disabled.')
    if cfg.model.use_reasoning is False and cfg.infer.seed_from == 'refined_structure':
        cfg.infer.seed_from = 'center'


def load_config(path: str, overrides: list[str] | None = None) -> ConfigNode:
    config_path = Path(path)
    root = Path(__file__).resolve().parents[1] / 'configs'
    default_cfg = _load_yaml(root / 'default.yaml')
    override_cfg = _load_yaml(config_path)
    merged = merge_config(default_cfg, override_cfg)
    merged = merge_config(merged, _parse_cli_overrides(overrides))
    cfg = to_namespace(merged)
    validate_config(cfg)
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--config', default='sfr_net/configs/base.yaml')
    parser.add_argument('--set', nargs='*', default=[], help='Override keys like model.use_reasoning=true')
    return parser
