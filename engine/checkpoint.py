from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _to_serializable_config(cfg: Any) -> Any:
    """Convert config-like objects into a JSON-safe nested structure."""
    if cfg is None:
        return None
    if isinstance(cfg, (str, int, float, bool)):
        return cfg
    if isinstance(cfg, Path):
        return str(cfg)
    if isinstance(cfg, dict):
        return {str(key): _to_serializable_config(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_to_serializable_config(value) for value in cfg]
    if hasattr(cfg, 'items'):
        try:
            return {str(key): _to_serializable_config(value) for key, value in cfg.items()}
        except Exception:
            # Some config containers expose .items() but still fail during iteration.
            # In that case we intentionally continue to the __dict__ fallback below.
            pass
    if hasattr(cfg, '__dict__'):
        return {
            str(key): _to_serializable_config(value)
            for key, value in vars(cfg).items()
            if not key.startswith('_')
        }
    return str(cfg)


def _build_config_summary(cfg: Any) -> dict[str, Any] | None:
    serialized = _to_serializable_config(cfg)
    if not isinstance(serialized, dict):
        return None if serialized is None else {'value': serialized}

    project = serialized.get('project', {}) if isinstance(serialized.get('project'), dict) else {}
    dataset = serialized.get('dataset', {}) if isinstance(serialized.get('dataset'), dict) else {}
    model = serialized.get('model', {}) if isinstance(serialized.get('model'), dict) else {}
    train = serialized.get('train', {}) if isinstance(serialized.get('train'), dict) else {}

    return {
        'project_name': project.get('name'),
        'output_dir': project.get('output_dir'),
        'dataset_root': dataset.get('root'),
        'backbone': model.get('backbone'),
        'use_continuity': model.get('use_continuity'),
        'use_uncertainty': model.get('use_uncertainty'),
        'use_reasoning': model.get('use_reasoning'),
        'epochs': train.get('epochs'),
        'batch_size': train.get('batch_size'),
        'lr': train.get('lr'),
    }


def _checkpoint_payload(
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    best_metric: float | None = None,
    cfg=None,
    history=None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model': model.state_dict(),
        'epoch': int(epoch),
        'best_metric': best_metric,
        'history': history or [],
        'config': _to_serializable_config(cfg),
        'config_summary': _build_config_summary(cfg),
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def save_last_checkpoint(
    output_dir: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    best_metric: float | None = None,
    cfg=None,
    history=None,
) -> Path:
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=epoch,
        best_metric=best_metric,
        cfg=cfg,
        history=history,
    )
    return save_checkpoint(Path(output_dir) / 'last.pt', payload)


def save_best_checkpoint(
    output_dir: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    best_metric: float | None = None,
    cfg=None,
    history=None,
) -> Path:
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=epoch,
        best_metric=best_metric,
        cfg=cfg,
        history=history,
    )
    return save_checkpoint(Path(output_dir) / 'best.pt', payload)


def load_checkpoint(path: str | Path, map_location: str | torch.device = 'cpu') -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {target}')
    payload = torch.load(target, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f'Checkpoint payload must be a dict, but got: {type(payload).__name__}')
    if 'model' not in payload:
        raise KeyError(f'Checkpoint payload at {target} does not contain a model state_dict.')
    return payload


def resume_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = 'cpu',
) -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload['model'])
    if optimizer is not None and payload.get('optimizer') is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and payload.get('scheduler') is not None:
        scheduler.load_state_dict(payload['scheduler'])
    if scaler is not None and payload.get('scaler') is not None:
        scaler.load_state_dict(payload['scaler'])
    return payload


def resume_training_state(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """Backward-compatible training-state restore helper."""
    payload = resume_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=map_location,
    )
    return {
        'payload': payload,
        'epoch': int(payload.get('epoch', 0)),
        'best_metric': payload.get('best_metric'),
        'history': payload.get('history', []),
        'config': payload.get('config'),
        'config_summary': payload.get('config_summary'),
    }
