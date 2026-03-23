from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast

from ..datasets.io import save_json
from ..engine.checkpoint import resume_training_state, save_best_checkpoint, save_last_checkpoint
from ..engine.evaluator import SFREvaluator
from ..utils.logger import build_logger
from ..utils.visualization import save_prediction_visualization


class Trainer:
    def __init__(self, cfg, model, criterion, train_loader, val_loader):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg.project.device == 'cuda' else 'cpu')
        self.model.to(self.device)
        self.output_dir = Path(cfg.project.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = build_logger('train', self.output_dir)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(int(cfg.train.epochs), 1))
        self.scaler = GradScaler(enabled=bool(cfg.train.amp and self.device.type == 'cuda'))
        self.start_epoch = 0
        self.best_metric = float('-inf')
        self.history: list[dict[str, Any]] = []
        self.fixed_vis_batch = None
        if getattr(cfg.train, 'resume', ''):
            self._resume(cfg.train.resume)

    def _resume(self, checkpoint_path: str) -> None:
        state = resume_training_state(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.start_epoch = int(state.get('epoch', -1)) + 1
        self.best_metric = float(state.get('best_metric', float('-inf')))
        self.history = state.get('history', []) or []
        self.logger.info(f'resumed from {checkpoint_path} at epoch={self.start_epoch}')

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch['image'] = batch['image'].to(self.device)
        for key in ('center', 'orientation', 'continuity', 'uncertainty'):
            if batch['targets'][key] is not None:
                batch['targets'][key] = batch['targets'][key].to(self.device)
        for key in ('center', 'orientation', 'continuity', 'uncertainty'):
            mask = batch['targets']['valid_masks'][key]
            if mask is not None:
                batch['targets']['valid_masks'][key] = mask.to(self.device)
        for key, aux in list(batch['targets']['aux'].items()):
            if torch.is_tensor(aux):
                batch['targets']['aux'][key] = aux.to(self.device)
        return batch

    def _mean_items(self, items_list: list[dict[str, float]]) -> dict[str, float]:
        if not items_list:
            return {
                'total': 0.0,
                'center': 0.0,
                'orientation': 0.0,
                'orientation_norm': 0.0,
                'continuity': 0.0,
                'structure': 0.0,
                'uncertainty': 0.0,
            }
        keys = ['total', 'center', 'orientation', 'orientation_norm', 'continuity', 'structure', 'uncertainty']
        return {
            key: sum(item.get(key, 0.0) for item in items_list) / len(items_list)
            for key in keys
        }

    def _format_loss_log(self, prefix: str, epoch: int, metrics: dict[str, float]) -> str:
        return (
            f'{prefix} epoch={epoch} '
            f'total={metrics.get("total", 0.0):.4f} '
            f'center={metrics.get("center", 0.0):.4f} '
            f'orientation={metrics.get("orientation", 0.0):.4f} '
            f'orientation_norm={metrics.get("orientation_norm", 0.0):.4f} '
            f'continuity={metrics.get("continuity", 0.0):.4f} '
            f'structure={metrics.get("structure", 0.0):.4f} '
            f'uncertainty={metrics.get("uncertainty", 0.0):.4f}'
        )

    def _denormalize_image(self, image_tensor: torch.Tensor) -> Any:
        image = image_tensor.detach().cpu().permute(1, 2, 0)
        if self.cfg.dataset.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(1, 1, 3)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(1, 1, 3)
            image = image * std + mean
        image = (image.clamp(0.0, 1.0).numpy() * 255.0).astype('uint8')
        return image

    def _save_visualization(self, epoch: int, batch: dict[str, Any], outputs: dict[str, Any]) -> None:
        if not batch['meta']:
            return
        image = self._denormalize_image(batch['image'][0])
        rows = outputs['decoded']['rows'][0] if outputs['decoded']['rows'] else []
        seeds = outputs['decoded']['seeds'][0] if outputs['decoded']['seeds'] else []
        center = outputs['fields']['center'][0, 0].detach().cpu().numpy()
        continuity = outputs['fields']['continuity'][0, 0].detach().cpu().numpy() if outputs['fields'].get('continuity') is not None else None
        structure = outputs['refined']['structure'][0, 0].detach().cpu().numpy() if outputs['refined'].get('structure') is not None else None
        uncertainty = outputs['fields']['uncertainty'][0, 0].detach().cpu().numpy() if outputs['fields'].get('uncertainty') is not None else None
        save_prediction_visualization(
            self.output_dir / 'vis' / f'epoch_{epoch:03d}.png',
            image,
            center,
            rows,
            seeds=seeds,
            continuity_map=continuity,
            structure_map=structure,
            uncertainty_map=uncertainty,
        )

    def _maybe_capture_fixed_batch(self, batch: dict[str, Any]) -> None:
        if self.fixed_vis_batch is not None:
            return
        self.fixed_vis_batch = {
            'image': batch['image'][:1].detach().cpu().clone(),
            'targets': {
                'center': None if batch['targets']['center'] is None else batch['targets']['center'][:1].detach().cpu().clone(),
                'orientation': None if batch['targets']['orientation'] is None else batch['targets']['orientation'][:1].detach().cpu().clone(),
                'continuity': None if batch['targets']['continuity'] is None else batch['targets']['continuity'][:1].detach().cpu().clone(),
                'uncertainty': None if batch['targets']['uncertainty'] is None else batch['targets']['uncertainty'][:1].detach().cpu().clone(),
                'valid_masks': {
                    key: None if value is None else value[:1].detach().cpu().clone()
                    for key, value in batch['targets']['valid_masks'].items()
                },
                'aux': {
                    key: value[:1].detach().cpu().clone() if torch.is_tensor(value) else value[:1] if isinstance(value, list) else value
                    for key, value in batch['targets']['aux'].items()
                },
            },
            'meta': batch['meta'][:1],
        }

    def _fixed_vis_enabled(self) -> bool:
        return bool(getattr(self.cfg.infer, 'save_vis', True))

    def _run_fixed_visualization(self, epoch: int) -> None:
        if self.fixed_vis_batch is None or not self._fixed_vis_enabled():
            return
        batch = {
            'image': self.fixed_vis_batch['image'].to(self.device),
            'targets': {
                key: (None if value is None else value.to(self.device))
                for key, value in self.fixed_vis_batch['targets'].items()
                if key not in {'valid_masks', 'aux'}
            },
            'meta': self.fixed_vis_batch['meta'],
        }
        batch['targets']['valid_masks'] = {
            key: (None if value is None else value.to(self.device))
            for key, value in self.fixed_vis_batch['targets']['valid_masks'].items()
        }
        batch['targets']['aux'] = {}
        for key, value in self.fixed_vis_batch['targets']['aux'].items():
            if torch.is_tensor(value):
                batch['targets']['aux'][key] = value.to(self.device)
            else:
                batch['targets']['aux'][key] = value
        with torch.no_grad():
            outputs = self.model(batch['image'], batch['targets'], mode='infer')
            outputs['decoded'] = self.model.decoder.decode(outputs['fields'], refined=outputs['refined'], meta=batch['meta'])
        self._save_visualization(epoch, batch, outputs)

    def _save_history(self) -> None:
        save_json(self.output_dir / 'train_history.json', self.history)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running_items: list[dict[str, float]] = []
        for step, batch in enumerate(self.train_loader):
            batch = self._move_batch(batch)
            if self._fixed_vis_enabled() and step == 0:
                self._maybe_capture_fixed_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(batch['image'], batch['targets'], mode='train')
                loss_dict = self.criterion(outputs, batch['targets'])
            self.scaler.scale(loss_dict['total']).backward()
            grad_clip = float(getattr(self.cfg.train, 'grad_clip', 0.0) or 0.0)
            if grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step_items = {'total': float(loss_dict['total'].item())}
            for key, value in loss_dict['items'].items():
                step_items[key] = float(value.item())
            running_items.append(step_items)

            if step % 10 == 0:
                self.logger.info(self._format_loss_log('train', epoch, step_items))

        if self.scheduler is not None:
            self.scheduler.step()
        epoch_items = self._mean_items(running_items)
        epoch_items['lr'] = float(self.optimizer.param_groups[0]['lr'])
        epoch_items['epoch'] = float(epoch)
        self.logger.info(self._format_loss_log('train_epoch', epoch, epoch_items) + f' lr={epoch_items["lr"]:.6f}')
        return epoch_items

    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        evaluator = SFREvaluator(self.cfg, export_dir=self.output_dir / 'val_results' / f'epoch_{epoch:03d}')
        running_items: list[dict[str, float]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = self._move_batch(batch)
                outputs = self.model(batch['image'], batch['targets'], mode='infer')
                outputs['decoded'] = self.model.decoder.decode(outputs['fields'], refined=outputs['refined'], meta=batch['meta'])
                loss_dict = self.criterion(outputs, batch['targets'])
                evaluator.update(outputs, batch['targets'], batch['meta'])

                step_items = {'total': float(loss_dict['total'].item())}
                for key, value in loss_dict['items'].items():
                    step_items[key] = float(value.item())
                running_items.append(step_items)

                if batch_idx == 0 and not self._fixed_vis_enabled():
                    self._save_visualization(epoch, batch, outputs)

        summary = evaluator.summarize()
        val_losses = self._mean_items(running_items)
        summary.update({f'val_{key}': value for key, value in val_losses.items()})
        self.logger.info(self._format_loss_log('val_epoch', epoch, val_losses))
        self.logger.info(
            'val_metrics epoch=%d row_f1=%.4f center_f1=%.4f continuity_f1=%.4f uncertainty_mae=%.4f false_split=%.4f false_merge=%.4f',
            epoch,
            summary.get('row_f1', 0.0),
            summary.get('pixel_f1_center', 0.0),
            summary.get('pixel_f1_continuity', 0.0),
            summary.get('uncertainty_mae', 0.0),
            summary.get('false_split', 0.0),
            summary.get('false_merge', 0.0),
        )
        return summary

    def fit(self) -> list[dict[str, Any]]:
        epochs = int(self.cfg.train.epochs)
        val_every = max(int(getattr(self.cfg.train, 'val_every', 1) or 1), 1)
        for epoch in range(self.start_epoch, epochs):
            train_log = self.train_one_epoch(epoch)
            val_log: dict[str, Any] = {}
            if self.val_loader is not None and ((epoch + 1) % val_every == 0):
                val_log = self.validate(epoch)
            if self._fixed_vis_enabled():
                self._run_fixed_visualization(epoch)

            record = {'epoch': epoch, 'train': train_log, 'val': val_log}
            self.history.append(record)
            self._save_history()

            save_last_checkpoint(
                self.output_dir,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_metric=self.best_metric,
                cfg=self.cfg,
                history=self.history,
            )

            current_metric = float(val_log.get('row_f1', float('-inf'))) if val_log else -float(train_log.get('total', 0.0))
            if current_metric >= self.best_metric:
                self.best_metric = current_metric
                save_best_checkpoint(
                    self.output_dir,
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    cfg=self.cfg,
                    history=self.history,
                )

        return self.history
