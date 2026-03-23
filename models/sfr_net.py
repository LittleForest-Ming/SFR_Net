from __future__ import annotations

import torch.nn as nn

from .backbones import ResNetBackbone
from .context import build_context_module
from .decoder import StructuralDecoder
from .heads import FieldHeads
from .necks import FPN
from .reasoning import build_structure_refiner


class SFRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResNetBackbone(name=cfg.model.backbone, pretrained=cfg.model.pretrained)
        self.neck = FPN(in_channels=self.backbone.out_channels, out_channels=cfg.model.fused_channels)
        # Context implementation is selected by cfg.model.context_module while preserving the same
        # [B, C, H, W] -> [B, C, H, W] contract expected by the rest of the model.
        self.context = build_context_module(cfg, channels=cfg.model.context_channels)
        self.heads = FieldHeads(cfg, in_channels=cfg.model.context_channels)
        # Reasoning implementation is selected by cfg and must keep the refined structure contract
        # stable for downstream decoder usage.
        self.refiner = build_structure_refiner(cfg)
        self.decoder = StructuralDecoder(cfg)

    def forward(self, images, targets=None, mode='train'):
        backbone_feats = self.backbone(images)
        fused = self.neck(backbone_feats)
        context = self.context(fused)

        raw_fields = self.heads(context)
        fields = {
            'center': raw_fields.get('center'),
            'orientation': raw_fields.get('orientation'),
            'continuity': raw_fields.get('continuity'),
            'uncertainty': raw_fields.get('uncertainty'),
        }

        refined = {'structure': None, 'fields': None}
        if self.refiner is not None:
            reasoning_out = self.refiner(fields)
            refined = {
                'structure': reasoning_out.get('structure'),
                'fields': reasoning_out.get('fields'),
            }

        decoded = {'rows': None, 'scores': None, 'seeds': None}
        should_decode = mode == 'infer' or self.cfg.model.decode_during_forward
        if should_decode:
            decoded_raw = self.decoder.decode(fields, refined=refined, meta=None)
            decoded = {
                'rows': decoded_raw.get('rows'),
                'scores': decoded_raw.get('scores'),
                'seeds': decoded_raw.get('seeds'),
            }

        return {
            'features': {'backbone': backbone_feats, 'fused': fused, 'context': context},
            'fields': fields,
            'refined': refined,
            'decoded': decoded,
            'aux': {
                'loss_inputs': {'targets': targets},
                'debug': {
                    'continuity_enabled': fields['continuity'] is not None,
                    'uncertainty_enabled': fields['uncertainty'] is not None,
                    'reasoning_enabled': self.refiner is not None,
                    'decode_enabled': bool(should_decode),
                },
            },
        }
