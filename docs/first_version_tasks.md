# SFR-Net First-Version Development Tasks

## Summary

This checklist is extracted from the task breakdown document and aligned with the current repository state.
The first version prioritizes a minimal runnable loop based on center and orientation fields.

## Phase 1: Data Loop

- [x] `sfr_net/configs/default.yaml`
- [x] `sfr_net/configs/base.yaml`
- [x] `sfr_net/utils/config.py`
- [x] `sfr_net/utils/seed.py`
- [x] `sfr_net/utils/logger.py`
- [x] `sfr_net/datasets/io.py`
- [x] `sfr_net/utils/geometry.py`
- [x] `sfr_net/datasets/transforms.py`
- [x] `sfr_net/datasets/field_label_generator.py`
- [x] `sfr_net/datasets/crop_row_dataset.py`
- [x] `sfr_net/datasets/collate.py`

## Phase 2: Model Forward

- [x] `sfr_net/models/backbones/resnet.py`
- [x] `sfr_net/models/necks/fpn.py`
- [x] `sfr_net/models/context/large_kernel_context.py`
- [x] `sfr_net/models/heads/field_heads.py`
- [x] `sfr_net/models/sfr_net.py`

## Phase 3: Train Loop

- [x] `sfr_net/losses/center_loss.py`
- [x] `sfr_net/losses/orientation_loss.py`
- [x] `sfr_net/losses/criterion.py`
- [x] `sfr_net/engine/checkpoint.py`
- [x] `sfr_net/engine/trainer.py`
- [x] `sfr_net/tools/train.py`

## Phase 4: Decode Loop

- [x] `sfr_net/models/decoder/seed_generator.py`
- [x] `sfr_net/models/decoder/trajectory_propagation.py`
- [x] `sfr_net/models/decoder/postprocess.py`
- [x] `sfr_net/models/decoder/structural_decoder.py`
- [x] `sfr_net/engine/inferencer.py`
- [x] `sfr_net/tools/infer.py`

## Phase 5: Eval and Visualization

- [x] `sfr_net/metrics/pixel_metrics.py`
- [x] `sfr_net/metrics/row_metrics.py`
- [x] `sfr_net/engine/evaluator.py`
- [x] `sfr_net/utils/visualization.py`
- [x] `sfr_net/tools/test.py`
- [x] `sfr_net/tools/visualize.py`

## Remaining Work For The Next Batch

- Replace the lightweight custom ResNet with the final backbone implementation if needed.
- Improve center-field generation with a more exact distance-transform pipeline.
- Add continuity, uncertainty, and reasoning modules under the existing interfaces.
- Add richer row matching metrics and better visualization of orientation vectors.
- Run end-to-end validation once a Python runtime and sample dataset are available.
