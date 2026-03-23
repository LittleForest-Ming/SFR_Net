# SFR-Net Full Development Tasks

## Full Goal

Upgrade the current Core pipeline:

- center field
- orientation field
- continuity field
- explicit structural reasoning
- continuity-guided decoding

into the Full pipeline:

- center field
- orientation field
- continuity field
- uncertainty field
- uncertainty-aware reasoning
- uncertainty-aware decoding
- trajectory confidence
- complete training / evaluation / visualization / docs / experiment tooling

## Reuse Audit

### Can Reuse Directly

- single-project config system and stable public interfaces
- dataset / collate protocol
- backbone / neck / context / multi-head layout
- center / orientation / continuity supervision chain
- explicit reasoning skeleton and decoder public entrypoint
- trainer / evaluator / inferencer skeleton
- debug tools created in Base/Core stages

### Must Extend

- `datasets/field_label_generator.py`: real uncertainty pseudo-labels
- `models/heads/field_heads.py`: uncertainty head output
- `losses/criterion.py`: integrate uncertainty loss
- `models/reasoning/affinity.py`: uncertainty penalty
- `models/reasoning/iterative_refiner.py`: uncertainty-aware refinement
- `models/sfr_net.py`: uncertainty-aware full graph
- `models/decoder/*`: uncertainty-aware propagation and trajectory confidence
- `metrics/*`: final continuity / uncertainty / row metrics
- `engine/*`: final trainer/evaluator/inferencer/checkpoint system
- `tools/*`: final train/test/infer/visualize and experiment tools
- docs / README / schema

### Must Add

- `sfr_net/losses/uncertainty_loss.py`
- `sfr_net/metrics/uncertainty_metrics.py`
- `sfr_net/tools/collect_results.py`
- `sfr_net/tools/make_ablation_table.py`
- `docs/method_overview.md`
- `docs/config_guide.md`
- `docs/data_format.md`
- `docs/ablation_plan.md`
- `docs/visualization_guide.md`

## Current Status

### Batch 1: Full Method Loop

- [x] `sfr_net/datasets/field_label_generator.py`
- [x] `sfr_net/models/heads/field_heads.py`
- [x] `sfr_net/losses/uncertainty_loss.py`
- [x] `sfr_net/losses/criterion.py`
- [x] `sfr_net/models/reasoning/affinity.py`
- [x] `sfr_net/models/reasoning/iterative_refiner.py`
- [x] `sfr_net/models/sfr_net.py`
- [x] `sfr_net/models/decoder/trajectory_propagation.py`
- [x] `sfr_net/models/decoder/structural_decoder.py`
- [x] `sfr_net/models/decoder/postprocess.py`
- [x] `sfr_net/configs/full.yaml`

### Batch 2: Training System

- [x] `sfr_net/engine/checkpoint.py`
- [x] `sfr_net/engine/trainer.py`
- [x] `sfr_net/tools/train.py`
- [x] `sfr_net/configs/default.yaml`

### Batch 3: Evaluation System

- [x] `sfr_net/metrics/continuity_metrics.py`
- [x] `sfr_net/metrics/uncertainty_metrics.py`
- [x] `sfr_net/metrics/row_metrics.py`
- [x] `sfr_net/engine/evaluator.py`
- [x] `sfr_net/tools/test.py`

### Batch 4: Inference And Visualization

- [x] `sfr_net/engine/inferencer.py`
- [x] `sfr_net/tools/infer.py`
- [x] `sfr_net/utils/visualization.py`
- [x] `sfr_net/tools/visualize.py`

### Batch 5: Docs And Result Tools

- [x] `sfr_net/tools/collect_results.py`
- [x] `sfr_net/tools/make_ablation_table.py`
- [x] `README.md`
- [x] `sfr_net/configs/schema.md`
- [x] `docs/method_overview.md`
- [x] `docs/config_guide.md`
- [x] `docs/data_format.md`
- [x] `docs/ablation_plan.md`
- [x] `docs/visualization_guide.md`

## Minimum Success Criteria

- `base.yaml`, `core.yaml`, and `full.yaml` all instantiate correctly
- `outputs['fields']['uncertainty']` is no longer `None` in Full mode
- `refined['structure']` is uncertainty-aware in Full mode
- decoder supports refined structure + continuity + uncertainty jointly
- trajectory outputs carry confidence-related statistics
- evaluator can report field-level and row-level Full metrics
