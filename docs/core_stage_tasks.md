# SFR-Net Core Development Tasks

## Core Goal

Upgrade the current Base pipeline:

- center field
- orientation field
- basic decoding

into the Core pipeline:

- center field
- orientation field
- continuity field
- explicit structural reasoning
- continuity-guided decoding

## Reuse Audit

### Can Reuse Directly

- config layering and public config loader
- dataset return protocol and collate protocol
- backbone / neck / context layout
- main model return schema
- decoder public entrypoint shape
- trainer / inferencer / evaluator skeleton
- debug tools added during Base stage

### Must Extend

- `datasets/field_label_generator.py`: generate continuity supervision and masks
- `models/heads/field_heads.py`: real continuity head output
- `losses/criterion.py`: integrate continuity and structure losses
- `models/sfr_net.py`: connect explicit reasoning
- `models/decoder/*`: accept refined structure and continuity-guided propagation
- `metrics/continuity_metrics.py`: replace placeholder implementation
- `engine/evaluator.py`: report continuity metrics
- `configs/core.yaml`: make Core config truly runnable

### Must Add

- `sfr_net/losses/continuity_loss.py`
- `sfr_net/losses/structure_loss.py`
- `sfr_net/models/reasoning/affinity.py`
- `sfr_net/models/reasoning/iterative_refiner.py`

## Current Status

### Phase Core-1: Supervision

- [x] `sfr_net/datasets/field_label_generator.py`
- [x] `sfr_net/models/heads/field_heads.py`
- [x] `sfr_net/losses/continuity_loss.py`
- [x] `sfr_net/losses/structure_loss.py`
- [x] `sfr_net/losses/criterion.py`

### Phase Core-2: Reasoning

- [x] `sfr_net/models/reasoning/affinity.py`
- [x] `sfr_net/models/reasoning/iterative_refiner.py`
- [x] `sfr_net/models/reasoning/__init__.py`
- [x] `sfr_net/models/sfr_net.py`

### Phase Core-3: Decoder Upgrade

- [x] `sfr_net/models/decoder/trajectory_propagation.py`
- [x] `sfr_net/models/decoder/structural_decoder.py`
- [x] `sfr_net/models/decoder/postprocess.py`

### Phase Core-4: Evaluation And Visualization

- [x] `sfr_net/metrics/continuity_metrics.py`
- [x] `sfr_net/engine/evaluator.py`
- [x] `sfr_net/configs/core.yaml`
- [x] `sfr_net/utils/visualization.py`
- [x] `sfr_net/metrics/row_metrics.py`

## Minimum Success Criteria

- `core.yaml` can instantiate the Core graph
- `outputs['fields']['continuity']` is no longer `None`
- `outputs['refined']['structure']` is a valid tensor when reasoning is on
- decoder can read `refined_structure` and `continuity`
- criterion reports non-zero continuity / structure items when enabled
- evaluator reports continuity metrics without breaking Base behavior
