# SFR-Net Configuration Guide

## Configuration File Hierarchy

SFR-Net uses a layered YAML configuration system.

The current configuration files are:

- `sfr_net/configs/default.yaml`
- `sfr_net/configs/base.yaml`
- `sfr_net/configs/core.yaml`
- `sfr_net/configs/full.yaml`

The loading order is:

1. load `default.yaml`
2. merge the selected stage profile such as `base.yaml`, `core.yaml`, or `full.yaml`
3. apply CLI overrides from `--set key=value`

This is implemented in `sfr_net/utils/config.py`.

## Relationship Between default / base / core / full

### `default.yaml`

`default.yaml` is the full schema and default source of truth.

It defines the global configuration tree for:

- `project`
- `dataset`
- `model`
- `context`
- `reasoning`
- `loss`
- `train`
- `infer`
- `eval`

If a field is not overridden in `base/core/full`, the project uses the value from `default.yaml`.

### `base.yaml`

`base.yaml` is the smallest stage-specific override.

It keeps only the Base-specific differences, mainly:

- `model.use_continuity = false`
- `model.use_uncertainty = false`
- `model.use_reasoning = false`
- `dataset.labels.generate_continuity = false`
- `dataset.labels.generate_uncertainty = false`
- `loss.lambda_continuity = 0.0`
- `loss.lambda_structure = 0.0`
- `loss.lambda_uncertainty = 0.0`
- `infer.seed_from = center`

### `core.yaml`

`core.yaml` enables continuity and reasoning.

Key Core overrides include:

- `model.use_continuity = true`
- `model.use_uncertainty = false`
- `model.use_reasoning = true`
- `model.reasoning_mode = explicit`
- `model.reasoning_num_iters = 4`
- `dataset.labels.generate_continuity = true`
- `dataset.labels.use_soft_continuity = true`
- `loss.lambda_continuity = 0.8`
- `loss.lambda_structure = 0.5`
- `infer.seed_from = refined_structure`
- `infer.stop_continuity_threshold = 0.15`

### `full.yaml`

`full.yaml` extends Core by enabling uncertainty.

Key Full overrides include:

- `model.use_continuity = true`
- `model.use_uncertainty = true`
- `model.use_reasoning = true`
- `dataset.labels.generate_continuity = true`
- `dataset.labels.generate_uncertainty = true`
- `dataset.labels.use_soft_continuity = true`
- `dataset.labels.use_pseudo_uncertainty = true`
- `loss.lambda_continuity = 0.8`
- `loss.lambda_structure = 0.5`
- `loss.lambda_uncertainty = 0.2`
- `infer.seed_from = refined_structure`
- `infer.stop_continuity_threshold = 0.15`
- `infer.stop_uncertainty_threshold = 0.70`

## Top-Level Field Guide

The current top-level configuration groups are:

- `project`
- `dataset`
- `model`
- `context`
- `reasoning`
- `loss`
- `train`
- `infer`
- `eval`

## `project`

Project-level runtime settings.

Current fields in `default.yaml`:

- `project.name`
- `project.seed`
- `project.output_dir`
- `project.device`
- `project.num_workers`

Practical meaning:

- `name`: project identifier for logging or export context
- `seed`: global random seed
- `output_dir`: root directory for training, testing, inference, and visualization outputs
- `device`: preferred device such as `cuda`
- `num_workers`: dataloader worker count

Commonly tuned fields:

- `project.output_dir`
- `project.device`
- `project.num_workers`

## `dataset`

Dataset location, image shape, normalization, and label generation controls.

Current fields:

- `dataset.name`
- `dataset.root`
- `dataset.train_split`
- `dataset.val_split`
- `dataset.test_split`
- `dataset.input_height`
- `dataset.input_width`
- `dataset.normalize`
- `dataset.labels.*`

### `dataset.labels`

Current label fields:

- `dataset.labels.generate_center`
- `dataset.labels.generate_orientation`
- `dataset.labels.generate_continuity`
- `dataset.labels.generate_uncertainty`
- `dataset.labels.center_sigma`
- `dataset.labels.orientation_band_width`
- `dataset.labels.continuity_band_width`
- `dataset.labels.use_soft_continuity`
- `dataset.labels.use_pseudo_uncertainty`

Practical meaning:

- `generate_center`: build center GT
- `generate_orientation`: build orientation GT
- `generate_continuity`: enable continuity GT generation
- `generate_uncertainty`: enable uncertainty GT generation
- `center_sigma`: softness of center supervision
- `orientation_band_width`: valid band around rows for orientation supervision
- `continuity_band_width`: valid band for continuity supervision
- `use_soft_continuity`: use soft continuity-style supervision instead of a harder binary form
- `use_pseudo_uncertainty`: enable heuristic uncertainty pseudo-label generation

Commonly tuned fields:

- `dataset.root`
- `dataset.input_height`
- `dataset.input_width`
- `dataset.labels.center_sigma`
- `dataset.labels.orientation_band_width`
- `dataset.labels.continuity_band_width`

## `model`

Model architecture and stage switches.

Current fields:

- `model.backbone`
- `model.pretrained`
- `model.neck`
- `model.fused_channels`
- `model.context_module`
- `model.context_channels`
- `model.use_continuity`
- `model.use_uncertainty`
- `model.use_reasoning`
- `model.reasoning_mode`
- `model.reasoning_num_iters`
- `model.decoder_mode`
- `model.decode_during_forward`

Practical meaning:

- `backbone`: current backbone selection, default is `resnet34`
- `pretrained`: whether to initialize backbone with pretrained weights
- `neck`: current feature neck type, default is `fpn`
- `fused_channels`: neck output width
- `context_module`: context aggregation module selector, currently `large_kernel` or `transformer`
- `context_channels`: channel width fed into the selected context module and field heads
- `use_continuity`: enable continuity branch
- `use_uncertainty`: enable uncertainty branch
- `use_reasoning`: enable reasoning module
- `reasoning_mode`: reasoning type, currently `explicit` or `transformer`
- `reasoning_num_iters`: number of refinement iterations for explicit reasoning
- `decoder_mode`: decoder style, current default is `greedy_bidirectional`
- `decode_during_forward`: whether to decode inside model forward during non-infer usage

Commonly tuned fields:

- `model.use_continuity`
- `model.use_uncertainty`
- `model.use_reasoning`
- `model.reasoning_mode`
- `model.reasoning_num_iters`
- `model.backbone`
- `model.context_module`
- `model.context_channels`

## Selecting Context Variants

The unified SFR-Net project currently supports two context implementations under the same main model:

- `large_kernel`: the default and current stable convolutional context block
- `transformer`: a lightweight Transformer-based context block with the same input/output tensor contract

The default behavior is:

```yaml
model:
  context_module: large_kernel
```

To switch to the transformer variant, set:

```yaml
model:
  context_module: transformer
```

The related configuration keys are split across:

- `model.context_module`
- `model.context_channels`
- `context.transformer_num_layers`
- `context.transformer_num_heads`
- `context.transformer_mlp_ratio`
- `context.transformer_dropout`

A minimal transformer context configuration looks like this:

```yaml
model:
  context_module: transformer
  context_channels: 128

context:
  transformer_num_layers: 2
  transformer_num_heads: 4
  transformer_mlp_ratio: 2.0
  transformer_dropout: 0.0
```

Practical notes:

- `large_kernel` remains the safest default if you want current stable behavior.
- `transformer` only changes the context block; it does not create a second model or change the external `SFRNet.forward(...)` contract.
- Both variants preserve the same shape contract: `[B, C, H, W] -> [B, C, H, W]`.

## Selecting Reasoning Variants

The unified SFR-Net project also supports multiple reasoning implementations under the same refined-structure output contract:

- `explicit`: the default lightweight iterative refinement path
- `transformer`: a lightweight transformer-based refinement path

The default behavior is:

```yaml
model:
  use_reasoning: true
  reasoning_mode: explicit
```

To switch to transformer reasoning, set:

```yaml
model:
  use_reasoning: true
  reasoning_mode: transformer

reasoning:
  transformer_num_layers: 2
  transformer_num_heads: 4
  transformer_mlp_ratio: 2.0
  transformer_dropout: 0.0
```

The related configuration keys are split across:

- `model.use_reasoning`
- `model.reasoning_mode`
- `model.reasoning_num_iters`
- `reasoning.transformer_num_layers`
- `reasoning.transformer_num_heads`
- `reasoning.transformer_mlp_ratio`
- `reasoning.transformer_dropout`

Practical notes:

- `explicit` remains the stable default and continues to use `model.reasoning_num_iters`.
- `transformer` keeps the same `refined["structure"]` contract for downstream decoder usage.
- Switching reasoning does not create a second model; it only swaps the refinement implementation under the same unified `SFRNet.forward(...)` interface.

## `context`

Context-module-specific settings.

Current fields:

- `context.transformer_num_layers`
- `context.transformer_num_heads`
- `context.transformer_mlp_ratio`
- `context.transformer_dropout`

Practical meaning:

- `transformer_num_layers`: number of transformer encoder layers used by `TransformerContext`
- `transformer_num_heads`: number of attention heads
- `transformer_mlp_ratio`: feed-forward expansion ratio inside the transformer encoder
- `transformer_dropout`: dropout used inside transformer encoder layers

Commonly tuned fields:

- `context.transformer_num_layers`
- `context.transformer_num_heads`
- `context.transformer_mlp_ratio`
- `context.transformer_dropout`

## `reasoning`

Reasoning-module-specific settings.

Current fields:

- `reasoning.transformer_num_layers`
- `reasoning.transformer_num_heads`
- `reasoning.transformer_mlp_ratio`
- `reasoning.transformer_dropout`

Practical meaning:

- `transformer_num_layers`: number of transformer encoder layers used by `TransformerStructureRefiner`
- `transformer_num_heads`: number of attention heads used by transformer reasoning
- `transformer_mlp_ratio`: feed-forward expansion ratio inside the transformer reasoning encoder
- `transformer_dropout`: dropout used inside transformer reasoning layers

Commonly tuned fields:

- `reasoning.transformer_num_layers`
- `reasoning.transformer_num_heads`
- `reasoning.transformer_mlp_ratio`
- `reasoning.transformer_dropout`

## `loss`

Unified training loss configuration.

Current fields:

- `loss.center_type`
- `loss.orientation_type`
- `loss.continuity_type`
- `loss.uncertainty_type`
- `loss.lambda_center`
- `loss.lambda_orientation`
- `loss.lambda_orientation_norm`
- `loss.lambda_continuity`
- `loss.lambda_structure`
- `loss.lambda_uncertainty`
- `loss.use_direction_smooth`
- `loss.use_continuity_preserve`
- `loss.use_inter_row_separation`

Practical meaning:

- `*_type`: select the internal loss variant
- `lambda_*`: scalar weights for each loss term
- `use_direction_smooth`: optional direction-related regularization flag
- `use_continuity_preserve`: optional continuity-preserving regularization flag
- `use_inter_row_separation`: optional row-separation regularization flag

Commonly tuned fields:

- `loss.lambda_center`
- `loss.lambda_orientation`
- `loss.lambda_continuity`
- `loss.lambda_structure`
- `loss.lambda_uncertainty`
- `loss.uncertainty_type`

## `train`

Training-loop configuration.

Current fields:

- `train.batch_size`
- `train.epochs`
- `train.optimizer`
- `train.lr`
- `train.weight_decay`
- `train.scheduler`
- `train.warmup_iters`
- `train.amp`
- `train.grad_clip`
- `train.resume`
- `train.save_every`
- `train.val_every`
- `train.augment.*`

### `train.augment`

Current augmentation fields:

- `train.augment.hflip`
- `train.augment.brightness_contrast`
- `train.augment.scale_jitter`
- `train.augment.mild_affine`
- `train.augment.gaussian_blur`
- `train.augment.strong_rotation`

Practical meaning:

- `amp`: enable mixed precision training when supported
- `grad_clip`: max gradient norm, `<= 0` means disabled
- `resume`: checkpoint path for resuming training
- `save_every`: checkpoint save period
- `val_every`: validation frequency in epochs

Commonly tuned fields:

- `train.batch_size`
- `train.epochs`
- `train.lr`
- `train.weight_decay`
- `train.amp`
- `train.grad_clip`
- `train.resume`

## `infer`

Decoder and export-time controls.

Current fields:

- `infer.save_fields`
- `infer.save_field_arrays`
- `infer.save_rows`
- `infer.save_vis`
- `infer.seed_from`
- `infer.seed_threshold`
- `infer.seed_nms_kernel`
- `infer.restrict_seed_region`
- `infer.seed_region`
- `infer.step_size`
- `infer.candidate_radius`
- `infer.stop_score_threshold`
- `infer.stop_continuity_threshold`
- `infer.stop_uncertainty_threshold`
- `infer.max_steps`
- `infer.smooth_trajectory`
- `infer.prune_short_rows`
- `infer.min_row_points`
- `infer.merge_rows`
- `infer.merge_distance_thresh`
- `infer.merge_angle_thresh`

Practical meaning:

- `save_fields`: export field visualization images
- `save_field_arrays`: export raw field arrays as `.npy` files when supported by the inferencer
- `save_rows`: export decoded row JSON
- `save_vis`: export overlay images
- `seed_from`: `center` or `refined_structure`
- `seed_threshold`: minimum support for seed generation
- `seed_nms_kernel`: local suppression kernel size
- `restrict_seed_region`: whether seed generation is spatially restricted
- `seed_region`: current region label used when restriction is enabled
- `step_size`: decoder propagation step length
- `candidate_radius`: decoder candidate search radius
- `stop_score_threshold`: minimum candidate score to continue
- `stop_continuity_threshold`: continuity stop rule
- `stop_uncertainty_threshold`: uncertainty stop rule in Full
- `max_steps`: propagation length cap
- `smooth_trajectory`: whether to smooth decoded trajectories
- `prune_short_rows`: whether to drop short trajectories
- `min_row_points`: minimum decoded row length in points
- `merge_rows`: whether to merge nearby rows
- `merge_distance_thresh`: spatial merge threshold
- `merge_angle_thresh`: angular merge threshold

Commonly tuned fields:

- `infer.save_field_arrays`
- `infer.seed_from`
- `infer.seed_threshold`
- `infer.step_size`
- `infer.candidate_radius`
- `infer.stop_score_threshold`
- `infer.stop_continuity_threshold`
- `infer.stop_uncertainty_threshold`
- `infer.min_row_points`

## `eval`

Evaluation configuration.

Current fields:

- `eval.metrics`
- `eval.eval_on_decoded_rows`

Practical meaning:

- `metrics`: list of enabled evaluation metric groups
- `eval_on_decoded_rows`: whether row-level evaluation is based on decoded trajectories

Commonly tuned fields:

- `eval.metrics`
- `eval.eval_on_decoded_rows`

## Commonly Tunable Parameters

Below are the most commonly adjusted fields in daily development.

### Data and label tuning

- `dataset.input_height`
- `dataset.input_width`
- `dataset.labels.center_sigma`
- `dataset.labels.orientation_band_width`
- `dataset.labels.continuity_band_width`

### Stage switches

- `model.use_continuity`
- `model.use_uncertainty`
- `model.use_reasoning`
- `model.reasoning_mode`
- `model.reasoning_num_iters`

### Architecture choices

- `model.backbone`
- `model.context_module`
- `model.context_channels`
- `context.transformer_num_layers`
- `context.transformer_num_heads`
- `reasoning.transformer_num_layers`
- `reasoning.transformer_num_heads`

### Loss balance

- `loss.lambda_center`
- `loss.lambda_orientation`
- `loss.lambda_continuity`
- `loss.lambda_structure`
- `loss.lambda_uncertainty`

### Training behavior

- `train.batch_size`
- `train.epochs`
- `train.lr`
- `train.amp`
- `train.grad_clip`

### Decoder behavior

- `infer.save_field_arrays`
- `infer.seed_from`
- `infer.seed_threshold`
- `infer.step_size`
- `infer.candidate_radius`
- `infer.stop_continuity_threshold`
- `infer.stop_uncertainty_threshold`
- `infer.min_row_points`

## Base / Core / Full Key Differences

The following table summarizes the most important stage-specific switches.

| Stage | use_continuity | use_uncertainty | use_reasoning | generate_continuity | generate_uncertainty | seed_from | lambda_continuity | lambda_structure | lambda_uncertainty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | false | false | false | false | false | center | 0.0 | 0.0 | 0.0 |
| Core | true | false | true | true | false | refined_structure | 0.8 | 0.5 | 0.0 |
| Full | true | true | true | true | true | refined_structure | 0.8 | 0.5 | 0.2 |

Additional differences:

- Base keeps the pipeline minimal and uses `center + orientation`
- Core adds continuity supervision, explicit reasoning, and continuity-guided decoding
- Full adds uncertainty supervision, uncertainty-aware reasoning, and uncertainty-aware decoding

## CLI Override Examples

All main tool scripts accept `--config` and `--set`.

### Change batch size and epochs

```bash
python -m sfr_net.tools.train \
  --config sfr_net/configs/full.yaml \
  --set train.batch_size=4 train.epochs=50
```

### Move outputs to another directory

```bash
python -m sfr_net.tools.train \
  --config sfr_net/configs/core.yaml \
  --set project.output_dir=./outputs_core_debug
```

### Switch context from large-kernel to transformer

```bash
python -m sfr_net.tools.train \
  --config sfr_net/configs/full.yaml \
  --set model.context_module=transformer context.transformer_num_layers=2 context.transformer_num_heads=4
```

### Switch reasoning from explicit to transformer

```bash
python -m sfr_net.tools.train \
  --config sfr_net/configs/full.yaml \
  --set model.use_reasoning=true model.reasoning_mode=transformer reasoning.transformer_num_layers=2 reasoning.transformer_num_heads=4
```

### Increase decoder seed threshold for inference

```bash
python -m sfr_net.tools.infer \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt \
  --input data/images/sample_001.png \
  --set infer.seed_threshold=0.45 infer.candidate_radius=5
```

### Disable field export during inference

```bash
python -m sfr_net.tools.infer \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt \
  --input data/images/sample_001.png \
  --set infer.save_fields=false infer.save_vis=true
```

### Force Base-style seeding while keeping Core model switches

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/core.yaml \
  --mode decode \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --set infer.seed_from=center
```

## Recommended Usage Patterns

### Base debugging

```bash
python -m sfr_net.tools.debug_labels --config sfr_net/configs/base.yaml
python -m sfr_net.tools.debug_decoder_with_gt --config sfr_net/configs/base.yaml
```

### Core training

```bash
python -m sfr_net.tools.train --config sfr_net/configs/core.yaml
```

### Full training and evaluation

```bash
python -m sfr_net.tools.train --config sfr_net/configs/full.yaml
python -m sfr_net.tools.test --config sfr_net/configs/full.yaml --checkpoint outputs/best.pt
```

### Full visualization and inference

```bash
python -m sfr_net.tools.infer --config sfr_net/configs/full.yaml --checkpoint outputs/best.pt --input data/images
python -m sfr_net.tools.visualize --config sfr_net/configs/full.yaml --mode summary --checkpoint outputs/best.pt --split test --index 0
```

## Practical Notes

- Keep `base.yaml`, `core.yaml`, and `full.yaml` small. Only override what differs from `default.yaml`.
- Add new configuration fields to `default.yaml` first so the full schema stays centralized.
- Prefer `--set` for temporary experiments and profile files for long-lived stage definitions.
- Keep stage switches and corresponding loss weights aligned. For example, if `model.use_uncertainty=false`, then `loss.lambda_uncertainty` should also be `0.0`.
- The config loader already performs simple consistency correction, such as falling back from `infer.seed_from=refined_structure` to `center` when reasoning is disabled.
