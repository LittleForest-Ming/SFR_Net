# Config Schema

SFR-Net uses one layered configuration tree:

- `default.yaml`: full schema and safe defaults
- `base.yaml`: Base profile override
- `core.yaml`: Core profile override
- `full.yaml`: Full profile override

All keys use lowercase `snake_case` names and are grouped under stable top-level sections.

## Top-Level Groups

### `project`

- `name`: experiment or project name
- `seed`: random seed
- `output_dir`: root directory for checkpoints, logs, visualizations, and summaries
- `device`: requested runtime device string such as `cuda` or `cpu`
- `num_workers`: dataloader worker count

### `dataset`

- `name`: dataset identifier
- `root`: root directory containing split JSON files and image paths
- `train_split`, `val_split`, `test_split`: split names without `.json`
- `input_height`, `input_width`: final network input size
- `normalize`: whether to apply ImageNet-style normalization
- `labels`: label-generation settings

#### `dataset.labels`

- `generate_center`: enable center target generation
- `generate_orientation`: enable orientation target generation
- `generate_continuity`: enable continuity target generation
- `generate_uncertainty`: enable uncertainty target generation
- `center_sigma`: center soft-label decay width
- `orientation_band_width`: valid narrow-band width for orientation GT
- `continuity_band_width`: valid band width for continuity GT
- `use_soft_continuity`: switch for soft continuity labels
- `use_pseudo_uncertainty`: switch for heuristic uncertainty labels

### `model`

- `backbone`: backbone name
- `pretrained`: whether to request pretrained backbone weights
- `neck`: feature neck name
- `fused_channels`: neck output width
- `context_module`: context block type selector
  - `large_kernel`: use the current convolutional large-kernel context block
  - `transformer`: use the lightweight Transformer-based context block
  - default: `large_kernel`
- `context_channels`: context input/output width expected by the context module and field heads
- `use_continuity`: enable continuity head and losses
- `use_uncertainty`: enable uncertainty head and losses
- `use_reasoning`: enable structural reasoning
- `reasoning_mode`: reasoning implementation family
  - `explicit`: use the current iterative structure refiner
  - `transformer`: use the lightweight Transformer-based structure refiner
  - default: `explicit`
- `reasoning_num_iters`: number of refinement iterations for explicit reasoning
- `decoder_mode`: decoder strategy name
- `decode_during_forward`: whether to decode inside `model.forward`

### `context`

- `transformer_num_layers`: number of transformer encoder layers used when `model.context_module = transformer`
  - default: `2`
- `transformer_num_heads`: number of attention heads used by the transformer context
  - default: `4`
- `transformer_mlp_ratio`: feed-forward expansion ratio inside the transformer encoder
  - default: `2.0`
- `transformer_dropout`: dropout used inside the transformer encoder
  - default: `0.0`

Notes:

- The `context` section is only used by the transformer context path.
- When `model.context_module = large_kernel`, the model keeps the current stable large-kernel behavior and ignores the transformer-only options.
- Both context implementations preserve the same tensor contract: `[B, C, H, W] -> [B, C, H, W]`.

### `reasoning`

- `transformer_num_layers`: number of transformer encoder layers used when `model.reasoning_mode = transformer`
  - default: `2`
- `transformer_num_heads`: number of attention heads used by transformer reasoning
  - default: `4`
- `transformer_mlp_ratio`: feed-forward expansion ratio inside the transformer reasoning encoder
  - default: `2.0`
- `transformer_dropout`: dropout used inside transformer reasoning layers
  - default: `0.0`

Notes:

- The `reasoning` section is only used by the transformer reasoning path.
- When `model.reasoning_mode = explicit`, the model keeps the current iterative refinement behavior and continues to use `model.reasoning_num_iters`.
- Both reasoning implementations preserve the same refined structure contract for downstream decoder usage.

### `loss`

- `center_type`, `orientation_type`, `continuity_type`, `uncertainty_type`: loss family selectors
- `lambda_center`, `lambda_orientation`, `lambda_orientation_norm`: Base-stage loss weights
- `lambda_continuity`, `lambda_structure`, `lambda_uncertainty`: Core / Full loss weights
- `use_direction_smooth`: optional structural smoothing term switch
- `use_continuity_preserve`: optional continuity-preserving term switch
- `use_inter_row_separation`: optional inter-row separation term switch

### `train`

- `batch_size`, `epochs`, `optimizer`, `lr`, `weight_decay`, `scheduler`, `warmup_iters`: training hyper-parameters
- `amp`: enable automatic mixed precision when available
- `grad_clip`: max gradient norm
- `resume`: optional checkpoint path to resume from
- `save_every`: checkpoint interval in epochs
- `val_every`: validation interval in epochs
- `augment`: augmentation switches

#### `train.augment`

- `hflip`: random horizontal flip
- `brightness_contrast`: photometric augmentation
- `scale_jitter`: resize jitter path
- `mild_affine`: mild affine transform shared by image and polylines
- `gaussian_blur`: reserved switch
- `strong_rotation`: reserved switch

### `infer`

- `save_fields`, `save_field_arrays`, `save_rows`, `save_vis`: export switches
- `seed_from`: `center` or `refined_structure`
- `seed_threshold`, `seed_nms_kernel`: seed generation settings
- `restrict_seed_region`, `seed_region`: optional seed region restriction
- `step_size`, `candidate_radius`, `max_steps`: trajectory propagation controls
- `stop_score_threshold`, `stop_continuity_threshold`, `stop_uncertainty_threshold`: stopping gates
- `smooth_trajectory`: reserved post-smoothing switch
- `prune_short_rows`, `min_row_points`: short-trajectory filtering controls
- `merge_rows`, `merge_distance_thresh`, `merge_angle_thresh`: row merge controls

### `eval`

- `metrics`: list of summary metrics to report
- `eval_on_decoded_rows`: whether row-level metrics use decoded rows

## Profile Expectations

- `base.yaml`: `use_continuity = false`, `use_uncertainty = false`, `use_reasoning = false`
- `core.yaml`: `use_continuity = true`, `use_reasoning = true`, `use_uncertainty = false`
- `full.yaml`: `use_continuity = true`, `use_reasoning = true`, `use_uncertainty = true`

## Override Pattern

Use CLI overrides with repeated `--set key=value` pairs. Example:

```bash
python -m sfr_net.tools.train --config sfr_net/configs/full.yaml --set train.batch_size=2 --set infer.seed_threshold=0.4
```

## Stability Rule

New options should extend the existing tree instead of replacing public keys, because the engineering contract assumes the dataset, model, criterion, decoder, trainer, evaluator, and visualization tools all read from the same stable schema.
