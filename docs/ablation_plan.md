# Ablation Plan

## Purpose

This document fixes a standard ablation matrix for the current SFR-Net project so that later training, result collection, and paper drafting all refer to the same experimental structure.

The goal is not to fill in results here, but to define:

- which experiments should be run
- what each experiment changes
- which config keys or modules are involved
- which metrics should be reported consistently

This plan is aligned with the current single-project implementation:

- `sfr_net/configs/base.yaml`
- `sfr_net/configs/core.yaml`
- `sfr_net/configs/full.yaml`
- `sfr_net/models/reasoning/`
- `sfr_net/models/decoder/`
- `sfr_net/losses/criterion.py`

## General Reporting Rules

For all ablation groups, keep the following as stable as possible unless the experiment is specifically testing them:

- dataset split
- image resolution
- backbone
- optimizer and base learning rate
- training epochs
- evaluation protocol
- decoding post-processing thresholds not under test

Recommended primary metrics:

- `row_f1`
- `row_precision`
- `row_recall`
- `pixel_f1_center`
- `pixel_f1_continuity`
- `uncertainty_mae`
- `avg_centerline_distance`

Recommended auxiliary metrics:

- `false_split`
- `false_merge`
- `decoded_row_count`
- `gt_row_count`

Recommended qualitative checks:

- one easy sample
- one crowded / near-row sample
- one weak-contrast sample
- one difficult Full-stage ambiguity sample

## Ablation Group A: Method Structure Ablation

This is the main method-evolution matrix. It should be the core table in later writing.

## A1. Base

### Goal

Evaluate the minimal field-and-decoder pipeline.

### Active components

- `center`
- `orientation`
- seed generation from `center`
- bidirectional propagation

### Config reference

Start from:

- `sfr_net/configs/base.yaml`

### Key switches

- `model.use_continuity = false`
- `model.use_uncertainty = false`
- `model.use_reasoning = false`
- `dataset.labels.generate_continuity = false`
- `dataset.labels.generate_uncertainty = false`
- `infer.seed_from = center`

### Suggested name

- `Base`

## A2. +K

### Goal

Measure the effect of continuity supervision without explicit reasoning.

### Active components

- Base components
- continuity label generation
- continuity head
- continuity loss

### Config idea

Use `base.yaml` as the starting point, then enable:

- `model.use_continuity = true`
- `dataset.labels.generate_continuity = true`
- `dataset.labels.use_soft_continuity = true`
- `loss.lambda_continuity > 0`
- `loss.lambda_structure = 0`
- `model.use_reasoning = false`

### Suggested name

- `Base + K`

## A3. +Reasoning

### Goal

Measure the effect of explicit structure refinement on top of continuity.

### Active components

- Base components
- continuity branch
- explicit reasoning
- refined structure output

### Config reference

Close to:

- `sfr_net/configs/core.yaml`

with decoding kept conservative if needed.

### Key switches

- `model.use_continuity = true`
- `model.use_reasoning = true`
- `model.reasoning_mode = explicit`
- `model.reasoning_num_iters = 4`
- `loss.lambda_continuity > 0`
- `loss.lambda_structure > 0`

### Suggested name

- `Base + K + Reasoning`

## A4. +Decoder Gating

### Goal

Measure the effect of continuity-guided decoder behavior on top of refined structure.

### Active components

- Core reasoning
- `infer.seed_from = refined_structure`
- continuity-aware candidate filtering
- continuity-aware stop rule

### Config reference

Start from:

- `sfr_net/configs/core.yaml`

### Key switches

- `infer.seed_from = refined_structure`
- `infer.stop_continuity_threshold > 0`

### Suggested name

- `Base + K + Reasoning + Decoder Gating`

## A5. Full

### Goal

Evaluate the complete current method.

### Active components

- center
- orientation
- continuity
- uncertainty
- explicit reasoning
- uncertainty-aware decoding

### Config reference

- `sfr_net/configs/full.yaml`

### Key switches

- `model.use_continuity = true`
- `model.use_uncertainty = true`
- `model.use_reasoning = true`
- `dataset.labels.generate_continuity = true`
- `dataset.labels.generate_uncertainty = true`
- `dataset.labels.use_pseudo_uncertainty = true`
- `loss.lambda_uncertainty > 0`
- `infer.stop_uncertainty_threshold > 0`

### Suggested name

- `Full`

## Suggested Structure Table Layout

| ID | Structure Variant | use_continuity | use_reasoning | use_uncertainty | seed_from | continuity stop | uncertainty stop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Base | false | false | false | center | no | no |
| A2 | Base + K | true | false | false | center | optional off | no |
| A3 | Base + K + Reasoning | true | true | false | center or refined_structure | optional | no |
| A4 | Base + K + Reasoning + Decoder Gating | true | true | false | refined_structure | yes | no |
| A5 | Full | true | true | true | refined_structure | yes | yes |

## Ablation Group B: Loss Ablation

This group studies whether each supervision term is actually useful.

All experiments in this group should be run relative to the strongest corresponding stage baseline.

Recommended reference baseline:

- Core loss ablation: `core.yaml`
- Full loss ablation: `full.yaml`

## B1. Remove continuity loss

### Change

- `loss.lambda_continuity = 0.0`

### Keep

- continuity branch can remain enabled so the effect is isolated to supervision removal

### Suggested name

- `w/o L_cont`

## B2. Remove structure loss

### Change

- `loss.lambda_structure = 0.0`

### Suggested name

- `w/o L_struct`

## B3. Remove uncertainty loss

### Change

- `loss.lambda_uncertainty = 0.0`

### Suggested name

- `w/o L_unc`

## B4. Remove continuity-related auxiliary regularizers

### Change candidates

- `loss.use_direction_smooth = false`
- `loss.use_continuity_preserve = false`
- `loss.use_inter_row_separation = false`

These can be tested as:

- all off together
- one-by-one if time permits

### Suggested names

- `w/o direction smooth`
- `w/o continuity preserve`
- `w/o inter-row separation`

## Suggested Loss Table Layout

| ID | Loss Variant | lambda_continuity | lambda_structure | lambda_uncertainty | direction smooth | continuity preserve | inter-row separation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Baseline | default | default | default | default | default | default |
| B1 | w/o L_cont | 0 | default | default | default | default | default |
| B2 | w/o L_struct | default | 0 | default | default | default | default |
| B3 | w/o L_unc | default | default | 0 | default | default | default |
| B4 | w/o aux regularizers | default | default | default | false | false | false |

## Ablation Group C: Decoder Ablation

This group tests the behavior of the structure-guided decoder itself.

Recommended reference baseline:

- Core decoder ablation from `core.yaml`
- Full decoder ablation from `full.yaml`

## C1. Seed source ablation

### Compare

- `infer.seed_from = center`
- `infer.seed_from = refined_structure`

### Goal

Measure whether refined structure gives better seed quality than raw center.

## C2. Continuity stop ablation

### Compare

- `infer.stop_continuity_threshold = 0.0` or disabled behavior
- `infer.stop_continuity_threshold = 0.15` or chosen baseline value

### Goal

Measure how much continuity-guided early stopping improves row quality.

## C3. Uncertainty stop ablation

### Compare

- `infer.stop_uncertainty_threshold` disabled or very high
- baseline Full value such as `0.70`

### Goal

Measure how uncertainty-aware stopping changes false extension and row reliability.

## C4. Candidate support ablation

### Compare

- decoder seeded from `center`
- decoder seeded from `refined_structure`
- Full decoder with uncertainty penalty active

### Goal

Separate support-source improvements from uncertainty-aware gating improvements.

## Suggested Decoder Table Layout

| ID | Decoder Variant | seed_from | continuity stop | uncertainty stop | stage |
| --- | --- | --- | --- | --- | --- |
| C1 | center seed | center | default | default | Core / Full |
| C2 | refined structure seed | refined_structure | default | default | Core / Full |
| C3 | no continuity stop | baseline seed | off | default | Core |
| C4 | no uncertainty stop | baseline seed | on | off | Full |
| C5 | full decoder | refined_structure | on | on | Full |

## Ablation Group D: Parameter Sensitivity

This group is for practical parameter sweeps rather than binary module toggles.

## D1. Reasoning iterations

### Parameter

- `model.reasoning_num_iters`

### Recommended values

- `0`
- `2`
- `4`
- `6`

### Goal

Measure whether more refinement iterations continue to help or begin to saturate.

## D2. Orientation band width

### Parameter

- `dataset.labels.orientation_band_width`

### Recommended values

- `3`
- `5`
- `7`

### Goal

Measure sensitivity of local direction supervision bandwidth.

## D3. Continuity band width

### Parameter

- `dataset.labels.continuity_band_width`

### Recommended values

- `5`
- `7`
- `9`

### Goal

Measure sensitivity of continuity support spread.

## D4. Continuity stop threshold

### Parameter

- `infer.stop_continuity_threshold`

### Recommended values

- `0.05`
- `0.10`
- `0.15`
- `0.20`

### Goal

Measure the decoder¡¯s tolerance to weak continuity regions.

## D5. Uncertainty stop threshold

### Parameter

- `infer.stop_uncertainty_threshold`

### Recommended values

- `0.60`
- `0.70`
- `0.80`
- `0.90`

### Goal

Measure how strongly the Full decoder avoids high-uncertainty regions.

## D6. Seed threshold

### Parameter

- `infer.seed_threshold`

### Recommended values

- `0.25`
- `0.35`
- `0.45`

### Goal

Measure the tradeoff between missing rows and noisy seeds.

## Suggested Parameter Table Layout

| ID | Parameter | Values | Main metrics |
| --- | --- | --- | --- |
| D1 | reasoning_num_iters | 0 / 2 / 4 / 6 | row_f1, avg_centerline_distance |
| D2 | orientation_band_width | 3 / 5 / 7 | pixel_f1_center, row_f1 |
| D3 | continuity_band_width | 5 / 7 / 9 | pixel_f1_continuity, row_f1 |
| D4 | stop_continuity_threshold | 0.05 / 0.10 / 0.15 / 0.20 | row_f1, false_split |
| D5 | stop_uncertainty_threshold | 0.60 / 0.70 / 0.80 / 0.90 | row_f1, uncertainty_mae, false_merge |
| D6 | seed_threshold | 0.25 / 0.35 / 0.45 | decoded_row_count, row_precision, row_recall |

## Recommended Execution Order

To keep the workload manageable, run ablations in this order:

1. structure ablation main line: `A1 -> A5`
2. loss ablation around Core and Full baselines
3. decoder ablation around Core and Full baselines
4. parameter sensitivity on the final preferred configuration

This ordering helps ensure that:

- the main method story is established first
- later ablations are anchored to a stable baseline

## Suggested Output Organization

Recommended output directory convention:

```text
outputs/
©À©¤ base_main/
©À©¤ core_main/
©À©¤ full_main/
©À©¤ ablate_loss_no_cont/
©À©¤ ablate_loss_no_struct/
©À©¤ ablate_loss_no_unc/
©À©¤ ablate_decoder_seed_center/
©À©¤ ablate_decoder_seed_refined/
©À©¤ sens_reasoning_iters_2/
©À©¤ sens_reasoning_iters_4/
©¸©¤ ...
```

For every run, keep:

- merged config snapshot
- checkpoints if needed
- `summary.json`
- `samples.json`
- representative visualizations

## Result Collection Workflow

After running a batch of experiments, use the current tooling:

### Collect experiment summaries

```bash
python -m sfr_net.tools.collect_results \
  --root_dir outputs \
  --output_file outputs/collected_results.json
```

### Generate ablation tables

```bash
python -m sfr_net.tools.make_ablation_table \
  --input_file outputs/collected_results.json \
  --output_dir outputs/ablation_tables \
  --format both
```

Optional grouping example:

```bash
python -m sfr_net.tools.make_ablation_table \
  --input_file outputs/collected_results.json \
  --output_dir outputs/ablation_tables \
  --format markdown \
  --group_by mode
```

## Final Checklist Before Running the Matrix

Before starting a full ablation sweep, verify:

- dataset format is stable
- `base.yaml`, `core.yaml`, `full.yaml` are frozen
- output directory naming is consistent
- evaluation protocol is unchanged across compared runs
- one or two visualization samples are checked for every ablation group

This document should be updated only when the real implementation changes in a way that affects the meaning of the ablation groups.
