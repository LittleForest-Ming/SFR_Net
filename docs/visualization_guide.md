# Visualization Guide

## Overview

SFR-Net provides a unified visualization stack for:

- label inspection
- prediction inspection
- decoder debugging
- summary-panel export
- result presentation and paper-figure preparation

The current visualization implementation is mainly located in:

- `sfr_net/utils/visualization.py`
- `sfr_net/tools/visualize.py`

Additional debug-oriented scripts include:

- `sfr_net/tools/debug_labels.py`
- `sfr_net/tools/debug_decoder_with_gt.py`

This guide explains what each visualization means, when it is useful, and how to generate it with the current project.

## Main Visualization Use Cases

The current visual outputs are most useful for four scenarios:

### 1. Label checking

Use these figures to verify whether the dataset and field-label generation are correct.

Recommended figures:

- input image
- gt rows overlay
- center_gt
- continuity_gt
- uncertainty_gt
- orientation vectors
- row raster related outputs from debug scripts

Recommended tools:

- `debug_labels.py`
- `visualize.py --mode label`

### 2. Training process observation

Use these figures to check whether the model is learning meaningful field maps and decoded rows.

Recommended figures:

- center_pred
- continuity_pred
- uncertainty_pred
- refined_structure
- pred rows overlay
- summary panel

Recommended tools:

- `visualize.py --mode pred`
- `visualize.py --mode summary`
- trainer-fixed visualization hook outputs under `vis/`

### 3. Failure case analysis

Use these figures to locate where the pipeline fails: labels, dense prediction, reasoning, or decoding.

Recommended figures:

- gt rows overlay vs pred rows overlay
- center_gt vs center_pred
- continuity_gt vs continuity_pred
- uncertainty_gt vs uncertainty_pred
- seeds
- raw trajectories
- final trajectories
- summary panel

Recommended tools:

- `visualize.py --mode decode`
- `visualize.py --mode summary`
- `debug_decoder_with_gt.py`

### 4. Paper figure export

Use these figures to generate concise qualitative panels suitable for slides, notes, or manuscript drafts.

Recommended figures:

- input image
- gt rows overlay
- pred rows overlay
- center_pred
- continuity_pred
- uncertainty_pred
- refined_structure
- final trajectories
- summary panel

Recommended tools:

- `visualize.py --mode summary`
- `infer.py` exported visual outputs
- `utils/visualization.py` summary-panel helpers

## Color and Display Conventions

The current implementation uses a consistent color convention.

- `center`: red heatmap
- `continuity`: green heatmap
- `refined_structure`: cyan heatmap
- `uncertainty`: magenta heatmap
- `orientation`: yellow vectors
- `gt_rows`: green line overlay
- `pred_rows`: orange line overlay in summary-style panels
- `raw_trajectories`: blue-ish overlay
- `seeds`: red seed points

These colors come from the actual helper functions in `sfr_net/utils/visualization.py`.

## Visualization Types

## Input Image

### Meaning

The original RGB input image used for label generation, prediction, or decoding.

### Purpose

- reference image for all overlays
- base layer for visual debugging
- paper qualitative figure background

### Typical use

- label inspection
- prediction inspection
- summary panel

### Current generation path

Used by almost all visualization functions in:

- `save_field_visualizations(...)`
- `save_prediction_visualization(...)`
- `make_prediction_summary_panel(...)`

## GT Rows Overlay

### Meaning

Annotated row polylines drawn on top of the input image.

### Purpose

- verify annotation quality
- verify transform consistency
- compare with predicted rows

### Typical use

- label checking
- summary panel
- failure case analysis

### Current generation path

- `draw_rows(...)`
- `save_comparison_visualization(...)`
- `make_prediction_summary_panel(...)`
- `visualize.py --mode label`

## Pred Rows Overlay

### Meaning

Decoded row trajectories drawn on top of the input image.

### Purpose

- inspect final row-level output quality
- compare with GT rows
- inspect decoder failure patterns such as early stopping or cross-row jumping

### Typical use

- training monitoring
- failure analysis
- paper qualitative examples

### Current generation path

- `draw_rows(...)`
- `save_prediction_visualization(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`
- `infer.py`
- `visualize.py --mode pred / decode / summary`

## Center GT / Center Pred

### Meaning

Soft heatmaps of row-center support.

### Purpose

- `center_gt`: inspect whether field labels align with the annotated row centerline
- `center_pred`: inspect whether the model has learned correct row support

### Typical use

- label checking
- early training diagnosis
- dense-field failure analysis

### What to look for

- the response should concentrate around row centerlines
- the field should not drift to row boundaries
- strong false positives in inter-row regions usually indicate supervision or feature issues

### Current generation path

- `overlay_center_heatmap(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`

## Continuity GT / Continuity Pred

### Meaning

Soft continuity support maps used in Core and Full.

### Purpose

- `continuity_gt`: inspect continuity supervision and band design
- `continuity_pred`: inspect whether the model has learned propagation-friendly structural support

### Typical use

- Core/Full label checking
- Core/Full training observation
- decoder gating diagnosis

### What to look for

- continuity should support row structure rather than large unrelated background areas
- weak continuity in genuine row regions can cause premature decoder stopping
- excessive continuity in ambiguous regions can cause over-propagation

### Current generation path

- `overlay_continuity_heatmap(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`

## Uncertainty GT / Uncertainty Pred

### Meaning

Reliability maps used in Full.

### Purpose

- `uncertainty_gt`: inspect heuristic pseudo-label generation
- `uncertainty_pred`: inspect whether the model learns to mark ambiguous or risky structural regions

### Typical use

- Full-stage label inspection
- Full-stage training observation
- failure case diagnosis

### What to look for

- uncertainty should highlight ambiguous, unstable, or structurally risky regions
- it should not light up the whole image uniformly
- high uncertainty near decoder mistakes can be a good sign if it suppresses bad propagation

### Current generation path

- `overlay_uncertainty_heatmap(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`

## Refined Structure

### Meaning

The structure map produced by the explicit reasoning module.

### Purpose

- inspect whether reasoning improves raw center support
- inspect how structure support changes before decoding
- compare Core and Full reasoning behavior

### Typical use

- Core/Full training observation
- reasoning-module debugging
- summary panel export

### What to look for

- refined structure should be cleaner or more decoder-friendly than raw center support
- over-smoothed structure may hide real gaps
- under-refined structure may offer little advantage over the base field

### Current generation path

- `overlay_structure_heatmap(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`

## Seeds

### Meaning

Decoder seed points selected from `center` or `refined_structure`.

### Purpose

- inspect whether seed generation is too sparse or too noisy
- inspect effect of `infer.seed_threshold` and `infer.seed_from`
- diagnose missing rows before propagation begins

### Typical use

- decoder debugging
- threshold tuning
- failure case analysis

### What to look for

- too few seeds: rows may be missed entirely
- too many seeds: duplicate or noisy trajectories may appear
- seeds should generally lie on strong row support regions

### Current generation path

- `draw_seeds(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`
- `debug_decoder_with_gt.py`

## Raw Trajectories

### Meaning

Intermediate trajectories before final post-processing.

### Purpose

- inspect propagation behavior directly
- see whether the propagator follows correct local support
- diagnose cross-row jumps, fragmentation, or over-propagation

### Typical use

- decoder debugging
- continuity/uncertainty stop analysis
- failure case diagnosis

### What to look for

- raw trajectories should broadly follow row support
- branching, jumping, or sudden stopping indicate candidate scoring or stopping-rule issues

### Current generation path

- `draw_trajectories(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`
- `visualize.py --mode decode`

## Final Trajectories

### Meaning

Post-processed decoded row trajectories.

### Purpose

- inspect the final row-level output used in evaluation and export
- compare against GT rows
- measure qualitative impact of post-processing

### Typical use

- final prediction review
- test result inspection
- paper figure generation

### What to look for

- rows should be smooth enough and structurally plausible
- duplicates should be reduced
- obvious short false tracks should be suppressed

### Current generation path

- `draw_rows(...)`
- `save_prediction_visualization(...)`
- `save_field_visualizations(...)`
- `make_prediction_summary_panel(...)`

## Summary Panel

### Meaning

A multi-view panel combining input, GT overlays, prediction fields, reasoning outputs, seeds, raw trajectories, and final trajectories.

### Purpose

- inspect the full pipeline in one figure
- create compact debug and reporting artifacts
- generate paper-style qualitative visual summaries

### Typical use

- failure case analysis
- training milestone comparison
- qualitative result export for notes or papers

### Current generation path

- `make_summary_panel(...)`
- `make_prediction_summary_panel(...)`
- `visualize.py --mode summary`
- `save_field_visualizations(...)` summary image

## Recommended Tools and Commands

## 1. Label Checking

### `debug_labels.py`

Recommended for first-pass dataset inspection.

```bash
python -m sfr_net.tools.debug_labels
```

Typical outputs include:

- input image
- row overlay
- center GT heatmap
- orientation-related views
- row raster

### `visualize.py --mode label`

Recommended when you want the project¡¯s unified visualization pipeline.

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/base.yaml \
  --mode label \
  --split train \
  --index 0 \
  --output-dir outputs/visualize_label
```

Use this to inspect:

- `center_gt`
- `continuity_gt` when enabled
- `uncertainty_gt` when enabled
- `gt rows overlay`

## 2. Prediction Visualization

### `visualize.py --mode pred`

Recommended for inspecting predicted fields and final rows for one sample.

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/full.yaml \
  --mode pred \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --output-dir outputs/visualize_pred
```

Use this to inspect:

- `center_pred`
- `continuity_pred`
- `uncertainty_pred`
- `refined_structure`
- final row overlays

## 3. Decoder Debugging

### `debug_decoder_with_gt.py`

Recommended for isolating decoder behavior from model prediction quality.

```bash
python -m sfr_net.tools.debug_decoder_with_gt
```

This tool directly feeds GT fields into the decoder and helps answer:

- are seeds reasonable?
- does propagation follow the correct local direction?
- does the decoder fail even with perfect fields?

### `visualize.py --mode decode`

Recommended for model-output-driven decoder debugging.

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/full.yaml \
  --mode decode \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --output-dir outputs/visualize_decode
```

Use this to inspect:

- seeds
- raw trajectories
- final trajectories
- GT vs prediction context in one panel

## 4. Summary Export

### `visualize.py --mode summary`

Recommended for one-stop qualitative export.

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/full.yaml \
  --mode summary \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --output-dir outputs/visualize_summary
```

This is the best default choice when preparing report figures or manuscript drafts.

## 5. Inference Export

### `infer.py`

Recommended for batch qualitative export over real images.

```bash
python -m sfr_net.tools.infer \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt \
  --input data/images \
  --output-dir outputs/infer_batch
```

This exports per-image visualizations such as:

- field maps
- row overlays
- summary images
- decoded row JSON

## Recommended Visualization Review Workflow

A practical review order is:

1. check labels first
2. inspect predicted fields
3. inspect seeds and raw trajectories
4. inspect final trajectories
5. inspect one summary panel for full context

Recommended script order:

1. `debug_labels.py`
2. `debug_decoder_with_gt.py`
3. `visualize.py --mode pred`
4. `visualize.py --mode decode`
5. `visualize.py --mode summary`
6. `infer.py` for larger image batches

## What to Check in Common Failure Cases

## Missing rows

Inspect:

- `center_pred`
- `refined_structure`
- `seeds`

Typical causes:

- weak support field
- seed threshold too high
- insufficient seed coverage

## Cross-row jumps

Inspect:

- `orientation_pred`
- `continuity_pred`
- `uncertainty_pred`
- `raw trajectories`

Typical causes:

- ambiguous orientation region
- continuity too permissive
- uncertainty penalty too weak

## Premature stopping

Inspect:

- `continuity_pred`
- `uncertainty_pred`
- `raw trajectories`
- `final trajectories`

Typical causes:

- continuity too low in valid row regions
- uncertainty too high in safe regions
- stopping thresholds too strict

## Overly noisy decoding

Inspect:

- `center_pred`
- `seeds`
- `raw trajectories`
- `final trajectories`

Typical causes:

- low seed threshold
- weak candidate filtering
- insufficient short-row pruning

## Practical Notes

- Use label visualizations before training on a new dataset.
- Use decode visualizations before trusting row-level metrics.
- For paper figures, prefer summary-panel based export over manually assembling many separate files.
- Keep one or two fixed representative samples for repeated qualitative comparison across Base / Core / Full.
- When comparing variants, generate figures with the same sample index and output style.

## Summary

The current SFR-Net visualization stack already supports the full qualitative path from labels to decoded rows:

- raw image
- GT rows
- predicted fields
- reasoning output
- seeds
- raw trajectories
- final trajectories
- summary panel

These views are sufficient for day-to-day debugging, structured failure analysis, and compact qualitative result export under the current Base / Core / Full implementation.
