# SFR-Net Method Overview

## Task Definition

SFR-Net addresses crop-row structural perception from an input image. The target is not only to classify pixels, but to recover row-wise geometric structure that can be used by downstream navigation, measurement, or scene understanding modules.

In the current project, the task can be viewed at two coupled levels:

1. **Dense field prediction**
   The model predicts spatial fields that describe where crop rows are, what local direction they follow, whether local structure is continuous, and whether a region is structurally unreliable.

2. **Row trajectory decoding**
   The predicted fields are decoded into polyline-like row trajectories through seed generation and bidirectional propagation.

This two-level design is reflected directly in the code:

- field heads live in `sfr_net/models/heads/field_heads.py`
- explicit reasoning lives in `sfr_net/models/reasoning/`
- trajectory decoding lives in `sfr_net/models/decoder/`

## Why Field Representation Instead of Direct Row Regression

The project uses a field-based representation instead of directly regressing a fixed number of rows or line parameters.

The main engineering and modeling reasons are:

- **Variable row count**
  Different images may contain different numbers of visible rows. A dense field formulation avoids fixing the number of row instances in advance.

- **Local ambiguity is common**
  Occlusion, weeds, illumination change, soil texture variation, and perspective distortion all create local uncertainty. Dense fields let the model express local evidence before committing to final row trajectories.

- **Easier supervision from row polylines**
  Existing annotations are naturally turned into center, orientation, continuity, and uncertainty supervision maps.

- **Decoding can be staged and debugged**
  The engineering stack can separately inspect labels, field predictions, reasoning outputs, seeds, raw trajectories, and final rows.

- **Better compatibility across Base / Core / Full**
  New structural cues can be added by enabling additional fields and losses through config, rather than redesigning the full output head and matching logic.

In short, the repository treats row extraction as:

`image -> structural fields -> optional refinement -> structured decoding`

rather than:

`image -> direct row parameter regression`

## Field Definitions

The current codebase uses four named structural fields.

### Center

`center` is a single-channel soft response map indicating how close each pixel is to a row centerline.

Practical meaning in the project:

- high response near annotated row centerlines
- low response away from row support
- used for both supervision and decoder seeding

In the current implementation:

- the GT is generated as a distance-decayed soft label rather than a hard binary raster
- the predicted tensor shape is `[B, 1, H, W]`
- Base / Core / Full all use it

Informally:

`center(x) ~ exp(-d(x, row)^2 / (2 * sigma^2))`

where `d(x, row)` is the distance from pixel `x` to the nearest row centerline sample.

### Orientation

`orientation` is a 2-channel local direction field.

Practical meaning in the project:

- channel 0 stores the local x-direction component
- channel 1 stores the local y-direction component
- only meaningful in a narrow band around row centerlines
- used by the decoder to propagate trajectories forward and backward

In the current implementation:

- GT tensor shape is `[2, H, W]`
- supervision is valid mainly inside an orientation band around the row
- model prediction shape is `[B, 2, H, W]`
- the prediction is normalized to unit length in the field head

Informally, if the local tangent direction is `t = (tx, ty)`, then:

`orientation(x) = t / ||t||`

for pixels near the row support.

### Continuity

`continuity` is a single-channel structural support map introduced in Core and used again in Full.

Practical meaning in the project:

- indicates whether row structure is locally continuous and safe to keep propagating
- helps the decoder avoid propagating through unsupported gaps
- also supports explicit reasoning by acting as a structural confidence signal

In the current implementation:

- GT is generated from row support with a band-limited soft structure prior
- shape is `[B, 1, H, W]` when enabled
- Base disables it
- Core and Full enable it through config

This field should be interpreted as a local “keep-going” signal for row structure, not as a semantic segmentation mask.

### Uncertainty

`uncertainty` is a single-channel reliability map introduced in Full.

Practical meaning in the project:

- high value means the local structural judgment is unreliable
- low value means the local structure is relatively trustworthy
- it is used to damp reasoning and penalize decoder propagation in risky regions

In the current implementation, uncertainty is generated as a heuristic pseudo-label using signals already available in the pipeline, including:

- inter-row proximity regions
- regions where continuity is weak but center response exists
- local orientation instability
- boundary / transition instability

Important constraints in the current project:

- uncertainty is **not** just `1 - continuity`
- uncertainty is clipped to `[0, 1]`
- Base and Core return `None` when uncertainty is disabled
- Full predicts `[B, 1, H, W]` through a sigmoid head

## Base / Core / Full Evolution

The repository evolves the method in three stages while keeping one engineering interface.

## Base

Base is the minimum working structural pipeline.

Enabled components:

- center
- orientation
- seed generation
- bidirectional trajectory propagation
- basic post-processing

Main goal:

- verify label generation
- verify stable model IO and loss protocol
- verify that `center + orientation` already support a usable decoder loop

Typical output characteristics:

- `fields.center`: available
- `fields.orientation`: available
- `fields.continuity`: `None`
- `fields.uncertainty`: `None`
- `refined.structure`: `None`

## Core

Core adds explicit structural continuity and iterative reasoning.

New components relative to Base:

- continuity label generation
- continuity prediction head
- continuity loss
- explicit local affinity and iterative structure refinement
- continuity-guided decoding

Main goal:

- make the structural representation more robust than `center + orientation` alone
- provide a refined structure map for decoder seeding and support
- reduce premature stopping and improve structural consistency

Typical output characteristics:

- `fields.continuity`: available
- `refined.structure`: available when reasoning is enabled
- decoder can seed from `refined_structure`

## Full

Full adds uncertainty awareness to both reasoning and decoding.

New components relative to Core:

- uncertainty pseudo-label generation
- uncertainty prediction head
- uncertainty loss
- uncertainty-aware affinity
- uncertainty-aware iterative refinement
- uncertainty-penalized decoding and stop rule
- uncertainty evaluation and visualization

Main goal:

- make the structural pipeline robust in locally ambiguous, weak, or conflicting regions
- provide trajectory confidence-related statistics and richer debug outputs

Typical output characteristics:

- `fields.uncertainty`: available
- `refined.structure`: uncertainty-aware refinement result
- decoder uses `structure / continuity / uncertainty` jointly

## Explicit Reasoning Module

The current reasoning implementation is intentionally lightweight and explicit. It does not use a heavy graph neural network or global optimization module.

The reasoning stage lives in:

- `sfr_net/models/reasoning/affinity.py`
- `sfr_net/models/reasoning/iterative_refiner.py`
- `sfr_net/models/reasoning/transformer_refiner.py`

### Core idea

Reasoning starts from the raw dense fields and refines a structure map that can be reused by the decoder.

Inputs:

- `center`
- `orientation`
- `continuity`
- optional `uncertainty`

Outputs:

- `refined["structure"]`
- debug states

The repository now supports multiple reasoning implementations under the same interface.

### Reasoning variants

The default reasoning path is:

- `reasoning_mode = explicit`

This uses a lightweight explicit iterative refiner based on local support, local smoothing, and fixed iteration count.

A second reasoning path can also be selected by config:

- `reasoning_mode = transformer`

This uses a lightweight transformer-based refiner that consumes the same structural fields, produces the same `refined structure`, and preserves the same downstream decoder contract.

In both cases:

- the output remains a structure map with shape `[B, 1, H, W]`
- the decoder still reads `refined["structure"]` through the same interface
- Base / Core / Full compatibility is preserved through config switches rather than a second main model

The transformer refiner should be understood as a context-aware refinement variant, not as a graph network or a strong global graph reasoning module.

### Local affinity in explicit mode

The explicit reasoning path computes a weighted local support score using:

- center support
- continuity support
- direction-related support
- structure support from the current iteration state
- optional uncertainty penalty in Full

At a high level, the code follows a form like:

`affinity = positive_support * uncertainty_gate`

where:

- `positive_support` is built from center / continuity / direction / structure terms
- `uncertainty_gate` reduces affinity in structurally unreliable regions

### Iterative refinement in explicit mode

The explicit refiner updates structure by mixing:

- current structure state
- local smoothing result
- local affinity support

In Full mode, uncertainty further damps smoothing and affinity injection.

The current implementation is deliberately simple:

- fixed number of iterations
- local convolutional smoothing
- cfg-controlled mixing weights
- optional debug history of intermediate structure maps

This keeps the explicit reasoning module inspectable and easy to maintain.

### Transformer-based refinement

The transformer reasoning path keeps the same input/output semantics, but changes the internal refinement style:

- structural fields are first projected into a compact hidden representation
- the spatial feature map is flattened into tokens
- a lightweight transformer encoder mixes token information
- the refined tokens are reshaped back into a single-channel structure map

This variant is designed as a configurable refinement alternative inside the same engineering stack. It should be described as a lightweight transformer-based refiner rather than as a graph-based structural inference module.

## Structure-Guided Decoder

The decoder converts dense fields into row trajectories.

Main code locations:

- `sfr_net/models/decoder/seed_generator.py`
- `sfr_net/models/decoder/trajectory_propagation.py`
- `sfr_net/models/decoder/postprocess.py`
- `sfr_net/models/decoder/structural_decoder.py`

### Seed generation

The decoder first produces seeds from a support map.

Depending on config, the seed source is:

- `center`
- `refined_structure`

Seed generation currently includes:

- support thresholding
- local-maximum style filtering
- NMS-like suppression

### Bidirectional propagation

From each seed, the decoder propagates in two directions.

The current trajectory propagation uses:

- local orientation direction
- a forward proposal point determined by step size
- candidate search in a configurable radius
- candidate scoring using structure or center support, continuity, direction consistency, distance feasibility, and optional uncertainty penalty

In Full mode, uncertainty affects decoding in two ways:

- high uncertainty lowers candidate score
- high uncertainty can trigger a stop rule

### Post-processing

After raw trajectories are generated, post-processing filters and merges them.

The current implementation includes lightweight engineering steps such as:

- short trajectory pruning
- simple quality filtering
- duplicate suppression / merge heuristics

The decoder is intentionally not over-engineered at this stage. It is designed to be understandable and easy to debug.

## Training Losses Overview

The unified loss entry is `sfr_net/losses/criterion.py`.

Current loss items include:

- `center`
- `orientation`
- `orientation_norm`
- `continuity`
- `structure`
- `uncertainty`

Each stage uses a subset of these through config.

### Center loss

Supervises the center field.

Role:

- learn row support distribution
- provide stable seed source and structure prior

### Orientation loss

Supervises local row direction.

Role:

- keep predicted orientation aligned with GT tangent direction
- regularize predicted vector norm through an additional norm-related term

### Continuity loss

Used in Core and Full.

Role:

- teach the model where row support should remain structurally connected
- improve refinement and propagation reliability

### Structure loss

Used when the project wants to supervise refined structural support more explicitly.

Role:

- encourage refined structure to remain useful as a decoding support map

### Uncertainty loss

Used in Full.

Role:

- supervise the uncertainty head using pseudo-label targets
- support uncertainty-aware reasoning and decoding

Current supported uncertainty loss types include:

- `l1`
- `smooth_l1`
- `bce`

### Config-controlled loss usage

Loss activation is driven by config flags and weights, for example:

- `model.use_continuity`
- `model.use_uncertainty`
- `loss.lambda_continuity`
- `loss.lambda_structure`
- `loss.lambda_uncertainty`

This is important for keeping Base / Core / Full in one criterion implementation.

## Output Format

The project uses a stable top-level model output contract.

`SFRNet.forward(...)` returns:

- `features`
- `fields`
- `refined`
- `decoded`
- `aux`

### `fields`

Raw dense predictions from model heads.

Keys always reserved by the project:

- `center`
- `orientation`
- `continuity`
- `uncertainty`

Disabled fields are kept as `None` rather than removed.

### `refined`

Reasoning-stage outputs.

Current keys:

- `structure`
- `fields`

### `decoded`

Decoder outputs.

Current keys:

- `rows`
- `scores`
- `seeds`
- `debug`

Each row trajectory is represented as a point sequence, and in the current decoder implementation typically stores metadata such as:

- `points`
- `score`
- `length`
- optional structural averages such as `avg_center`, `avg_continuity`, `avg_uncertainty`

### Dataset target format

The dataset returns a stable target dictionary with:

- `center`
- `orientation`
- `continuity`
- `uncertainty`
- `valid_masks`
- `aux`

Current dataset-level auxiliary fields currently include:

- `row_polylines`
- `row_raster`
- `inter_row_mask`
- `narrow_band_mask`

`FieldLabelGenerator` internally also produces additional debug maps such as `continuity_band_mask` and `uncertainty_source_mask`, but `CropRowDataset` does not currently expose them in `targets["aux"]`.

## Practical Design Notes

A few design decisions are important for future maintenance.

### One staged codebase

The project does not keep separate Base / Core / Full repositories. Instead, stage evolution is mostly controlled by config and optional fields.

### Stable disabled-field behavior

When a field is disabled, the project prefers:

- `None` for missing predictions or targets
- `0 tensor` for inactive loss items

rather than deleting keys. This reduces branching in engines and tools.

### Decoder and reasoning remain lightweight

The current implementation intentionally avoids writing down a heavier algorithm than what is already in code. The design is practical and debug-oriented rather than mathematically maximal.

### Method document is implementation-aligned

This document is meant to support:

- team communication
- code maintenance
- experiment bookkeeping
- future paper writing preparation

It should therefore stay aligned with the actual repository implementation, not with an aspirational future design.
