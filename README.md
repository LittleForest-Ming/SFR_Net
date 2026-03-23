# SFR-Net

SFR-Net is a unified single-project implementation for crop-row structural perception. The repository keeps **Base**, **Core**, and **Full** stages in one engineering stack so that dataset IO, configuration, model interfaces, losses, decoding, evaluation, and export tools stay consistent across experiments.

## Project Overview

The project is organized around one main model entrypoint, one configuration system, and one set of engine/tool scripts:

- `SFRNet.forward(images, targets=None, mode="train")`
- `SFRCriterion.forward(outputs, targets)`
- `StructuralDecoder.decode(fields, refined=None, meta=None)`

This design keeps Base/Core/Full comparable without forking the training or inference code.

## Method Summary

SFR-Net predicts dense structural fields and then decodes them into crop-row trajectories.

- `C`: center field, a soft response map around row centerlines
- `O`: orientation field, a 2-channel unit vector field defined on a narrow band
- `K`: continuity field, used in Core/Full to encourage structurally connected rows
- `U`: uncertainty field, used in Full to suppress unreliable structural regions

The pipeline is:

1. image -> backbone / neck / context
2. dense field heads -> `center`, `orientation`, optional `continuity`, optional `uncertainty`
3. optional explicit reasoning -> `refined["structure"]`
4. structural decoder -> seeds, raw trajectories, final decoded rows

Stage differences:

- **Base**: `C + O + base decoder`
- **Core**: `C + O + K + explicit reasoning + continuity-guided decoding`
- **Full**: `C + O + K + U + uncertainty-aware reasoning + uncertainty-aware decoding`

## Repository Structure

```text
SFR_Net/
├─ README.md
├─ main.py
├─ data/
├─ docs/
├─ outputs/
└─ sfr_net/
   ├─ configs/
   │  ├─ default.yaml
   │  ├─ base.yaml
   │  ├─ core.yaml
   │  ├─ full.yaml
   │  └─ schema.md
   ├─ datasets/
   │  ├─ io.py
   │  ├─ transforms.py
   │  ├─ field_label_generator.py
   │  ├─ crop_row_dataset.py
   │  └─ collate.py
   ├─ models/
   │  ├─ sfr_net.py
   │  ├─ backbones/
   │  ├─ necks/
   │  ├─ context/
   │  ├─ heads/
   │  ├─ reasoning/
   │  └─ decoder/
   ├─ losses/
   │  ├─ center_loss.py
   │  ├─ orientation_loss.py
   │  ├─ continuity_loss.py
   │  ├─ uncertainty_loss.py
   │  ├─ structure_loss.py
   │  └─ criterion.py
   ├─ metrics/
   │  ├─ pixel_metrics.py
   │  ├─ continuity_metrics.py
   │  ├─ uncertainty_metrics.py
   │  └─ row_metrics.py
   ├─ engine/
   │  ├─ checkpoint.py
   │  ├─ trainer.py
   │  ├─ evaluator.py
   │  └─ inferencer.py
   ├─ tools/
   │  ├─ train.py
   │  ├─ test.py
   │  ├─ infer.py
   │  ├─ visualize.py
   │  ├─ debug_labels.py
   │  ├─ debug_dataset.py
   │  ├─ debug_decoder_with_gt.py
   │  ├─ collect_results.py
   │  └─ make_ablation_table.py
   └─ utils/
      ├─ config.py
      ├─ geometry.py
      ├─ logger.py
      ├─ seed.py
      └─ visualization.py
```

## Installation

### 1. Create an environment

```bash
conda create -n sfrnet python=3.9 -y
conda activate sfrnet
```

### 2. Install dependencies

Install PyTorch following the official instructions for your CUDA / CPU environment, then install the Python packages used by this repository:

```bash
pip install torch torchvision
pip install numpy pillow pyyaml
```

If you use only CPU, install the CPU build of PyTorch instead.

## Dataset Format

By default the project expects:

- `dataset.root = ./data`
- `dataset.train_split = train`
- `dataset.val_split = val`
- `dataset.test_split = test`

That means the loader will look for:

- `data/train.json`
- `data/val.json`
- `data/test.json`

Each split file should contain dataset samples that can be turned into:

- an input image path
- row polyline annotations

A minimal sample layout for local debugging is already included under `data/`.

Typical image layout:

```text
data/
├─ train.json
├─ val.json
├─ test.json
└─ images/
   ├─ sample_001.png
   ├─ ...
```

At runtime, `CropRowDataset` returns a unified sample protocol:

- `image`
- `targets.center`
- `targets.orientation`
- `targets.continuity`
- `targets.uncertainty`
- `targets.valid_masks`
- `targets.aux`
- `meta`

For dataset details, see:

- `docs/data_format.md`
- `sfr_net/datasets/crop_row_dataset.py`
- `sfr_net/datasets/field_label_generator.py`

## Configuration System

The repository uses a layered YAML configuration system:

- `sfr_net/configs/default.yaml`: global defaults and full schema
- `sfr_net/configs/base.yaml`: Base-stage override
- `sfr_net/configs/core.yaml`: Core-stage override
- `sfr_net/configs/full.yaml`: Full-stage override

Configuration loading is handled by `sfr_net/utils/config.py`.

All tool scripts support:

- `--config path/to/config.yaml`
- `--set key=value key2=value2 ...`

Example:

```bash
python -m sfr_net.tools.train \
  --config sfr_net/configs/full.yaml \
  --set train.batch_size=4 train.epochs=50 infer.seed_threshold=0.40
```

Useful default config fields include:

- `project.output_dir`
- `dataset.root`
- `dataset.labels.*`
- `model.use_continuity`
- `model.use_uncertainty`
- `model.use_reasoning`
- `loss.lambda_*`
- `train.*`
- `infer.*`
- `eval.metrics`

## Training

### Main training entry

The main training entrypoints are:

- `python main.py`
- `python -m sfr_net.tools.train`

Both use the same training stack.

### Base

```bash
python -m sfr_net.tools.train --config sfr_net/configs/base.yaml
```

### Core

```bash
python -m sfr_net.tools.train --config sfr_net/configs/core.yaml
```

### Full

```bash
python -m sfr_net.tools.train --config sfr_net/configs/full.yaml
```

### Resume training

```bash
python -m sfr_net.tools.train \
  --config sfr_net/configs/full.yaml \
  --resume outputs/last.pt
```

Training uses the existing unified engine components:

- `sfr_net/engine/trainer.py`
- `sfr_net/engine/checkpoint.py`
- `sfr_net/losses/criterion.py`

## Testing

Use `sfr_net/tools/test.py` to evaluate a checkpoint on the configured test split.

### Base

```bash
python -m sfr_net.tools.test \
  --config sfr_net/configs/base.yaml \
  --checkpoint outputs/best.pt
```

### Core

```bash
python -m sfr_net.tools.test \
  --config sfr_net/configs/core.yaml \
  --checkpoint outputs/best.pt
```

### Full

```bash
python -m sfr_net.tools.test \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt
```

Optional output directory override:

```bash
python -m sfr_net.tools.test \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt \
  --output-dir outputs/test_full
```

## Inference

Use `sfr_net/tools/infer.py` for single-image or directory inference.

### Single image

```bash
python -m sfr_net.tools.infer \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt \
  --input data/images/sample_001.png \
  --output-dir outputs/infer_single
```

### Directory batch inference

```bash
python -m sfr_net.tools.infer \
  --config sfr_net/configs/full.yaml \
  --checkpoint outputs/best.pt \
  --input data/images \
  --output-dir outputs/infer_batch
```

Optional flags:

- `--limit N`
- `--no-save-fields`

The script delegates actual export work to `sfr_net/engine/inferencer.py`.

## Visualization

Use `sfr_net/tools/visualize.py` for label, prediction, decode, and summary-panel visualization.

### Label visualization

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/base.yaml \
  --mode label \
  --split train \
  --index 0 \
  --output-dir outputs/visualize_label
```

### Prediction visualization

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/full.yaml \
  --mode pred \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --output-dir outputs/visualize_pred
```

### Decoder visualization

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/full.yaml \
  --mode decode \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --output-dir outputs/visualize_decode
```

### Summary panel

```bash
python -m sfr_net.tools.visualize \
  --config sfr_net/configs/full.yaml \
  --mode summary \
  --checkpoint outputs/best.pt \
  --split test \
  --index 0 \
  --output-dir outputs/visualize_summary
```

You can also visualize an external image directly with `--input path/to/image.png` for `pred`, `decode`, and `summary` modes.

### Debug tools

The repository also includes lightweight debugging tools:

```bash
python -m sfr_net.tools.debug_labels
python -m sfr_net.tools.debug_dataset
python -m sfr_net.tools.debug_decoder_with_gt
```

## Reasoning Variants

The project supports multiple reasoning implementations under the same unified model and output interface.

The default reasoning path is:

- `model.reasoning_mode: explicit`

This uses the current explicit iterative refinement module.

You can also switch to a lightweight transformer-based refinement path through config:

```yaml
model:
  use_reasoning: true
  reasoning_mode: transformer
```

Both reasoning variants:

- run inside the same single-project SFR-Net codebase
- keep the same `SFRNet.forward(...)` contract
- produce `refined["structure"]` for downstream decoder usage
- remain compatible with Base / Core / Full mode control through cfg

The transformer variant should be understood as a lightweight refinement alternative, not as a separate main model or a graph-based reasoning system.
## Base / Core / Full Modes

### Base

Config: `sfr_net/configs/base.yaml`

Enabled components:

- `center`
- `orientation`
- base decoder

Typical behavior:

- `continuity = None`
- `uncertainty = None`
- `refined.structure = None`

### Core

Config: `sfr_net/configs/core.yaml`

Enabled components:

- `center`
- `orientation`
- `continuity`
- explicit reasoning
- continuity-guided decoding

Typical behavior:

- `uncertainty = None`
- `refined.structure` available when reasoning is enabled

### Full

Config: `sfr_net/configs/full.yaml`

Enabled components:

- `center`
- `orientation`
- `continuity`
- `uncertainty`
- uncertainty-aware reasoning
- uncertainty-aware decoding

Typical behavior:

- `refined.structure` available
- decoder uses `refined_structure + continuity + uncertainty`

## Output Files

By default outputs are written under `project.output_dir` from the active config. The default in `default.yaml` is:

```yaml
project:
  output_dir: ./outputs
```

Common files and directories produced by the current tools include:

### Training

- `train_history.json`
- `last.pt`
- `best.pt`
- `vis/epoch_XXX.png`
- `config.json`

### Testing

- `summary.json`
- `samples.json`
- `samples.csv`

### Inference

Per image directory may include:

- `*_overlay.png`
- `*_decoded_rows.png`
- `*_rows.json`
- `*_center.png`
- `*_continuity.png`
- `*_uncertainty.png`
- `*_structure.png`
- `*_orientation.png`
- `*_summary.png`
- optional `*.npy` field dumps when `infer.save_field_arrays=true`

### Visualization

Depending on mode:

- `*_label_summary.png`
- `*_pred_summary.png`
- `*_decode_panel.png`
- `*_summary_panel.png`
- row overlays and field maps

## Result Collection / Ablation Tools

### Collect results from multiple experiments

```bash
python -m sfr_net.tools.collect_results \
  --root_dir outputs \
  --output_file outputs/collected_results.json
```

Optional filtering:

```bash
python -m sfr_net.tools.collect_results \
  --root_dir outputs \
  --pattern full \
  --output_file outputs/collected_results.csv
```

`collect_results.py` scans standard experiment artifacts such as:

- `results.json`
- `summary.json`
- `test_summary.json`
- `infer_summary.json`
- `metrics.csv`
- `config_dump.yaml`
- `config.json`

### Generate ablation tables

```bash
python -m sfr_net.tools.make_ablation_table \
  --input_file outputs/collected_results.json \
  --output_dir outputs/ablation_tables \
  --format both
```

Optional grouping:

```bash
python -m sfr_net.tools.make_ablation_table \
  --input_file outputs/collected_results.json \
  --output_dir outputs/ablation_tables \
  --format markdown \
  --group_by mode
```

Generated tables include:

- structure ablation
- loss ablation
- parameter sensitivity

## Additional Documents

See `docs/` for project-specific notes and staged design summaries:

- `docs/interface_spec.md`
- `docs/first_version_tasks.md`
- `docs/core_stage_tasks.md`
- `docs/full_stage_tasks.md`
- `docs/method_overview.md`
- `docs/config_guide.md`
- `docs/data_format.md`
- `docs/ablation_plan.md`
- `docs/visualization_guide.md`


