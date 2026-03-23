# SFR-Net Data Format

## Overview

This document describes the data organization, annotation format, and split conventions used by the current SFR-Net project.

The current implementation reads image samples and row polylines, and then automatically generates training targets for:

- `center`
- `orientation`
- `continuity`
- `uncertainty`

These dense labels are **not expected as manual annotation files**. They are generated at runtime by `sfr_net/datasets/field_label_generator.py` from row polyline annotations.

## Recommended Image Data Organization

The default configuration expects:

- `dataset.root = ./data`
- `dataset.train_split = train`
- `dataset.val_split = val`
- `dataset.test_split = test`

So the recommended directory layout is:

```text
data/
й└йд train.json
й└йд val.json
й└йд test.json
й╕йд images/
   й└йд sample_001.png
   й└йд sample_002.png
   й╕йд ...
```

This means:

- `data/train.json` is the annotation file for the training split
- `data/val.json` is the annotation file for the validation split
- `data/test.json` is the annotation file for the test split

Image files are usually stored under a subdirectory such as `data/images/`, but any relative path under `dataset.root` is acceptable as long as `image_path` points to a valid file.

## Annotation JSON Format

The current loader is implemented in `sfr_net/datasets/io.py`.

It supports two equivalent top-level formats:

### Format A: top-level list

```json
[
  {
    "image_id": "sample_001",
    "image_path": "images/sample_001.png",
    "orig_size": [512, 512],
    "rows": [
      [[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]],
      [[300, 500], [310, 380], [320, 260], [330, 140], [340, 20]]
    ],
    "meta": {
      "scene_id": "scene_001",
      "camera_id": "cam_001"
    }
  }
]
```

### Format B: dict with `samples`

```json
{
  "samples": [
    {
      "image_id": "sample_001",
      "image_path": "images/sample_001.png",
      "orig_size": [512, 512],
      "rows": [
        [[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]]
      ],
      "meta": {
        "scene_id": "scene_001"
      }
    }
  ]
}
```

The loader accepts both forms and converts them into an internal list of samples.

## Required Sample Fields

At annotation level, the minimum useful fields for a training / validation / test sample are:

- `image_path`
- `rows`

Recommended additional fields are:

- `image_id`
- `orig_size`
- `meta`

### Minimum training sample

```json
{
  "image_path": "images/sample_001.png",
  "rows": [
    [[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]]
  ]
}
```

### Recommended training sample

```json
{
  "image_id": "sample_001",
  "image_path": "images/sample_001.png",
  "orig_size": [512, 512],
  "rows": [
    [[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]],
    [[300, 500], [310, 380], [320, 260], [330, 140], [340, 20]]
  ],
  "meta": {
    "scene_id": "scene_001",
    "camera_id": "cam_001"
  }
}
```

## Field Semantics

### `image_path`

`image_path` is the image file path relative to `dataset.root`.

Example:

```json
"image_path": "images/sample_001.png"
```

With `dataset.root = ./data`, the dataset will try to load:

```text
./data/images/sample_001.png
```

If `image_path` is missing or the file does not exist, the current dataset implementation falls back to a zero image of configured input size. This is useful for debugging, but real training data should always provide a valid image.

### `rows`

`rows` is a list of crop-row polylines.

Each row is:

- one polyline
- represented as a list of 2D points
- each point is `[x, y]`

Example:

```json
"rows": [
  [[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]],
  [[300, 500], [310, 380], [320, 260], [330, 140], [340, 20]]
]
```

The current loader will:

- keep `rows = []` if missing
- skip rows with fewer than 2 points
- cast coordinates to `float`

### `meta`

`meta` is optional sample-level auxiliary information.

It is not required by the current dataset loader, but is recommended for bookkeeping, visualization, and future analysis.

Example:

```json
"meta": {
  "scene_id": "scene_001",
  "camera_id": "cam_001",
  "note": "difficult lighting"
}
```

The current `CropRowDataset` does not directly copy the annotation `meta` field into the final sample output. Instead, it currently builds runtime `meta` with fields such as:

- `image_id`
- `orig_size`
- `input_size`
- `path`

So annotation-level `meta` is best treated as user-side dataset bookkeeping information.

## Row Polyline Point Order and Coordinate Convention

The project assumes each row is annotated as an ordered polyline.

### Coordinate convention

Each point uses image coordinates:

- `x`: horizontal axis, increasing from left to right
- `y`: vertical axis, increasing from top to bottom

So the point format is:

```text
[x, y]
```

not:

```text
[y, x]
```

### Point order

Points in each row should follow the physical row direction consistently.

Recommended practice:

- keep point order monotonic along the row
- do not randomly shuffle points
- use the same ordering convention across the dataset

For example, if rows are mostly visible from bottom to top, then annotate each row in bottom-to-top order:

```json
[[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]]
```

This matters because:

- the project interpolates polylines
- local tangents are computed from point order
- orientation GT depends on consistent tangent direction before unit normalization

Although the decoder is bidirectional and the orientation field is locally normalized, stable annotation order is still recommended for cleaner supervision and debugging.

## Split Files and Data Split Recommendations

The current project expects split files by name:

- `train.json`
- `val.json`
- `test.json`

under `dataset.root`.

Recommended practice:

- training split: most annotated images
- validation split: used during training model selection
- test split: held-out evaluation only

Typical layout:

```text
data/
й└йд train.json
й└йд val.json
й└йд test.json
й╕йд images/
```

### Suggested split discipline

- avoid near-duplicate images across train/val/test
- keep row density and scene difficulty reasonably balanced
- if using multiple fields or sites, try to distribute them consistently across splits
- keep the annotation format identical across all three split files

## Minimal Fields Needed for Each Split

### Training

Minimum:

- `image_path`
- `rows`

Recommended:

- `image_id`
- `orig_size`
- `meta`

### Validation

Minimum:

- `image_path`
- `rows`

Recommended:

- same structure as training samples

### Test

Minimum:

- `image_path`
- `rows`

Recommended:

- same structure as training samples
- `image_id` for easier result export and matching

## Label Generation in This Project

The current project does **not** require manually saved dense target maps for:

- `center`
- `orientation`
- `continuity`
- `uncertainty`

Instead, they are generated automatically from `rows` by `FieldLabelGenerator` in:

- `sfr_net/datasets/field_label_generator.py`

### Current behavior

From row polylines, the project generates:

- `center`: soft centerline support map
- `orientation`: 2-channel tangent field in a narrow band
- `continuity`: optional continuity support map when enabled by config
- `uncertainty`: optional heuristic pseudo-label map when enabled by config

So annotation files should provide polylines, not dense supervision maps.

## How `CropRowDataset` Uses the Annotation

The current dataset pipeline in `sfr_net/datasets/crop_row_dataset.py` works as follows:

1. load split annotation file
2. load image using `image_path`
3. read row polylines from `rows`
4. apply transforms jointly to image and row polylines
5. generate field targets from transformed polylines
6. return a unified sample dict

The runtime sample returned by the dataset has this structure:

```text
{
  "image": image_tensor,
  "targets": {
    "center": ...,
    "orientation": ...,
    "continuity": ...,
    "uncertainty": ...,
    "valid_masks": {...},
    "aux": {...}
  },
  "meta": {
    "image_id": ...,
    "orig_size": ...,
    "input_size": ...,
    "path": ...
  }
}
```

## Valid Masks and Auxiliary Fields

The dataset target package currently includes valid masks and auxiliary maps.

### `targets.valid_masks`

Current keys are:

- `center`
- `orientation`
- `continuity`
- `uncertainty`

Disabled fields remain present and may be `None`.

### `targets.aux`

Current dataset-level auxiliary content includes:

- `row_polylines`
- `row_raster`
- `inter_row_mask`
- `narrow_band_mask`

`FieldLabelGenerator` internally also creates additional debug maps such as:

- `continuity_band_mask`
- `uncertainty_source_mask`

but `CropRowDataset` does not currently expose them in `targets["aux"]`.

## Example Complete Annotation File

```json
[
  {
    "image_id": "sample_001",
    "image_path": "images/sample_001.png",
    "orig_size": [512, 512],
    "rows": [
      [[120, 500], [135, 380], [150, 260], [165, 140], [180, 20]],
      [[300, 500], [310, 380], [320, 260], [330, 140], [340, 20]]
    ],
    "meta": {
      "scene_id": "debug_scene_001",
      "camera_id": "debug_cam_001"
    }
  }
]
```

## Practical Recommendations

- keep `image_path` relative to `dataset.root`
- keep `rows` ordered and consistent
- avoid rows with only one point
- avoid mixing coordinate conventions
- keep train / val / test files structurally identical
- use `debug_labels.py` and `debug_dataset.py` to validate new datasets before training

## Summary

For the current SFR-Net implementation, the annotation source of truth is:

- image path
- ordered row polylines
- optional metadata

Dense structural targets are generated automatically inside the dataset pipeline, which keeps Base / Core / Full training compatible under one unified data format.
