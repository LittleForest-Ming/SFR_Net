# SFR-Net Interface and Configuration Specification

## Design Summary

The design document defines four non-negotiable engineering rules:

1. One project only: `sfr_net/`
2. One model entrypoint only: `model = SFRNet(cfg)`
3. One configuration-driven variant system for `Base`, `Core`, and `Full`
4. One stable return protocol for dataset, model, loss, decoder, and evaluator

The model abstraction is fixed as:

`image -> structure fields {center, orientation, continuity, uncertainty} -> optional reasoning -> structured row decoding`

## Required Dataset Return Format

```python
{
    "image": Tensor,
    "targets": {
        "center": Tensor | None,
        "orientation": Tensor | None,
        "continuity": Tensor | None,
        "uncertainty": Tensor | None,
        "valid_masks": {
            "center": Tensor | None,
            "orientation": Tensor | None,
            "continuity": Tensor | None,
            "uncertainty": Tensor | None,
        },
        "aux": {
            "row_polylines": list,
            "row_raster": Tensor | None,
            "inter_row_mask": Tensor | None,
        },
    },
    "meta": {
        "image_id": str,
        "orig_size": tuple[int, int],
        "input_size": tuple[int, int],
        "path": str,
    },
}
```

## Required Model Return Format

```python
{
    "features": {"backbone": list | tuple, "fused": Tensor | None, "context": Tensor | None},
    "fields": {"center": Tensor, "orientation": Tensor, "continuity": Tensor | None, "uncertainty": Tensor | None},
    "refined": {"structure": Tensor | None, "fields": dict | None},
    "decoded": {"rows": list | None, "scores": list | None, "seeds": list | None},
    "aux": {"loss_inputs": dict, "debug": dict},
}
```

## Required Loss Return Format

```python
{
    "total": Tensor,
    "items": {
        "center": Tensor,
        "orientation": Tensor,
        "orientation_norm": Tensor,
        "continuity": Tensor,
        "structure": Tensor,
        "uncertainty": Tensor,
    },
    "stats": {
        "num_pos_center": int,
        "num_pos_orientation": int,
        "num_pos_continuity": int,
    },
}
```

## Consistency Rules

- Disabled modules return `None` or zero, never missing keys.
- `Base`, `Core`, and `Full` are controlled only by config.
- Decoder cannot hard-require `refined_structure` when reasoning is disabled.

## Current Repository Compliance Check

Before this refactor, the repository contained only a default `main.py`
template. It did not satisfy the design document in any meaningful way:

- no `sfr_net/` project package
- no unified config system
- no stable dataset/model/loss/decoder contracts
- no `Base/Core/Full` profile files
- no tools for train/test/infer/visualize
