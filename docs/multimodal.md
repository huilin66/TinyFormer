# Multimodal TinyFormer

`engine.multimodal.MultiModalTinyFormer` adds arbitrary-`n` multimodal fusion
without changing the original single-modal `DEIM` model or
`DINOv3SSAs_4Scale` implementation.

## Fusion modes

| Mode | Branching and fusion point |
|---|---|
| `IF` | Fuse images, then one Backbone, SSA, Neck, and Decoder |
| `BF` | Per-modality Backbone, fuse Backbone features, then one SSA, Neck, and Decoder |
| `SF` | One Backbone on the fused image, per-modality SSA, then fuse SSA outputs |
| `EF` | Per-modality Backbone + SSA, then fuse Encoder outputs |
| `NF` | Per-modality Backbone + SSA + Neck, then fuse Neck outputs |
| `DF` | Per-modality paths through Decoder input projection, fuse projected memory, then run one shared query selection, Transformer Decoder, and prediction head |
| `FF` | Complete per-modality paths and a separately configurable final fusion operator |

The model accepts any `num_modalities >= 1`. Inputs may be supplied as:

- one BCHW tensor whose channels are concatenated according to `modality_channels`;
- a list/tuple of `n` BCHW tensors;
- a dictionary keyed by `modality_names`;
- a dictionary with an `images` entry containing either of the above.

All modalities must have the same batch and spatial dimensions. No modality
name such as RGB or infrared is hard-coded.

## Configuration

Start from
`configs/tinyformer/tinyformer_dinov3_xl_coco_pbm_multimodal.yml` and change
only the multimodal block:

```yaml
MultiModalTinyFormer:
  fusion_mode: EF       # IF/BF/SF/EF/NF/DF/FF
  num_modalities: 3
  modality_channels: [3, 3, 1]
  modality_names: [visible, thermal, depth]
  share_weight: false
  fusion: {type: AddFusion, normalize: true}
  image_fusion: {type: ConcatFusion, dim: 1}
  final_fusion: {type: AddFusion, normalize: true}
```

`share_weight: true` reuses the same module object for every branch before the
selected fusion point. It is rejected when modality channel counts differ.
When it is false, every branch is a deep copy with independent parameters.

The provided YAML describes the model and the original single-image COCO
pipeline. A multimodal training dataset must emit one of the accepted input
forms above and must apply identical geometric augmentation to aligned
modalities. The model itself is independent of dataset naming and layout.

## Fusion interface

Every fusion module follows:

```python
fused = fusion(features, images=None, masks=None, metadata=None)
```

`AddFusion` and `ConcatFusion` recursively support tensors, multi-scale tuples,
decoder dictionaries, and auxiliary-output lists. `ConcatFusion` changes the
channel/query dimension, so the next stage must be configured for the enlarged
dimension. `AddFusion(normalize=true)` preserves shapes by averaging branches.

Custom fusion methods can subclass `Fusion`, override `fuse_tensors()` for a
basic tensor operator, or override `forward()` when they require images, masks,
metadata, gates, attention, or NIF auxiliary inputs. Register the class with
the existing `@register()` decorator and reference it from YAML.

## Legacy checkpoints

The original single-modal configuration remains unchanged. During tuning,
`MultiModalTinyFormer.remap_tuning_state_dict()` expands legacy
`backbone.*`, `encoder.*`, and `decoder.*` parameters into all shared or
independent branches. New channel adapters and fusion-specific parameters keep
their initialization.
