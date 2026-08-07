"""Pluggable fusion operators for arbitrary-modality TinyFormer models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ..core import register


class Fusion(nn.Module, ABC):
    """Common recursive fusion interface.

    Subclasses only need to implement tensor fusion. Feature pyramids, decoder
    dictionaries, and auxiliary-output lists are traversed recursively.
    """

    def forward(
        self,
        features: Sequence[Any],
        images: Sequence[torch.Tensor] | None = None,
        masks: Any = None,
        metadata: Any = None,
    ) -> Any:
        del images, masks, metadata
        items = list(features)
        if not items:
            raise ValueError("fusion requires at least one modality feature")
        return self._fuse_node(items)

    def _fuse_node(self, items: list[Any]) -> Any:
        first = items[0]
        if torch.is_tensor(first):
            if not all(torch.is_tensor(item) for item in items):
                raise TypeError("all fused values must have the same container type")
            # Decoder output dictionaries contain integer/bool control tensors
            # such as dn_meta.dn_positive_idx.  They are indices, not features:
            # averaging them would cast them to float and break criterion
            # indexing.  Preserve branch-invariant metadata verbatim while
            # continuing to fuse floating-point predictions and features.
            if not (first.is_floating_point() or first.is_complex()):
                if any(
                    item.dtype != first.dtype
                    or item.shape != first.shape
                    or item.device != first.device
                    or not torch.equal(item, first)
                    for item in items[1:]
                ):
                    raise ValueError("cannot fuse unequal integer/bool tensor metadata")
                return first
            return self.fuse_tensors(items)

        if isinstance(first, Mapping):
            if not all(isinstance(item, Mapping) for item in items):
                raise TypeError("all fused values must be mappings")
            keys = list(first.keys())
            if any(set(item.keys()) != set(keys) for item in items[1:]):
                raise ValueError("all fused mappings must have identical keys")
            return type(first)((key, self._fuse_node([item[key] for item in items])) for key in keys)

        if isinstance(first, tuple):
            self._validate_sequence_shapes(items)
            return tuple(self._fuse_node([item[index] for item in items]) for index in range(len(first)))

        if isinstance(first, list):
            self._validate_sequence_shapes(items)
            return [self._fuse_node([item[index] for item in items]) for index in range(len(first))]

        # Non-tensor metadata (for example dn_meta integers) is branch-invariant.
        if not all(item == first for item in items[1:]):
            raise ValueError(f"cannot fuse unequal non-tensor values: {items!r}")
        return first

    @staticmethod
    def _validate_sequence_shapes(items: list[Any]) -> None:
        if not all(isinstance(item, (list, tuple)) for item in items):
            raise TypeError("all fused values must be sequences")
        length = len(items[0])
        if any(len(item) != length for item in items[1:]):
            raise ValueError("all fused feature sequences must have equal lengths")

    @abstractmethod
    def fuse_tensors(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse tensors at one feature level."""

    def output_channels(self, input_channels: Sequence[int]) -> int | None:
        """Return output channels when statically knowable."""

        del input_channels
        return None


@register()
class AddFusion(Fusion):
    """Elementwise addition with optional averaging."""

    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def fuse_tensors(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        reference_shape = features[0].shape
        if any(feature.shape != reference_shape for feature in features[1:]):
            raise ValueError("AddFusion requires identical tensor shapes")
        fused = torch.stack(list(features), dim=0).sum(dim=0)
        return fused / len(features) if self.normalize else fused

    def output_channels(self, input_channels: Sequence[int]) -> int:
        channels = list(input_channels)
        if not channels or any(value != channels[0] for value in channels[1:]):
            raise ValueError("AddFusion requires identical channel counts")
        return channels[0]


@register()
class ConcatFusion(Fusion):
    """Concatenate modality tensors along a configurable dimension."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def fuse_tensors(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        rank = features[0].ndim
        if any(feature.ndim != rank for feature in features[1:]):
            raise ValueError("ConcatFusion requires tensors with equal ranks")
        return torch.cat(list(features), dim=self.dim)

    def output_channels(self, input_channels: Sequence[int]) -> int | None:
        return sum(input_channels) if self.dim == 1 else None


@register()
class DetectionQueryFusion(Fusion):
    """Merge independent detector outputs without assuming query alignment.

    Prediction tensors are concatenated along the query dimension. Training
    metadata is preserved, and contrastive-denoising indices are offset to the
    concatenated branch layout so the standard criterion remains valid.
    """

    _QUERY_KEYS = {
        "pred_logits",
        "pred_boxes",
        "pred_corners",
        "ref_points",
        "teacher_corners",
        "teacher_logits",
    }

    def forward(
        self,
        features: Sequence[Any],
        images: Sequence[torch.Tensor] | None = None,
        masks: Any = None,
        metadata: Any = None,
    ) -> Any:
        del images, masks, metadata
        outputs = list(features)
        if not outputs:
            raise ValueError("detection fusion requires at least one branch output")
        if not all(isinstance(output, Mapping) for output in outputs):
            raise TypeError("DetectionQueryFusion requires decoder output mappings")
        return self._merge_mapping(outputs)

    def _merge_mapping(self, items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = list(items[0].keys())
        if any(set(item.keys()) != set(keys) for item in items[1:]):
            raise ValueError("all detection branches must expose identical output keys")
        merged = []
        for key in keys:
            values = [item[key] for item in items]
            if key == "dn_meta":
                value = self._merge_dn_meta(values)
            else:
                value = self._merge_value(key, values)
            merged.append((key, value))
        return type(items[0])(merged)

    def _merge_value(self, key: str, items: list[Any]) -> Any:
        first = items[0]
        if torch.is_tensor(first):
            if not all(torch.is_tensor(item) for item in items):
                raise TypeError(f"inconsistent detection output types for {key}")
            if key in self._QUERY_KEYS:
                reference = first.shape
                if first.ndim < 2 or any(
                    item.ndim != first.ndim
                    or item.shape[0] != reference[0]
                    or item.shape[2:] != reference[2:]
                    for item in items[1:]
                ):
                    raise ValueError(f"cannot concatenate incompatible query tensors for {key}")
                return torch.cat(items, dim=1)
            if any(
                item.dtype != first.dtype
                or item.shape != first.shape
                or item.device != first.device
                or not torch.equal(item, first)
                for item in items[1:]
            ):
                raise ValueError(f"branch metadata tensor {key!r} must be identical")
            return first
        if isinstance(first, Mapping):
            if not all(isinstance(item, Mapping) for item in items):
                raise TypeError(f"inconsistent detection output mappings for {key}")
            return self._merge_mapping(items)
        if isinstance(first, tuple):
            self._validate_sequence_shapes(items)
            return tuple(
                self._merge_value(key, [item[index] for item in items])
                for index in range(len(first))
            )
        if isinstance(first, list):
            self._validate_sequence_shapes(items)
            return [
                self._merge_value(key, [item[index] for item in items])
                for index in range(len(first))
            ]
        if not all(item == first for item in items[1:]):
            raise ValueError(f"branch metadata {key!r} must be identical")
        return first

    @staticmethod
    def _merge_dn_meta(items: list[Mapping[str, Any]]) -> dict[str, Any]:
        first = items[0]
        required = {"dn_positive_idx", "dn_num_group", "dn_num_split"}
        if any(not isinstance(item, Mapping) or not required.issubset(item) for item in items):
            raise ValueError("every FF branch must expose complete dn_meta")
        batch_size = len(first["dn_positive_idx"])
        if any(len(item["dn_positive_idx"]) != batch_size for item in items[1:]):
            raise ValueError("FF branches have inconsistent denoising batch metadata")

        combined_indices = [[] for _ in range(batch_size)]
        dn_offset = 0
        total_dn = 0
        total_queries = 0
        total_groups = 0
        for item in items:
            dn_count, query_count = (int(value) for value in item["dn_num_split"])
            for batch_index, indices in enumerate(item["dn_positive_idx"]):
                combined_indices[batch_index].append(indices + dn_offset)
            dn_offset += dn_count
            total_dn += dn_count
            total_queries += query_count
            total_groups += int(item["dn_num_group"])

        merged = dict(first)
        merged["dn_positive_idx"] = tuple(
            torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
            for parts in combined_indices
        )
        merged["dn_num_group"] = total_groups
        merged["dn_num_split"] = [total_dn, total_queries]
        return merged

    def fuse_tensors(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fallback tensor interface: concatenate detector queries."""
        return torch.cat(list(features), dim=1)


__all__ = ["Fusion", "AddFusion", "ConcatFusion", "DetectionQueryFusion"]
