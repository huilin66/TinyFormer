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


__all__ = ["Fusion", "AddFusion", "ConcatFusion"]
