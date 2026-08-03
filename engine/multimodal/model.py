"""Arbitrary-modality, stage-selectable fusion framework for TinyFormer."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ..core import register
from .fusion import Fusion


FUSION_MODES = ("IF", "BF", "SF", "EF", "NF", "DF", "FF")


class ChannelAdapter(nn.Module):
    """Adapt a modality or fused image to the three channels expected by DINO."""

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("image channel counts must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.projection = nn.Identity()
        elif in_channels == 1 and out_channels > 1:
            self.projection = None
        else:
            projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels % out_channels == 0:
                groups = in_channels // out_channels
                with torch.no_grad():
                    projection.weight.zero_()
                    for output_index in range(out_channels):
                        for group_index in range(groups):
                            projection.weight[output_index, output_index + group_index * out_channels, 0, 0] = 1.0 / groups
            self.projection = projection

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(
                f"expected BCHW image with {self.in_channels} channels, got {tuple(image.shape)}"
            )
        if self.projection is None:
            return image.repeat(1, self.out_channels, 1, 1)
        return self.projection(image)


class BranchBank(nn.Module):
    """Hold either one shared module or independent deep-copied branches."""

    def __init__(self, template: nn.Module, count: int, share_weight: bool):
        super().__init__()
        self.count = count
        self.share_weight = bool(share_weight and count > 1)
        if self.share_weight or count == 1:
            self.shared = template
            self.branches = None
        else:
            self.shared = None
            self.branches = nn.ModuleList([template, *[copy.deepcopy(template) for _ in range(count - 1)]])

    def at(self, index: int) -> nn.Module:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.shared if self.shared is not None else self.branches[index]

    def state_prefixes(self, name: str) -> list[str]:
        if self.shared is not None:
            return [f"{name}.shared."]
        return [f"{name}.branches.{index}." for index in range(self.count)]


@register()
class MultiModalTinyFormer(nn.Module):
    """Compose TinyFormer stages with one of seven multimodal fusion modes."""

    __inject__ = [
        "backbone",
        "ssa",
        "neck",
        "decoder",
        "fusion",
        "image_fusion",
        "final_fusion",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        ssa: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        fusion: Fusion,
        image_fusion: Fusion | None = None,
        final_fusion: Fusion | None = None,
        fusion_mode: str = "EF",
        num_modalities: int = 2,
        modality_channels: Sequence[int] | None = None,
        modality_names: Sequence[str] | None = None,
        share_weight: bool = False,
        model_input_channels: int = 3,
        fused_image_channels: int | None = None,
    ):
        super().__init__()
        mode = fusion_mode.upper()
        if mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}")
        if num_modalities < 1:
            raise ValueError("num_modalities must be at least one")

        channels = list(modality_channels or [model_input_channels] * num_modalities)
        if len(channels) != num_modalities or any(value < 1 for value in channels):
            raise ValueError("modality_channels must contain one positive value per modality")
        names = list(modality_names) if modality_names is not None else None
        if names is not None and (len(names) != num_modalities or len(set(names)) != len(names)):
            raise ValueError("modality_names must contain one unique name per modality")
        if share_weight and mode != "IF" and num_modalities > 1 and len(set(channels)) != 1:
            raise ValueError(
                "share_weight=true requires identical modality input channels before fusion"
            )

        self.fusion_mode = mode
        self.num_modalities = num_modalities
        self.modality_channels = channels
        self.input_channels = sum(channels)
        self.modality_names = names
        self.share_weight = share_weight
        self.model_input_channels = model_input_channels
        self.fusion = fusion
        self.image_fusion = image_fusion or fusion
        self.final_fusion = final_fusion or fusion

        branch_backbone = mode in {"BF", "EF", "NF", "DF", "FF"}
        branch_ssa = mode in {"SF", "EF", "NF", "DF", "FF"}
        branch_neck = mode in {"NF", "DF", "FF"}
        branch_decoder = mode in {"DF", "FF"}
        self.backbones = BranchBank(backbone, num_modalities if branch_backbone else 1, share_weight)
        self.ssas = BranchBank(ssa, num_modalities if branch_ssa else 1, share_weight)
        self.necks = BranchBank(neck, num_modalities if branch_neck else 1, share_weight)
        self.decoders = BranchBank(decoder, num_modalities if branch_decoder else 1, share_weight)

        adapter_count = num_modalities if mode != "IF" else 1
        if adapter_count > 1 and share_weight:
            adapter_template = ChannelAdapter(channels[0], model_input_channels)
            self.modality_adapters = BranchBank(adapter_template, adapter_count, True)
        elif adapter_count > 1:
            adapters = [ChannelAdapter(value, model_input_channels) for value in channels]
            self.modality_adapters = nn.ModuleList(adapters)
        else:
            self.modality_adapters = None

        if mode in {"IF", "BF", "SF"}:
            inferred_channels = self.image_fusion.output_channels(channels)
            if fused_image_channels is None:
                fused_image_channels = inferred_channels
            if fused_image_channels is None:
                raise ValueError(
                    "custom image fusion must define output_channels() or set fused_image_channels"
                )
            self.fused_image_adapter = ChannelAdapter(fused_image_channels, model_input_channels)
        else:
            self.fused_image_adapter = None

    def _split_modalities(self, inputs: Any) -> list[torch.Tensor]:
        if torch.is_tensor(inputs):
            expected_channels = sum(self.modality_channels)
            if inputs.ndim != 4 or inputs.shape[1] != expected_channels:
                raise ValueError(
                    f"expected concatenated BCHW input with {expected_channels} channels, "
                    f"got {tuple(inputs.shape)}"
                )
            return list(torch.split(inputs, self.modality_channels, dim=1))

        if isinstance(inputs, Mapping):
            if "images" in inputs:
                return self._split_modalities(inputs["images"])
            if self.modality_names is None:
                raise ValueError("dictionary inputs require modality_names or an 'images' entry")
            missing = [name for name in self.modality_names if name not in inputs]
            if missing:
                raise ValueError(f"missing modalities: {missing}")
            return self._validate_image_list([inputs[name] for name in self.modality_names])

        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
            return self._validate_image_list(list(inputs))
        raise TypeError("multimodal input must be a concatenated tensor, sequence, or mapping")

    def _validate_image_list(self, images: list[Any]) -> list[torch.Tensor]:
        if len(images) != self.num_modalities:
            raise ValueError(f"expected {self.num_modalities} modality images, got {len(images)}")
        for index, (image, channels) in enumerate(zip(images, self.modality_channels)):
            if not torch.is_tensor(image) or image.ndim != 4 or image.shape[1] != channels:
                shape = tuple(image.shape) if torch.is_tensor(image) else type(image).__name__
                raise ValueError(f"modality {index} must be BCHW with {channels} channels, got {shape}")
        spatial_shapes = {tuple(image.shape[0:1] + image.shape[2:]) for image in images}
        if len(spatial_shapes) != 1:
            raise ValueError("all modality images must share batch and spatial dimensions")
        return images

    def _adapt_modalities(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.modality_adapters is None:
            return images
        if isinstance(self.modality_adapters, BranchBank):
            return [self.modality_adapters.at(index)(image) for index, image in enumerate(images)]
        return [adapter(image) for adapter, image in zip(self.modality_adapters, images)]

    def _fuse(
        self,
        features: Sequence[Any],
        images: Sequence[torch.Tensor],
        masks: Any,
        metadata: Any,
        operator: Fusion | None = None,
    ) -> Any:
        return (operator or self.fusion)(features, images=images, masks=masks, metadata=metadata)

    def _fuse_image(
        self,
        raw_images: list[torch.Tensor],
        masks: Any,
        metadata: Any,
    ) -> torch.Tensor:
        fused = self.image_fusion(raw_images, images=raw_images, masks=masks, metadata=metadata)
        if not torch.is_tensor(fused):
            raise TypeError("image fusion must return a tensor")
        return self.fused_image_adapter(fused)

    def _encode_branch(self, index: int, image: torch.Tensor) -> Any:
        backbone_features = self.backbones.at(index)(image)
        return self.ssas.at(index)(backbone_features, image)

    def forward(
        self,
        inputs: Any,
        targets: Any = None,
        masks: Any = None,
        metadata: Any = None,
    ) -> Any:
        raw_images = self._split_modalities(inputs)
        branch_images = self._adapt_modalities(raw_images)
        mode = self.fusion_mode

        if mode == "IF":
            image = self._fuse_image(raw_images, masks, metadata)
            features = self.ssas.at(0)(self.backbones.at(0)(image), image)
            return self.decoders.at(0)(self.necks.at(0)(features), targets)

        if mode == "BF":
            backbone_features = [
                self.backbones.at(index)(image) for index, image in enumerate(branch_images)
            ]
            fused_backbone = self._fuse(backbone_features, branch_images, masks, metadata)
            fused_image = self._fuse_image(raw_images, masks, metadata)
            features = self.ssas.at(0)(fused_backbone, fused_image)
            return self.decoders.at(0)(self.necks.at(0)(features), targets)

        if mode == "SF":
            fused_image = self._fuse_image(raw_images, masks, metadata)
            backbone_features = self.backbones.at(0)(fused_image)
            ssa_features = [
                self.ssas.at(index)(backbone_features, image)
                for index, image in enumerate(branch_images)
            ]
            fused_ssa = self._fuse(ssa_features, branch_images, masks, metadata)
            return self.decoders.at(0)(self.necks.at(0)(fused_ssa), targets)

        encoder_features = [
            self._encode_branch(index, image) for index, image in enumerate(branch_images)
        ]
        if mode == "EF":
            fused_encoder = self._fuse(encoder_features, branch_images, masks, metadata)
            return self.decoders.at(0)(self.necks.at(0)(fused_encoder), targets)

        neck_features = [
            self.necks.at(index)(features) for index, features in enumerate(encoder_features)
        ]
        if mode == "NF":
            fused_neck = self._fuse(neck_features, branch_images, masks, metadata)
            return self.decoders.at(0)(fused_neck, targets)

        decoder_outputs = [
            self.decoders.at(index)(features, targets)
            for index, features in enumerate(neck_features)
        ]
        operator = self.final_fusion if mode == "FF" else self.fusion
        return self._fuse(decoder_outputs, branch_images, masks, metadata, operator=operator)

    def remap_tuning_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        """Map a legacy single-modal TinyFormer checkpoint into all branches."""

        remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
        for original_key, value in state_dict.items():
            key = original_key[7:] if original_key.startswith("module.") else original_key
            destinations: list[str] = []
            if key.startswith("backbone.dinov3."):
                suffix = key[len("backbone.") :]
                destinations = [prefix + suffix for prefix in self.backbones.state_prefixes("backbones")]
            elif key.startswith("backbone."):
                suffix = key[len("backbone.") :]
                destinations = [prefix + suffix for prefix in self.ssas.state_prefixes("ssas")]
            elif key.startswith("encoder."):
                suffix = key[len("encoder.") :]
                destinations = [prefix + suffix for prefix in self.necks.state_prefixes("necks")]
            elif key.startswith("decoder."):
                suffix = key[len("decoder.") :]
                destinations = [prefix + suffix for prefix in self.decoders.state_prefixes("decoders")]
            for destination in destinations:
                remapped[destination] = value
        return remapped

    def deploy(self):
        self.eval()
        for module in self.modules():
            if module is not self and hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self


__all__ = ["FUSION_MODES", "MultiModalTinyFormer"]
