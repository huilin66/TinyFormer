"""Decoupled DINOv3 backbone and SSA stages used by multimodal TinyFormer."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.dinov3 import DinoVisionTransformer
from ..backbone.vit_tiny import VisionTransformer
from ..core import register


@register()
class DINOv3BackboneStage(nn.Module):
    """DINO feature extraction without the spatial-semantic adapter."""

    def __init__(
        self,
        name: str = "dinov3_vitb16",
        weights_path: str | None = None,
        interaction_indexes: Sequence[int] = (5, 8, 11),
        finetune: bool = True,
        embed_dim: int = 192,
        num_heads: int = 3,
    ):
        super().__init__()
        if "dinov3" in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                self.dinov3.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=list(interaction_indexes),
            )
            if weights_path is not None and os.path.exists(weights_path):
                self.dinov3._model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        self.embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = list(interaction_indexes)
        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = image.shape
        height_16, width_16 = height // 16, width // 16
        if self.interaction_indexes and not isinstance(self.dinov3, VisionTransformer):
            layers = self.dinov3.get_intermediate_layers(
                image,
                n=self.interaction_indexes,
                return_class_token=True,
            )
        else:
            layers = self.dinov3(image)

        if len(layers) == 1:
            selected = (layers[0], layers[0], layers[0])
        elif len(layers) >= 3:
            selected = (layers[0], layers[1], layers[2])
        else:
            raise RuntimeError("DINO backbone must return one or at least three feature layers")

        outputs = []
        for layer in selected:
            tokens = layer[0] if isinstance(layer, (list, tuple)) else layer
            outputs.append(
                tokens.transpose(1, 2)
                .reshape(batch, -1, height_16, width_16)
                .contiguous()
            )
        return tuple(outputs)


@register()
class SSA4ScaleStage(nn.Module):
    """Four-scale spatial-semantic adapter operating on DINO features and an image."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        image_channels: int = 3,
        use_adapter: bool = True,
        conv_inplane: int = 128,
    ):
        super().__init__()
        self.use_sda = use_adapter
        if use_adapter:
            self.sda = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(image_channels, conv_inplane, 3, 2, 1, bias=False),
                    nn.SyncBatchNorm(conv_inplane),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(conv_inplane, conv_inplane, 3, 2, 1, bias=False),
                    nn.SyncBatchNorm(conv_inplane),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(conv_inplane, 2 * conv_inplane, 3, 2, 1, bias=False),
                    nn.SyncBatchNorm(2 * conv_inplane),
                    nn.GELU(),
                ),
            )
            c1_dim = conv_inplane
            sda_dim = 2 * conv_inplane
        else:
            c1_dim = embed_dim
            sda_dim = 0

        self.proj_c1 = nn.Sequential(
            nn.Conv2d(c1_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(sda_dim + embed_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj_c3 = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
        )
        self.proj_c4 = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
        )

    def forward(
        self,
        backbone_features: Sequence[torch.Tensor],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(backbone_features) != 3:
            raise ValueError("SSA4ScaleStage expects three DINO feature maps")
        feat0, feat1, feat2 = backbone_features

        if self.use_sda:
            scale_2 = self.sda[0](image)
            scale_4 = self.sda[1](scale_2)
            scale_8 = self.sda[2](scale_4)
            c1 = self.proj_c1(scale_4)
            target_size = scale_8.shape[-2:]
        else:
            scale_8 = None
            c1 = self.proj_c1(F.interpolate(feat0, scale_factor=4.0, mode="bilinear"))
            target_size = (image.shape[-2] // 8, image.shape[-1] // 8)

        feat0_up = F.interpolate(feat0, size=target_size, mode="bilinear", align_corners=False)
        c2_input = torch.cat([scale_8, feat0_up], dim=1) if scale_8 is not None else feat0_up
        c2 = self.proj_c2(c2_input)
        c3 = self.proj_c3(feat1)
        c4 = self.proj_c4(F.interpolate(feat2, scale_factor=0.5, mode="bilinear", align_corners=False))
        return c1, c2, c3, c4


__all__ = ["DINOv3BackboneStage", "SSA4ScaleStage"]
