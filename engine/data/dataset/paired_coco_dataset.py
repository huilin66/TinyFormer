"""Aligned two-modality COCO dataset for multimodal TinyFormer training."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...core import register
from .coco_dataset import CocoDetection


@register()
class PairedCocoDetection(CocoDetection):
    """Load aligned images and replay identical stochastic transforms on both."""

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        auxiliary_img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
    ):
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        self.auxiliary_img_folder = Path(auxiliary_img_folder)
        if not self.auxiliary_img_folder.is_dir():
            raise FileNotFoundError(f"Auxiliary image folder does not exist: {self.auxiliary_img_folder}")
        self._auxiliary_by_stem = self._index_by_relative_stem(self.auxiliary_img_folder)

    @staticmethod
    def _index_by_relative_stem(root: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            key = (path.relative_to(root).parent / path.stem).as_posix()
            if key in index:
                raise RuntimeError(f"Ambiguous aligned image stem {key!r} under {root}")
            index[key] = path
        return index

    @staticmethod
    def _rng_state():
        return random.getstate(), np.random.get_state(), torch.random.get_rng_state()

    @staticmethod
    def _set_rng_state(state) -> None:
        python_state, numpy_state, torch_state = state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    def _load_aligned_images(self, idx):
        primary, target = self.load_item(idx)
        file_name = Path(self.coco.loadImgs(self.ids[idx])[0]["file_name"])
        relative_stem = (file_name.parent / file_name.stem).as_posix()
        auxiliary_path = self._auxiliary_by_stem.get(relative_stem)
        if auxiliary_path is None:
            raise FileNotFoundError(
                f"No aligned auxiliary image for {file_name} under {self.auxiliary_img_folder}"
            )

        primary = primary.convert("RGB")
        with Image.open(auxiliary_path) as image:
            auxiliary = image.convert("RGB")
        if primary.size != auxiliary.size:
            raise RuntimeError(
                f"Aligned image sizes differ for {file_name}: primary={primary.size}, "
                f"auxiliary={auxiliary.size}"
            )
        return [primary, auxiliary], target

    def load_multimodal_item(self, idx):
        """Load one raw aligned pair for the synchronized Mosaic path."""

        return self._load_aligned_images(idx)

    def __getitem__(self, idx):
        images, target = self.load_multimodal_item(idx)

        if self._transforms is not None:
            multimodal_forward = getattr(self._transforms, "forward_multimodal", None)
            if callable(multimodal_forward):
                images, target, _ = multimodal_forward(images, target, self)
            else:
                initial_state = self._rng_state()
                original_target = copy.deepcopy(target)
                advanced_state = None
                transformed = []
                for index, image in enumerate(images):
                    self._set_rng_state(initial_state)
                    image_target = copy.deepcopy(original_target)
                    image, transformed_target, _ = self._transforms(image, image_target, self)
                    if index == 0:
                        target = transformed_target
                        advanced_state = self._rng_state()
                    transformed.append(image)
                if advanced_state is not None:
                    self._set_rng_state(advanced_state)
                images = transformed

        if any(not torch.is_tensor(image) for image in images):
            raise TypeError("PairedCocoDetection transforms must convert both images to tensors")
        if images[0].shape[1:] != images[1].shape[1:]:
            raise RuntimeError(
                f"Synchronized transforms produced different shapes: "
                f"primary={tuple(images[0].shape)}, auxiliary={tuple(images[1].shape)}"
            )
        return torch.cat(images, dim=0), target

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\n auxiliary_img_folder: {self.auxiliary_img_folder}\n"


__all__ = ["PairedCocoDetection"]
