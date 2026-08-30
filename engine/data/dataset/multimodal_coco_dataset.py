"""Aligned arbitrary-modality COCO dataset for multimodal TinyFormer."""

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
class MultiModalCocoDetection(CocoDetection):
    """Load n aligned images and replay identical stochastic transforms."""

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folders,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        img_folder=None,
    ):
        # The multimodal YAML inherits coco_detection.yml, whose dataset block
        # contributes the legacy single-image ``img_folder`` key.  Workspace
        # configuration merging keeps that key alongside ``img_folders``.
        # Accept it for compatibility, but never use it as a modality source.
        del img_folder
        folders = [Path(path) for path in img_folders]
        if not folders:
            raise ValueError("img_folders must contain at least one modality folder")
        super().__init__(folders[0], ann_file, transforms, return_masks, remap_mscoco_category)
        self.img_folders = folders
        self._images_by_stem = [self._index_by_relative_stem(path) for path in folders]

    @staticmethod
    def _index_by_relative_stem(root: Path) -> dict[str, Path]:
        if not root.is_dir():
            raise FileNotFoundError(f"Modality image folder does not exist: {root}")
        index: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_file():
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
        file_name = Path(self.coco.loadImgs(self.ids[idx])[0]["file_name"])
        key = (file_name.parent / file_name.stem).as_posix()
        images = []
        for root, index in zip(self.img_folders, self._images_by_stem):
            path = index.get(key)
            if path is None:
                raise FileNotFoundError(f"No aligned image for {file_name} under {root}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        sizes = {image.size for image in images}
        if len(sizes) != 1:
            raise RuntimeError(f"Aligned image sizes differ for {file_name}: {sorted(sizes)}")
        return images

    def load_multimodal_item(self, idx):
        """Load one raw, aligned sample without invoking dataset transforms.

        Mosaic uses this hook to sample four complete modality bundles.  Using
        ``load_item`` for the extra samples would only load the first image
        folder and would silently break cross-modal alignment.
        """

        target = self.load_item(idx)[1]
        return self._load_aligned_images(idx), target

    def __getitem__(self, idx):
        images, target = self.load_multimodal_item(idx)

        if self._transforms is not None:
            multimodal_forward = getattr(self._transforms, "forward_multimodal", None)
            if callable(multimodal_forward):
                images, target, _ = multimodal_forward(images, target, self)
            else:
                # Keep compatibility with custom transform containers that do
                # not implement the multimodal protocol yet.
                initial_state = self._rng_state()
                original_target = copy.deepcopy(target)
                advanced_state = None
                transformed = []
                for index, image in enumerate(images):
                    self._set_rng_state(initial_state)
                    # Replay every modality from the same raw target. The first
                    # pass converts BoundingBoxes to a plain Tensor near the
                    # end of the pipeline, which is not a valid input to a
                    # second pass through the earlier box-aware transforms.
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
            raise TypeError("MultiModalCocoDetection transforms must convert every image to a tensor")
        shapes = {tuple(image.shape[1:]) for image in images}
        if len(shapes) != 1:
            raise RuntimeError(f"Synchronized transforms produced different shapes: {sorted(shapes)}")
        return torch.cat(images, dim=0), target

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\n img_folders: {self.img_folders}\n"


__all__ = ["MultiModalCocoDetection"]
