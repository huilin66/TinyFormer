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

    def __getitem__(self, idx):
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

        if self._transforms is not None:
            initial_state = self._rng_state()
            original_target = copy.deepcopy(target)
            primary, target, _ = self._transforms(primary, target, self)
            advanced_state = self._rng_state()
            self._set_rng_state(initial_state)
            auxiliary, _, _ = self._transforms(auxiliary, original_target, self)
            self._set_rng_state(advanced_state)

        if not torch.is_tensor(primary) or not torch.is_tensor(auxiliary):
            raise TypeError("PairedCocoDetection transforms must convert both images to tensors")
        if primary.shape[1:] != auxiliary.shape[1:]:
            raise RuntimeError(
                f"Synchronized transforms produced different shapes: "
                f"primary={tuple(primary.shape)}, auxiliary={tuple(auxiliary.shape)}"
            )
        return torch.cat((primary, auxiliary), dim=0), target

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\n auxiliary_img_folder: {self.auxiliary_img_folder}\n"


__all__ = ["PairedCocoDetection"]
