"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


# TinyFormer receives every modality through the same DINOv3 input stem. The
# visible RGB stream can therefore use the ImageNet statistics expected by the
# pretrained backbone, but applying those statistics to infrared or
# depth-as-grayscale images introduces a modality-dependent colour bias. Keep
# the defaults here so they can be replaced through YAML without changing the
# dataset implementation.
DEFAULT_MODALITY_STATS = {
    "rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "visible": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "color": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    # IR is staged as RGB-compatible grayscale. A symmetric range keeps the
    # signal centred without pretending it has RGB colour channels.
    "infrared": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    "ir": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    "thermal": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    # Depth preprocessing maps valid values to [0, 1] and writes grayscale
    # PNG files, so a symmetric normalization is appropriate.
    "depth": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "d": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}


def _normalization_vector(value, *, name):
    """Convert a YAML scalar/list into a finite float vector."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        value = [value]
    result = [float(item) for item in value]
    if not result or any(not torch.isfinite(torch.tensor(item)) for item in result):
        raise ValueError(f"{name} must contain at least one finite number")
    return result


@register()
class ModalityNormalize(nn.Module):
    """Normalize each image in an aligned modality bundle independently.

    ``modalities`` gives the image names in bundle order. ``stats`` may be a
    mapping from modality name to ``mean``/``std`` dictionaries or a list with
    one dictionary per modality. Unknown names use ``default_mean`` and
    ``default_std`` (0.5/0.5 by default), rather than silently applying RGB
    ImageNet statistics.
    """

    def __init__(
        self,
        modalities,
        stats=None,
        default_mean=(0.5, 0.5, 0.5),
        default_std=(0.5, 0.5, 0.5),
    ):
        super().__init__()
        if isinstance(modalities, str):
            modalities = [modalities]
        if not isinstance(modalities, Sequence) or not modalities:
            raise ValueError("modalities must contain at least one modality name")
        self.modalities = tuple(str(item) for item in modalities)
        self.stats = stats if stats is not None else {}
        self.default_mean = _normalization_vector(default_mean, name="default_mean")
        self.default_std = _normalization_vector(default_std, name="default_std")
        self._stats_cache = {}
        if any(value <= 0 for value in self.default_std):
            raise ValueError("default_std values must be positive")

    @staticmethod
    def _canonical_name(value):
        return str(value).strip().lower().replace("-", "_")

    def _configured_stats(self, index):
        if index in self._stats_cache:
            return self._stats_cache[index]
        name = self._canonical_name(self.modalities[index])
        configured = None
        if isinstance(self.stats, Mapping):
            for key, value in self.stats.items():
                if self._canonical_name(key) == name:
                    configured = value
                    break
        elif isinstance(self.stats, Sequence) and not isinstance(self.stats, (str, bytes)):
            if index < len(self.stats):
                configured = self.stats[index]

        if configured is None:
            for key, value in DEFAULT_MODALITY_STATS.items():
                if self._canonical_name(key) == name:
                    configured = value
                    break
        if configured is None:
            resolved = (self.default_mean, self.default_std)
            self._stats_cache[index] = resolved
            return resolved
        if not isinstance(configured, Mapping) or "mean" not in configured or "std" not in configured:
            raise ValueError(
                f"stats for modality {self.modalities[index]!r} must contain mean and std"
            )
        mean = _normalization_vector(configured["mean"], name="mean")
        std = _normalization_vector(configured["std"], name="std")
        if any(value <= 0 for value in std):
            raise ValueError(f"std values for modality {self.modalities[index]!r} must be positive")
        resolved = (mean, std)
        self._stats_cache[index] = resolved
        return resolved

    @staticmethod
    def _normalize_image(image, mean, std, modality):
        if not torch.is_tensor(image):
            raise TypeError(
                "ModalityNormalize expects tensor input after ConvertPILImage; "
                f"got {type(image).__name__} for {modality!r}"
            )
        if image.ndim < 3:
            raise ValueError(
                f"ModalityNormalize expects a CHW image, got shape {tuple(image.shape)} "
                f"for {modality!r}"
            )
        if not image.is_floating_point():
            image = image.float()
        channels = int(image.shape[-3])
        if len(mean) == 1 and channels != 1:
            mean = mean * channels
        elif channels == 1 and len(mean) > 1:
            mean = [sum(mean) / len(mean)]
        if len(std) == 1 and channels != 1:
            std = std * channels
        elif channels == 1 and len(std) > 1:
            std = [sum(std) / len(std)]
        if len(mean) != channels or len(std) != channels:
            raise ValueError(
                f"normalization statistics for {modality!r} have {len(mean)}/{len(std)} "
                f"channels, but the image has {channels}"
            )
        shape = [1] * image.ndim
        shape[-3] = channels
        mean_tensor = torch.as_tensor(mean, dtype=image.dtype, device=image.device).reshape(shape)
        std_tensor = torch.as_tensor(std, dtype=image.dtype, device=image.device).reshape(shape)
        return (image - mean_tensor) / std_tensor

    def forward_multimodal(self, images, target, dataset):
        if len(images) != len(self.modalities):
            raise ValueError(
                f"ModalityNormalize received {len(images)} images for "
                f"{len(self.modalities)} modalities"
            )
        normalized = []
        for index, image in enumerate(images):
            mean, std = self._configured_stats(index)
            normalized.append(self._normalize_image(image, mean, std, self.modalities[index]))
        return normalized, target, dataset

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if isinstance(sample, tuple):
            image = self._normalize_image(sample[0], *self._configured_stats(0), self.modalities[0])
            return (image, *sample[1:])
        if isinstance(sample, list) and sample and torch.is_tensor(sample[0]):
            image = self._normalize_image(sample[0], *self._configured_stats(0), self.modalities[0])
            return [image, *sample[1:]]
        return self._normalize_image(sample, *self._configured_stats(0), self.modalities[0])


@register()
class ModalityPhotometricDistort(nn.Module):
    """Apply photometric augmentation only to selected modalities.

    Colour jitter is useful for RGB but invalid for depth and generally
    undesirable for infrared. ``active_modalities`` accepts names or integer
    indices. The random state is replayed for multiple active streams so
    aligned visible streams receive identical parameters, while advancing
    the state only once.
    """

    def __init__(self, modalities, active_modalities=None, p=0.5):
        super().__init__()
        if isinstance(modalities, str):
            modalities = [modalities]
        if not isinstance(modalities, Sequence) or not modalities:
            raise ValueError("modalities must contain at least one modality name")
        self.modalities = tuple(str(item) for item in modalities)
        if active_modalities is None:
            active_modalities = ["rgb"]
        if isinstance(active_modalities, (str, int)):
            active_modalities = [active_modalities]
        self.active_modalities = tuple(active_modalities)
        self.active_indices = self._resolve_indices()
        self.transform = T.RandomPhotometricDistort(p=p)

    def _resolve_indices(self):
        canonical = {
            str(name).strip().lower().replace("-", "_"): index
            for index, name in enumerate(self.modalities)
        }
        indices = []
        for value in self.active_modalities:
            if isinstance(value, int):
                index = value
            else:
                index = canonical.get(str(value).strip().lower().replace("-", "_"))
            if index is not None and 0 <= index < len(self.modalities) and index not in indices:
                indices.append(index)
        return tuple(indices)

    def forward_multimodal(self, images, target, dataset):
        if not self.active_indices:
            return list(images), target, dataset
        state_fn = getattr(dataset, "_rng_state", None)
        restore_fn = getattr(dataset, "_set_rng_state", None)
        transformed = list(images)
        if callable(state_fn) and callable(restore_fn):
            initial_state = state_fn()
            advanced_state = None
            for index in self.active_indices:
                restore_fn(initial_state)
                transformed[index] = self.transform(images[index])
                if advanced_state is None:
                    advanced_state = state_fn()
            if advanced_state is not None:
                restore_fn(advanced_state)
        else:
            for index in self.active_indices:
                transformed[index] = self.transform(images[index])
        return transformed, target, dataset

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if isinstance(sample, tuple):
            return (self.transform(sample[0]), *sample[1:])
        if isinstance(sample, list) and sample:
            return [self.transform(sample[0]), *sample[1:]]
        return self.transform(sample)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]
    

    def transform(self, inpt, params):
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt
    

    def transform(self, inpt, params):
        return self._transform(inpt, params)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
    

    def transform(self, inpt, params):
        return self._transform(inpt, params)
