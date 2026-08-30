"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import copy
import random

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range,
                                               scale=scaling_range, fill=fill_value)
        self.use_cache = use_cache
        self.mosaic_cache = []
        # A separate cache is required for multimodal samples.  The original
        # cache stores one image at a time, so replaying Mosaic for RGB/IR/depth
        # would otherwise mix unrelated modalities into one mosaic.
        self.multimodal_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            target['boxes'] = target['boxes'] + offsets[i]
            mosaic_target.append(target)

        merged_target = {}
        for key in mosaic_target[0]:
            merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {
            key: value.clone() if hasattr(value, "clone") else copy.deepcopy(value)
            for (key, value) in tensor_dict.items()
        }

    def _resize_multimodal_sample(self, images, target):
        """Resize an aligned image bundle once while transforming its target."""

        resized_images = []
        resized_target = None
        for index, image in enumerate(images):
            if index == 0:
                resized_image, resized_target = self.resize(image, copy.deepcopy(target))
            else:
                resized_image = self.resize(image)
            resized_images.append(resized_image)
        return resized_images, resized_target

    def _load_multimodal_samples_from_dataset(self, images, target, dataset):
        """Load four aligned samples without applying their transforms twice."""

        resized_images, resized_target = self._resize_multimodal_sample(images, target)
        samples = [{"images": resized_images, "labels": resized_target}]
        sample_indices = random.choices(range(len(dataset)), k=3)
        for index in sample_indices:
            sample_images, sample_target = dataset.load_multimodal_item(index)
            sample_images, sample_target = self._resize_multimodal_sample(sample_images, sample_target)
            samples.append({"images": sample_images, "labels": sample_target})
        return samples

    def _load_multimodal_samples_from_cache(self, images, target):
        """Cache complete aligned modality bundles, never individual images."""

        resized_images, resized_target = self._resize_multimodal_sample(images, target)
        self.multimodal_cache.append(
            {
                "images": [image.copy() for image in resized_images],
                "labels": self._clone(resized_target),
            }
        )
        cache = self.multimodal_cache
        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # keep the current sample
            else:
                index = 0
            cache.pop(index)

        sample_indices = random.choices(range(len(cache)), k=3)
        samples = [
            {
                "images": [image.copy() for image in cache[index]["images"]],
                "labels": self._clone(cache[index]["labels"]),
            }
            for index in sample_indices
        ]
        samples.insert(
            0,
            {
                "images": [image.copy() for image in resized_images],
                "labels": self._clone(resized_target),
            },
        )
        return samples

    def create_multimodal_mosaic(self, samples, max_height, max_width):
        """Create one geometrically identical mosaic for every modality."""

        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        modality_count = len(samples[0]["images"])
        merged_images = [
            Image.new(
                mode=samples[0]["images"][modality].mode,
                size=(max_width * 2, max_height * 2),
                color=0,
            )
            for modality in range(modality_count)
        ]
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        mosaic_targets = []
        for index, sample in enumerate(samples):
            for modality, image in enumerate(sample["images"]):
                merged_images[modality].paste(image, placement_offsets[index])
            target = self._clone(sample["labels"])
            target["boxes"] = target["boxes"] + offsets[index]
            mosaic_targets.append(target)

        merged_target = {}
        for key in mosaic_targets[0]:
            values = [target[key] for target in mosaic_targets]
            merged_target[key] = (
                torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values
            )
        return merged_images, merged_target

    @staticmethod
    def _spatial_size(image):
        return F.get_size(image) if hasattr(F, "get_size") else F.get_spatial_size(image)

    def _apply_shared_affine(self, images, target, dataset):
        """Replay RandomAffine with exactly one sampled parameter set."""

        state_fn = getattr(dataset, "_rng_state", None)
        restore_fn = getattr(dataset, "_set_rng_state", None)
        if not callable(state_fn) or not callable(restore_fn):
            # MultiModalCocoDetection supplies these methods.  Keep a safe
            # fallback for custom datasets that use this transform directly.
            transformed = [self.affine_transform(image) for image in images]
            transformed_image, transformed_target = self.affine_transform(images[0], target)
            del transformed_image
            return transformed, transformed_target

        initial_state = state_fn()
        transformed = []
        for image in images:
            restore_fn(initial_state)
            transformed.append(self.affine_transform(image))

        # Applying the transform to the target consumes the RNG exactly once,
        # which is the state subsequent transforms should observe.
        restore_fn(initial_state)
        _, transformed_target = self.affine_transform(images[0], target)
        advanced_state = state_fn()
        restore_fn(advanced_state)
        return transformed, transformed_target

    def forward_multimodal(self, images, target, dataset):
        """Apply Mosaic to an aligned list of modality images.

        This method is called by the multimodal Compose path.  It deliberately
        keeps the sample identity and spatial transform shared across all
        modalities while maintaining the original single-image ``forward``
        implementation unchanged.
        """

        if self.probability < 1.0 and random.random() > self.probability:
            return images, target, dataset

        if self.use_cache:
            samples = self._load_multimodal_samples_from_cache(images, target)
        else:
            samples = self._load_multimodal_samples_from_dataset(images, target, dataset)

        sizes = [self._spatial_size(sample["images"][0]) for sample in samples]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)
        mosaic_images, mosaic_target = self.create_multimodal_mosaic(samples, max_height, max_width)

        if "boxes" in mosaic_target:
            mosaic_target["boxes"] = convert_to_tv_tensor(
                mosaic_target["boxes"],
                "boxes",
                box_format="xyxy",
                spatial_size=mosaic_images[0].size[::-1],
            )
        if "masks" in mosaic_target:
            mosaic_target["masks"] = convert_to_tv_tensor(mosaic_target["masks"], "masks")

        mosaic_images, mosaic_target = self._apply_shared_affine(
            mosaic_images, mosaic_target, dataset
        )
        return mosaic_images, mosaic_target, dataset

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        return mosaic_image, mosaic_target, dataset
