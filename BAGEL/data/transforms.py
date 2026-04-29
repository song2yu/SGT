# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from PIL import Image

import cv2
import numpy as np
import torch
# [REMOVED] All torchvision imports have been dropped; we rely on Pillow,
# NumPy and native PyTorch only.
# from torchvision import transforms
# from torchvision.transforms import functional as F
# from torchvision.transforms import InterpolationMode


class MaxLongEdgeMinShortEdgeResize(object):
    """Pure-Pillow resize that does not require torchvision.

    Resize the input image so that both the longest edge and the shortest
    edge stay within a configured range, while also making sure both
    dimensions are divisible by a given stride.

    Args:
        max_size (int): maximum length of the longest edge.
        min_size (int): minimum length of the shortest edge.
        stride (int): output height/width must be divisible by this value.
        max_pixels (int): cap on the total pixel count of the full image.
        interpolation (int): a Pillow interpolation constant, e.g.
            ``PIL.Image.BICUBIC``.
    """
    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=Image.BICUBIC  # [CHANGED] use a Pillow constant directly
    ):
        # [CHANGED] no longer inherits from torch.nn.Module
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        # [REMOVED] ``antialias`` is not a direct Pillow resize argument.

    def _make_divisible(self, value, stride):
        """Round ``value`` to the nearest multiple of ``stride``."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def __call__(self, img, img_num=1):
        """
        [CHANGED] ``forward`` has been renamed to ``__call__``.

        Args:
            img (PIL Image): the image to resize.
            img_num (int): number of images, used to adjust the max-pixel cap.
        Returns:
            PIL Image: a resized image whose dimensions are divisible by ``stride``.
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input img must be a PIL Image, but got {type(img)}")

        width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Make sure the pixel count stays within max_pixels.
        if new_width * new_height > self.max_pixels / img_num:
            scale = (self.max_pixels / img_num / (width * height)) ** 0.5
            new_width, new_height = self._apply_scale(width, height, scale)

        # Make sure the longest edge does not exceed max_size.
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # [CHANGED] use Pillow's resize instead of torchvision's F.resize.
        return img.resize((new_width, new_height), resample=self.interpolation)


class ImageTransform:
    """Image transform pipeline implemented on top of Pillow / NumPy / PyTorch."""
    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
            interpolation=Image.BICUBIC
        )

        # [CHANGED] Pre-compute mean / std as tensors for faster normalization.
        self.image_mean = torch.tensor(image_mean).view(3, 1, 1)
        self.image_std = torch.tensor(image_std).view(3, 1, 1)

    def __call__(self, img, img_num=1):
        # 1. Resize: both input and output are PIL Image.
        img = self.resize_transform(img, img_num=img_num)

        # 2. ToTensor: implemented manually.
        # PIL Image (H, W, C) -> NumPy array -> (C, H, W) Tensor.
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))

        # Cast to float and scale into [0, 1].
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0

        # 3. Normalize: implemented manually.
        # [CHANGED] Use the pre-computed mean / std.
        # In-place normalization is expressed via sub_/div_.
        img_tensor.sub_(self.image_mean).div_(self.image_std)

        return img_tensor


# The utilities below already rely on Pillow / OpenCV / NumPy, so they do not
# need to be rewritten for this open-source release.

def decolorization(image):
    gray_image = image.convert('L')
    return Image.merge(image.mode, [gray_image] * 3) if image.mode in ('RGB', 'L') else gray_image


def downscale(image, scale_factor):
    new_width = int(round(image.width * scale_factor))
    new_height = int(round(image.height * scale_factor))
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def crop(image, crop_factors):
    target_h, target_w = crop_factors
    img_w, img_h = image.size

    if target_h > img_h or target_w > img_w:
        raise ValueError("Crop size exceeds image dimensions")

    x = random.randint(0, img_w - target_w)
    y = random.randint(0, img_h - target_h)

    return image.crop((x, y, x + target_w, y + target_h)), [[x, y], [x + target_w, y + target_h]]


def motion_blur_opencv(image, kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    rotated_kernel /= rotated_kernel.sum() if rotated_kernel.sum() != 0 else 1
    img = np.array(image)
    if img.ndim == 2:
        blurred = cv2.filter2D(img, -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)
    else:
        blurred = np.zeros_like(img)
        for c in range(img.shape[2]):
            blurred[..., c] = cv2.filter2D(img[..., c], -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)
    return Image.fromarray(blurred.astype(np.uint8))


def shuffle_patch(image, num_splits, gap_size=2):
    h_splits, w_splits = num_splits
    img_w, img_h = image.size
    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))
    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))
    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop((current_x, current_y, current_x + patch_w, current_y + patch_h))
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h
    random.shuffle(patches)
    total_width = sum(patch_widths) + (w_splits - 1) * gap_size
    total_height = sum(patch_heights) + (h_splits - 1) * gap_size
    new_image = Image.new(image.mode, (total_width, total_height), color=(255, 255, 255))
    current_y = 0
    patch_idx = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch = patches[patch_idx]
            patch_w = patch_widths[j]
            new_image.paste(patch, (current_x, current_y))
            current_x += patch_w + gap_size
            patch_idx += 1
        current_y += patch_h + gap_size
    return new_image


def inpainting(image, num_splits, blank_ratio=0.3, blank_color=(255, 255, 255)):
    h_splits, w_splits = num_splits
    img_w, img_h = image.size
    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))
    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))
    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop((current_x, current_y, current_x + patch_w, current_y + patch_h))
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h
    total_patches = h_splits * w_splits
    num_blank = int(total_patches * blank_ratio)
    num_blank = max(0, min(num_blank, total_patches))
    blank_indices = random.sample(range(total_patches), num_blank)
    processed_patches = []
    for idx, patch in enumerate(patches):
        if idx in blank_indices:
            blank_patch = Image.new("RGB", patch.size, color=blank_color)
            processed_patches.append(blank_patch)
        else:
            processed_patches.append(patch)
    result_image = Image.new("RGB", (img_w, img_h))
    current_y = 0
    patch_idx = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch = processed_patches[patch_idx]
            patch_w = patch_widths[j]
            result_image.paste(patch, (current_x, current_y))
            current_x += patch_w
            patch_idx += 1
        current_y += patch_h
    return result_image
