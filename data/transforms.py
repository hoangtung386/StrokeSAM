"""Aspect-preserving image and mask preprocessing.

Resizes 2D images and masks to the model input size and applies configured
normalization for RGB images.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def resize_and_pad(
    tensor: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str,
) -> torch.Tensor:
    """Resize a ``[C,H,W]`` tensor without changing aspect ratio, then pad bottom/right."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected a [C, H, W] tensor, got shape {tuple(tensor.shape)}.")

    height, width = tensor.shape[-2:]
    target_height, target_width = target_size
    if min(height, width, target_height, target_width) <= 0:
        raise ValueError("Source and target spatial dimensions must be positive.")
    if mode not in {"bilinear", "bicubic", "nearest"}:
        raise ValueError(f"Unsupported resize mode: {mode}")
    scale = min(target_height / height, target_width / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))

    resize_options = {"size": (resized_height, resized_width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        resize_options["align_corners"] = False
        resize_options["antialias"] = True
    resized = F.interpolate(tensor.unsqueeze(0), **resize_options).squeeze(0)

    pad_bottom = target_height - resized_height
    pad_right = target_width - resized_width
    return F.pad(resized, (0, pad_right, 0, pad_bottom))


class ImageMaskTransform:
    """Preprocess paired RGB images and binary masks for SAM 3.1 HPMA."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (1008, 1008),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply transform.

        Args:
            image: HxWxC uint8 numpy array.
            mask: HxW uint8 or int32 numpy array.

        Returns:
            Tuple of (transformed_image, transformed_mask).
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image with shape [H, W, 3], got {image.shape}.")
        if mask is not None and mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
        if mask is not None and mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Image and mask dimensions differ: {image.shape[:2]} vs {mask.shape[:2]}."
            )

        # Convert to float tensor [3, H, W] in [0, 1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Normalize before padding so padded pixels remain zero after normalization.
        mean = torch.tensor(self.mean, dtype=img_tensor.dtype)[:, None, None]
        std = torch.tensor(self.std, dtype=img_tensor.dtype)[:, None, None]
        img_tensor = (img_tensor - mean) / std
        img_tensor = resize_and_pad(img_tensor, self.target_size, mode="bilinear")

        mask_tensor = None
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]
            mask_t = torch.from_numpy(mask).float()
            mask_tensor = resize_and_pad(mask_t, self.target_size, mode="nearest")

        return img_tensor, mask_tensor
