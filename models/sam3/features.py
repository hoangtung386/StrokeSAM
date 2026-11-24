"""Image-only feature extraction helpers for official SAM 3 and test doubles."""

from typing import List

import torch
import torch.nn as nn


def extract_image_fpn_features(
    encoder: nn.Module,
    images: torch.Tensor,
) -> List[torch.Tensor]:
    """Extract the three image FPN maps without invoking SAM's text pathway."""
    forward_image = getattr(encoder, "forward_image", None)
    outputs = forward_image(images) if callable(forward_image) else encoder(images)

    if isinstance(outputs, dict):
        fpn_features = outputs.get("fpn_features", outputs.get("backbone_fpn"))
    elif isinstance(outputs, (list, tuple)):
        fpn_features = outputs
    else:
        fpn_features = None

    if not isinstance(fpn_features, (list, tuple)):
        raise ValueError(
            "The image encoder must return FPN maps as 'fpn_features' or 'backbone_fpn'."
        )
    if len(fpn_features) < 3:
        raise ValueError(
            "The image encoder must return local, structural, and global FPN maps; "
            f"received {len(fpn_features)} map(s)."
        )
    return list(fpn_features[:3])
