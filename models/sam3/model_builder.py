"""Explicit SAM 3.1 builder with an opt-in lightweight development mock."""

import os
from typing import Optional

import torch.nn as nn

from models.sam3.mock_sam3 import MockSAM3Model


def build_sam3_image_model(
    checkpoint_path: Optional[str] = None,
    feature_dim: int = 256,
    use_mock: bool = False,
) -> nn.Module:
    """Build the official SAM image model or an explicitly requested mock.

    Args:
        checkpoint_path: Path to the official SAM 3.1 checkpoint.
        feature_dim: Channel dimension for feature representations.
        use_mock: Return the repository's mock model for tests and shape checks.

    Returns:
        SAM 3.1 compatible PyTorch model.
    """
    if use_mock:
        return MockSAM3Model(feature_dim=feature_dim)
    if not checkpoint_path:
        raise ValueError("A SAM 3.1 checkpoint is required unless use_mock=True.")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM 3.1 checkpoint does not exist: {checkpoint_path}")

    try:
        from sam3.model_builder import build_sam3_image_model as official_build
    except ImportError as error:
        raise ImportError(
            "The official 'sam3' package is required to load a SAM 3.1 checkpoint. "
            "Use use_mock=True only for development and shape checks."
        ) from error

    print(f"[*] Loading official SAM 3.1 weights from {checkpoint_path}")
    return official_build(checkpoint_path=checkpoint_path)
