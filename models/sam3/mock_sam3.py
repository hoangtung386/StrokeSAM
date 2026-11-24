"""Self-contained Mock implementation of SAM 3.1 architecture.

Provides image-path sub-module interfaces (backbone with FPN, fusion_encoder,
mask_decoder, segmentation_head) conforming to the HPMA integration contract.
Enables standalone unit tests, shape checks, and development without
requiring proprietary weights initially.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hpma_heads import (
    HPMAMaskDecoder as MockMaskDecoder,
    HPMASegmentationHead as MockSegmentationHead,
    HPMAFusionEncoder as MockFusionEncoder,
    VisionTrunkBlock,
)


class VisionTrunk(nn.Module):
    """Vision backbone trunk containing transformer blocks."""

    def __init__(self, num_blocks: int = 12, dim: int = 256) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([VisionTrunkBlock(dim=dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class MockFPNImageEncoder(nn.Module):
    """Mock Image Encoder extracting multi-scale FPN features.

    Produces:
        - F_local: [B, D, 288, 288]
        - F_struct: [B, D, 144, 144]
        - F_global: [B, D, 72, 72]
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.trunk = VisionTrunk(num_blocks=12, dim=feature_dim)
        self.conv_in = nn.Conv2d(64, feature_dim, kernel_size=1)

        # Multi-scale FPN projections
        self.fpn_local = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.fpn_struct = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.fpn_global = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        feat = self.stem(x)
        feat = self.conv_in(feat)

        # Keep the mock's transformer sequence small while preserving SAM FPN shapes.
        f_base = F.adaptive_avg_pool2d(feat, output_size=(18, 18))

        # Run trunk blocks on flattened sequence then reshape back
        b, c, h, w = f_base.shape
        flat = f_base.permute(0, 2, 3, 1).reshape(b, h * w, c)
        flat = self.trunk(flat)
        f_trunk = flat.reshape(b, h, w, c).permute(0, 3, 1, 2)

        local_base = F.interpolate(f_trunk, size=(72, 72), mode="bilinear", align_corners=False)
        f_local = F.interpolate(
            self.fpn_local(local_base), size=(288, 288), mode="bilinear", align_corners=False
        )
        f_struct = F.interpolate(
            self.fpn_struct(local_base), size=(144, 144), mode="bilinear", align_corners=False
        )
        f_global = F.interpolate(
            self.fpn_global(local_base), size=(72, 72), mode="bilinear", align_corners=False
        )

        return {"fpn_features": [f_local, f_struct, f_global]}


class MockSAM3Model(nn.Module):
    """Full Mock SAM 3.1 Model with sub-module hierarchy."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone = MockFPNImageEncoder(feature_dim=feature_dim)
        self.fusion_encoder = MockFusionEncoder(feature_dim=feature_dim)
        self.mask_decoder = MockMaskDecoder(feature_dim=feature_dim)
        self.segmentation_head = MockSegmentationHead(feature_dim=feature_dim)
