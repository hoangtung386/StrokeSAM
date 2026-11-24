"""Self-contained Mock implementation of SAM 3.1 architecture.

Provides image-path sub-module interfaces (backbone with FPN, fusion_encoder,
mask_decoder, segmentation_head) conforming to the HPMA integration contract.
Enables standalone unit tests, shape checks, and development without
requiring proprietary weights initially.
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTrunkBlock(nn.Module):
    """Transformer block in SAM 3.1 vision trunk."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


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


class MockFusionEncoder(nn.Module):
    """Mock Multimodal Fusion Encoder."""

    def __init__(self, feature_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([VisionTrunkBlock(dim=feature_dim) for _ in range(num_layers)])
        self.cross_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        image_features: Union[List[torch.Tensor], torch.Tensor],
        condition_features: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(image_features, list):
            img_feat = image_features[0]  # Take primary multi-scale map
        else:
            img_feat = image_features

        b, c, h, w = img_feat.shape
        work_size = (min(h, 18), min(w, 18))
        img_feat = F.adaptive_avg_pool2d(img_feat, work_size)
        img_flat = img_feat.permute(0, 2, 3, 1).reshape(b, -1, c)

        condition_summary = condition_features.mean(dim=1, keepdim=True)
        fused = img_flat + self.cross_proj(condition_summary)

        for layer in self.layers:
            fused = layer(fused)

        fused = fused.reshape(b, work_size[0], work_size[1], c).permute(0, 3, 1, 2)
        return F.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)


class MockMaskDecoder(nn.Module):
    """Mock Transformer Decoder generating Mask Queries."""

    def __init__(self, feature_dim: int = 256, num_queries: int = 16) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        self.decoder_layers = nn.ModuleList([VisionTrunkBlock(dim=feature_dim) for _ in range(4)])
        self.feature_projection = nn.Linear(feature_dim, feature_dim)

    def get_object_queries(self, batch_size: int = 1) -> torch.Tensor:
        return self.object_queries.repeat(batch_size, 1, 1)

    def forward(
        self,
        fused_features: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if fused_features.ndim == 4:
            feature_summary = fused_features.mean(dim=(2, 3))
        else:
            feature_summary = fused_features.mean(dim=1)
        q = queries + self.feature_projection(feature_summary).unsqueeze(1)
        for layer in self.decoder_layers:
            q = layer(q)
        return q


class MockSegmentationHead(nn.Module):
    """Frozen SAM 3.1 Segmentation Head (Figure 2b)."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.conv_out = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(
        self,
        mask_queries: torch.Tensor,
        high_res_features: torch.Tensor,
    ) -> torch.Tensor:
        # High resolution feature map: [B, D, 288, 288]
        query_context = self.query_projection(mask_queries.mean(dim=1))
        conditioned_features = high_res_features + query_context[:, :, None, None]
        conditioned_features = F.adaptive_avg_pool2d(conditioned_features, (72, 72))
        logits = self.conv_out(conditioned_features)  # [B, 1, 288, 288]
        # Upsample to full image resolution: [B, 1, 1008, 1008]
        pred_masks = F.interpolate(logits, size=(1008, 1008), mode="bilinear", align_corners=False)
        return torch.sigmoid(pred_masks)


class MockSAM3Model(nn.Module):
    """Full Mock SAM 3.1 Model with sub-module hierarchy."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone = MockFPNImageEncoder(feature_dim=feature_dim)
        self.fusion_encoder = MockFusionEncoder(feature_dim=feature_dim)
        self.mask_decoder = MockMaskDecoder(feature_dim=feature_dim)
        self.segmentation_head = MockSegmentationHead(feature_dim=feature_dim)
