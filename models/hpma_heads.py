"""Trainable image-conditioned fusion and mask heads used by the HPMA model."""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTrunkBlock(nn.Module):
    """Small transformer-style residual block for image tokens and queries."""

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


class HPMAFusionEncoder(nn.Module):
    """Fuse multi-scale image maps with image-derived visual context tokens."""

    def __init__(self, feature_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VisionTrunkBlock(dim=feature_dim) for _ in range(num_layers)]
        )
        self.cross_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        image_features: Union[List[torch.Tensor], torch.Tensor],
        condition_features: torch.Tensor,
    ) -> torch.Tensor:
        image = image_features[0] if isinstance(image_features, list) else image_features
        batch, channels, height, width = image.shape
        work_size = (min(height, 18), min(width, 18))
        image = F.adaptive_avg_pool2d(image, work_size)
        tokens = image.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        tokens = tokens + self.cross_proj(condition_features.mean(dim=1, keepdim=True))

        for layer in self.layers:
            tokens = layer(tokens)

        fused = tokens.reshape(batch, *work_size, channels).permute(0, 3, 1, 2)
        return F.interpolate(fused, size=(height, width), mode="bilinear", align_corners=False)


class HPMAMaskDecoder(nn.Module):
    """Generate mask queries from fused image features and learned object queries."""

    def __init__(self, feature_dim: int = 256, num_queries: int = 16) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        self.decoder_layers = nn.ModuleList(
            [VisionTrunkBlock(dim=feature_dim) for _ in range(4)]
        )
        self.feature_projection = nn.Linear(feature_dim, feature_dim)

    def get_object_queries(self, batch_size: int = 1) -> torch.Tensor:
        return self.object_queries.repeat(batch_size, 1, 1)

    def forward(self, fused_features: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        if fused_features.ndim == 4:
            summary = fused_features.mean(dim=(2, 3))
        else:
            summary = fused_features.mean(dim=1)
        queries = queries + self.feature_projection(summary).unsqueeze(1)
        for layer in self.decoder_layers:
            queries = layer(queries)
        return queries


class HPMASegmentationHead(nn.Module):
    """Predict a dense binary mask from high-resolution features and mask queries."""

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
        query_context = self.query_projection(mask_queries.mean(dim=1))
        conditioned = high_res_features + query_context[:, :, None, None]
        conditioned = F.adaptive_avg_pool2d(conditioned, (72, 72))
        logits = self.conv_out(conditioned)
        logits = F.interpolate(
            logits,
            size=(1008, 1008),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits)
