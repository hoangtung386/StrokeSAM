"""Offline Multi-Scale Visual Prototype Memory Bank Builder (Figure 2a).

Extracts multi-scale FPN features from annotated 2D training images,
aggregates features via Masked Average Pooling (MAP), and performs offline
K-Means clustering (K=4) to construct the frozen prototype memory bank.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class PrototypeMemoryBuilder:
    """Builder for multi-scale frozen prototype bank."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        num_classes: int = 1,
        num_prototypes: int = 4,
        dim: int = 256,
        device: str = "cpu",
    ) -> None:
        """Initialize PrototypeMemoryBuilder.

        Args:
            vision_encoder: Frozen SAM 3.1 image encoder backbone.
            num_classes: Total categories |C|.
            num_prototypes: K prototypes per class (K=4).
            dim: Feature channel dimension (256).
            device: Device to run feature extraction on.
        """
        self.vision_encoder = vision_encoder.to(device)
        self.vision_encoder.eval()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.device = device

        # Raw feature pools: dict of lists per scale and class
        # scales: 0 -> local (288x288), 1 -> struct (144x144), 2 -> global (72x72)
        self.pools: Dict[int, Dict[int, List[np.ndarray]]] = {
            s: {c: [] for c in range(num_classes)} for s in range(3)
        }

    @torch.no_grad()
    def collect_features_from_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Extract multi-scale features and perform Masked Average Pooling.

        Args:
            images: Tensor of shape [B, 3, 1008, 1008].
            masks: Ground truth masks of shape [B, N_obj, H, W].
            labels: Category indices of shape [B, N_obj].
            eps: Epsilon to prevent division by zero.
        """
        images = images.to(self.device)
        feats = self.vision_encoder(images)

        if isinstance(feats, dict) and "fpn_features" in feats:
            fpn_feats = feats["fpn_features"]
        elif isinstance(feats, (list, tuple)):
            fpn_feats = feats
        else:
            raise ValueError("Vision encoder output does not contain valid FPN features.")

        # fpn_feats[0]: local [B, D, 288, 288]
        # fpn_feats[1]: struct [B, D, 144, 144]
        # fpn_feats[2]: global [B, D, 72, 72]

        b, n_obj = labels.shape
        for s_idx, feat_map in enumerate(fpn_feats[:3]):
            h_f, w_f = feat_map.shape[-2:]
            # Interpolate binary masks to current scale resolution
            resized_masks = F.interpolate(
                masks.float().view(-1, 1, masks.shape[-2], masks.shape[-1]),
                size=(h_f, w_f),
                mode="nearest",
            ).view(b, n_obj, h_f, w_f)

            for b_i in range(b):
                for obj_i in range(n_obj):
                    class_id = int(labels[b_i, obj_i].item())
                    if class_id < 0 or class_id >= self.num_classes:
                        continue

                    w_mask = resized_masks[b_i, obj_i]  # [H_f, W_f]
                    mask_sum = w_mask.sum().item()
                    if mask_sum < 1.0:
                        continue

                    # Masked Average Pooling
                    pooled = (feat_map[b_i] * w_mask.unsqueeze(0)).sum(dim=(1, 2)) / (
                        mask_sum + eps
                    )
                    pooled = F.normalize(pooled, p=2, dim=0)
                    self.pools[s_idx][class_id].append(pooled.cpu().numpy())

    def build_bank(self) -> torch.Tensor:
        """Run K-Means clustering (K=4) on collected features.

        Returns:
            Bank tensor of shape [3, |C|, K, D].
        """
        bank = torch.zeros(3, self.num_classes, self.num_prototypes, self.dim)

        for s_idx in range(3):
            for c_idx in range(self.num_classes):
                vectors = self.pools[s_idx][c_idx]
                if len(vectors) == 0:
                    raise ValueError(
                        f"No foreground features collected for class {c_idx} "
                        f"at scale {s_idx}; refusing to create random memory prototypes."
                    )

                x = np.array(vectors)  # [N_samples, D]
                num_samples = len(x)

                if num_samples < self.num_prototypes:
                    # Tile or duplicate samples if fewer than K
                    reps = int(np.ceil(self.num_prototypes / num_samples))
                    x = np.tile(x, (reps, 1))[: self.num_prototypes]
                    centroids = x
                else:
                    kmeans = KMeans(
                        n_clusters=self.num_prototypes,
                        random_state=42,
                        n_init=10,
                    )
                    kmeans.fit(x)
                    centroids = kmeans.cluster_centers_

                bank[s_idx, c_idx] = F.normalize(
                    torch.from_numpy(centroids).float(),
                    p=2,
                    dim=-1,
                )

        print(f"[✓] Successfully built prototype bank with shape: {tuple(bank.shape)}")
        return bank
