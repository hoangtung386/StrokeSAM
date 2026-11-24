"""Local Alignment Objective (Figure 2c - Local Level Alignment).

Computes directional Cosine Distance alignment loss between spatially pooled
high-resolution image features (F0 at 288x288) and projected local prototypes.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapters.local_adapter import LocalProjectionAdapter


class LocalAlignmentLoss(nn.Module):
    """Directional Cosine Distance Alignment Loss."""

    def __init__(
        self,
        local_adapter: Optional[LocalProjectionAdapter] = None,
        eps: float = 1e-6,
    ) -> None:
        """Initialize LocalAlignmentLoss.

        Args:
            local_adapter: Trainable local projection adapter (Figure 2c).
            eps: Numerical stability constant.
        """
        super().__init__()
        self.local_adapter = local_adapter or LocalProjectionAdapter()
        self.eps = eps

    def forward(
        self,
        f0_local: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        p_local_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate local alignment loss.

        Args:
            f0_local: High-resolution image features of shape [B, D, 288, 288].
            masks: Ground truth masks of shape [B, N_obj, H, W].
            labels: Category indices of shape [B, N_obj].
            p_local_bank: Frozen local prototype bank of shape [|C|, K, D].

        Returns:
            Scalar alignment loss.
        """
        b, d, h_f, w_f = f0_local.shape
        loss_items: List[torch.Tensor] = []

        # Interpolate masks to 288x288 resolution
        n_obj = masks.shape[1]
        masks_resized = F.interpolate(
            masks.float().view(-1, 1, masks.shape[-2], masks.shape[-1]),
            size=(h_f, w_f),
            mode="nearest",
        ).view(b, n_obj, h_f, w_f)

        for b_i in range(b):
            for obj_i in range(n_obj):
                w_mask = masks_resized[b_i, obj_i]  # [H_f, W_f]
                mask_sum = w_mask.sum()
                if mask_sum < 1.0:
                    continue

                # Masked Average Pooling (Figure 2c)
                w_exp = w_mask.unsqueeze(0)  # [1, H_f, W_f]
                phi_i = (f0_local[b_i] * w_exp).sum(dim=(1, 2)) / (mask_sum + self.eps)  # [D]
                phi_i_norm = F.normalize(phi_i, p=2, dim=0)

                c_i = int(labels[b_i, obj_i].item())
                if c_i < 0 or c_i >= p_local_bank.shape[0]:
                    continue

                # Local prototype projection (Figure 2c: Proj.)
                p_c_raw = p_local_bank[c_i]  # [K, D]
                p_c_proj = self.local_adapter(p_c_raw)  # [K, D]
                p_c_norm = F.normalize(p_c_proj, p=2, dim=-1)

                # Directional alignment: min_{k in {1..K}} (1 - cos(phi_i, p_k))
                cos_sim = torch.mv(p_c_norm, phi_i_norm)  # [K]
                cos_dist = 1.0 - cos_sim
                min_cos_dist = torch.min(cos_dist)
                loss_items.append(min_cos_dist)

        if len(loss_items) == 0:
            return f0_local.sum() * 0.0

        return torch.stack(loss_items).mean()
