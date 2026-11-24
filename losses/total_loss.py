"""Total Loss Objective for HPMA.

Combines base SAM 3.1 loss with the Local Alignment loss:
    L = L_SAM3 + lambda_align * L_align
Where lambda_align = 0.01.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from losses.local_align_loss import LocalAlignmentLoss
from losses.sam3_loss import SAM3BaseLoss
from models.adapters.local_adapter import LocalProjectionAdapter


class HPMATotalLoss(nn.Module):
    """Total HPMA optimization objective."""

    def __init__(
        self,
        lambda_align: float = 0.01,
        local_adapter: Optional[LocalProjectionAdapter] = None,
    ) -> None:
        """Initialize HPMATotalLoss.

        Args:
            lambda_align: Weighting scalar for local alignment (0.01).
            local_adapter: Trainable local projection module.
        """
        super().__init__()
        self.lambda_align = lambda_align
        self.base_loss = SAM3BaseLoss()
        self.align_loss = LocalAlignmentLoss(local_adapter=local_adapter)

    def forward(
        self,
        pred_masks: torch.Tensor,
        f0_local: torch.Tensor,
        gt_masks: torch.Tensor,
        labels: torch.Tensor,
        p_local_bank: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            pred_masks: Predicted binary masks [B, 1, 1008, 1008].
            f0_local: High-res FPN feature map [B, D, 288, 288].
            gt_masks: Target binary masks [B, 1, 1008, 1008].
            labels: Category labels [B, 1].
            p_local_bank: Frozen local prototype memory [|C|, K, D].

        Returns:
            Dictionary containing individual losses and total scalar loss.
        """
        base_losses = self.base_loss(pred_masks, gt_masks)
        l_sam3 = base_losses["loss_sam3_total"]

        l_align = self.align_loss(f0_local, gt_masks, labels, p_local_bank)
        total = l_sam3 + self.lambda_align * l_align

        return {
            "loss_total": total,
            "loss_sam3": l_sam3,
            "loss_align": l_align,
            "loss_focal": base_losses["loss_focal"],
            "loss_dice": base_losses["loss_dice"],
        }
