"""Standard SAM 3.1 loss components.

Includes Focal Loss, Dice Loss, and presence classification loss.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for segmentation masks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.float()
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation masks."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1).float()

        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + self.eps) / (total + self.eps)
        return 1.0 - dice


class SAM3BaseLoss(nn.Module):
    """Compound SAM 3.1 segmentation loss."""

    def __init__(self) -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        focal = self.focal_loss(pred_masks, target_masks)
        dice = self.dice_loss(pred_masks, target_masks)
        total = focal + dice
        return {
            "loss_focal": focal,
            "loss_dice": dice,
            "loss_sam3_total": total,
        }
