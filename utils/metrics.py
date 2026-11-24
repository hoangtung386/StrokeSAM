"""Evaluation metrics for binary and class-conditioned segmentation.

Calculates:
    - IoU: Global mean Intersection over Union.
    - Mean Class IoU (mc IoU): Macro-averaged IoU across all categories.
    - Mean Dice (mDice): Macro-averaged Dice coefficient.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch


def calculate_iou_and_dice(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Calculate IoU and Dice for binary masks."""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = (intersection + eps) / (union + eps) if union > 0 else 1.0

    total = pred.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (total + eps) if total > 0 else 1.0
    return float(iou), float(dice)


class SegmentationMetrics:
    """Accumulate per-class and binary segmentation metrics across batches."""

    def __init__(self, num_classes: int = 1) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.class_ious: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
        self.class_dices: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
        self.total_intersection = 0
        self.total_union = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Accumulate predictions and compute frame-level metrics.

        Args:
            pred_masks: Tensor of shape [B, 1, H, W] in [0, 1].
            gt_masks: Tensor of shape [B, 1, H, W] binary.
            labels: Tensor of shape [B, 1] class indices.
            threshold: Probability cutoff for binary mask.
        """
        preds_bin = (pred_masks > threshold).cpu().numpy().astype(bool)
        gts_bin = (gt_masks > 0.5).cpu().numpy().astype(bool)
        lbls = labels.cpu().numpy().flatten()

        for i in range(len(preds_bin)):
            c = int(lbls[i])
            if c < 0 or c >= self.num_classes:
                continue

            iou, dice = calculate_iou_and_dice(preds_bin[i, 0], gts_bin[i, 0])
            self.class_ious[c].append(iou)
            self.class_dices[c].append(dice)
            intersection = np.logical_and(preds_bin[i, 0], gts_bin[i, 0]).sum()
            union = np.logical_or(preds_bin[i, 0], gts_bin[i, 0]).sum()
            self.true_positive += int(intersection)
            self.false_positive += int(preds_bin[i, 0].sum() - intersection)
            self.false_negative += int(gts_bin[i, 0].sum() - intersection)
            self.total_intersection += int(intersection)
            self.total_union += int(union)

    def compute(self) -> Dict[str, float]:
        """Compute aggregated benchmark metrics."""
        mc_iou_list = []
        mdice_list = []

        per_class_results = {}
        for c in range(self.num_classes):
            if len(self.class_ious[c]) > 0:
                c_iou = float(np.mean(self.class_ious[c]))
                c_dice = float(np.mean(self.class_dices[c]))
                mc_iou_list.append(c_iou)
                mdice_list.append(c_dice)
                per_class_results[f"class_{c}_iou"] = c_iou
            else:
                per_class_results[f"class_{c}_iou"] = 0.0

        mc_iou = float(np.mean(mc_iou_list)) if mc_iou_list else 0.0
        mdice = float(np.mean(mdice_list)) if mdice_list else 0.0
        global_iou = (self.total_intersection / self.total_union) if self.total_union else 1.0
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = self.true_positive / precision_denominator if precision_denominator else 1.0
        recall = self.true_positive / recall_denominator if recall_denominator else 1.0

        return {
            "iou": global_iou * 100.0,
            "mc_iou": mc_iou * 100.0,
            "mdice": mdice * 100.0,
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            **per_class_results,
        }
