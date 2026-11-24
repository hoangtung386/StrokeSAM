"""Unit tests for HPMA loss functions."""

import unittest

import torch

from losses.local_align_loss import LocalAlignmentLoss
from losses.total_loss import HPMATotalLoss
from models.adapters.local_adapter import LocalProjectionAdapter


class TestHPMALosses(unittest.TestCase):
    """Test loss computations and gradients."""

    def setUp(self):
        self.b = 2
        self.dim = 256
        self.k = 4
        self.num_classes = 7

    def test_local_align_loss(self):
        local_adapter = LocalProjectionAdapter(dim=self.dim)
        loss_fn = LocalAlignmentLoss(local_adapter=local_adapter)

        f0_local = torch.randn(self.b, self.dim, 288, 288, requires_grad=True)
        masks = torch.ones(self.b, 1, 1008, 1008)
        labels = torch.tensor([[0], [1]], dtype=torch.long)
        p_bank = torch.randn(self.num_classes, self.k, self.dim)

        loss = loss_fn(f0_local, masks, labels, p_bank)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(f0_local.grad)

    def test_total_loss(self):
        total_loss_fn = HPMATotalLoss(lambda_align=0.01)
        pred_masks = torch.rand(self.b, 1, 1008, 1008, requires_grad=True)
        f0_local = torch.randn(self.b, self.dim, 288, 288, requires_grad=True)
        gt_masks = torch.ones(self.b, 1, 1008, 1008)
        labels = torch.tensor([[0], [1]], dtype=torch.long)
        p_bank = torch.randn(self.num_classes, self.k, self.dim)

        loss_dict = total_loss_fn(pred_masks, f0_local, gt_masks, labels, p_bank)
        self.assertIn("loss_total", loss_dict)
        self.assertIn("loss_align", loss_dict)
        self.assertIn("loss_sam3", loss_dict)

        loss = loss_dict["loss_total"]
        loss.backward()
        self.assertIsNotNone(pred_masks.grad)


if __name__ == "__main__":
    unittest.main()
