"""Unit test for full HPMA forward pass and selective parameter freezing."""

import unittest

import torch

from models.hpma_model import HPMAModel


class TestHPMAModelForward(unittest.TestCase):
    """Test full forward integration and parameter status."""

    def setUp(self):
        self.model = HPMAModel(feature_dim=256, num_classes=1, use_mock_sam3=True)

    def test_parameter_freezing(self):
        """Verify that Segmentation Head is frozen and Decoder is trainable."""
        # 1. Segmentation Head parameters must be frozen ❄️
        if self.model.segmentation_head is not None:
            for p in self.model.segmentation_head.parameters():
                self.assertFalse(p.requires_grad, "Segmentation Head must be FROZEN ❄️!")

        # 2. Decoder parameters must be trainable 🔥
        trainable_decoder = any(
            p.requires_grad for p in self.model.transformer_decoder.parameters()
        )
        self.assertTrue(trainable_decoder, "Decoder must be TRAINABLE 🔥!")

        # 3. Adapters must be trainable 🔥
        self.assertTrue(self.model.global_visual_adapter.learnable_gate.requires_grad)
        self.assertTrue(self.model.structural_adapter.learnable_gate.requires_grad)

    def test_forward_shape(self):
        """Test output mask and feature shapes."""
        b = 2
        images = torch.randn(b, 3, 1008, 1008)
        out = self.model(images)
        self.assertIn("pred_masks", out)
        self.assertIn("high_res_features", out)

        pred_masks = out["pred_masks"]
        f0 = out["high_res_features"]

        self.assertEqual(pred_masks.shape, (b, 1, 1008, 1008))
        self.assertEqual(f0.shape, (b, 256, 288, 288))


if __name__ == "__main__":
    unittest.main()
