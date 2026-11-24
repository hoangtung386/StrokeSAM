"""Unit tests for HPMA adapter modules."""

import unittest

import torch

from models.adapters.global_adapter import GlobalVisualAdapter, VisualTokenGenerator
from models.adapters.local_adapter import LocalProjectionAdapter
from models.adapters.structural_adapter import StructuralQueryAdapter


class TestHPMAAdapters(unittest.TestCase):
    """Test shape and gradients of adapters."""

    def setUp(self):
        self.dim = 256
        self.b = 2
        self.k = 4
        self.num_tokens = 16
        self.num_queries = 16

    def test_visual_token_generator(self):
        generator = VisualTokenGenerator(dim=self.dim, token_grid=(4, 4))
        global_features = torch.randn(self.b, self.dim, 72, 72)
        out = generator(global_features)
        self.assertEqual(out.shape, (self.b, 16, self.dim))

    def test_global_visual_adapter(self):
        adapter = GlobalVisualAdapter(dim=self.dim)
        visual_tokens = torch.randn(self.b, self.num_tokens, self.dim)
        global_proto = torch.randn(self.k * 7, self.dim)

        out = adapter(visual_tokens, global_proto)
        self.assertEqual(out.shape, (self.b, self.num_tokens, self.dim))

        # Check gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(adapter.learnable_gate.grad)

    def test_structural_query_adapter(self):
        adapter = StructuralQueryAdapter(dim=self.dim)
        queries = torch.randn(self.b, self.num_queries, self.dim)
        struct_features = torch.randn(self.b, self.dim, 144, 144)
        struct_proto = torch.randn(self.k * 7, self.dim)

        out = adapter(queries, struct_features, struct_proto)
        self.assertEqual(out.shape, (self.b, self.num_queries, self.dim))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(adapter.learnable_gate.grad)

    def test_local_projection_adapter(self):
        adapter = LocalProjectionAdapter(dim=self.dim)
        local_proto = torch.randn(self.k, self.dim)

        out = adapter(local_proto)
        self.assertEqual(out.shape, (self.k, self.dim))


if __name__ == "__main__":
    unittest.main()
