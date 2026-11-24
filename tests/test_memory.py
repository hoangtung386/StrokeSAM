"""Unit tests for Prototype Memory Bank."""

import os
import unittest

import torch

from memory.prototype_bank import PrototypeMemoryBank


class TestPrototypeMemoryBank(unittest.TestCase):
    """Test memory bank buffering and indexing."""

    def setUp(self):
        self.num_classes = 7
        self.num_prototypes = 4
        self.dim = 256
        self.cache_path = "memory/bank_cache/test_bank.pt"

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def test_bank_shapes_and_freezing(self):
        bank = PrototypeMemoryBank(
            num_classes=self.num_classes,
            num_prototypes=self.num_prototypes,
            dim=self.dim,
        )

        self.assertEqual(bank.bank.shape, (3, self.num_classes, self.num_prototypes, self.dim))
        self.assertFalse(bank.bank.requires_grad)
        self.assertEqual(
            bank.global_all().shape,
            (self.num_classes * self.num_prototypes, self.dim),
        )
        self.assertEqual(
            bank.structural_all().shape,
            (self.num_classes * self.num_prototypes, self.dim),
        )

        self.assertEqual(
            bank.local_prototypes.shape,
            (self.num_classes, self.num_prototypes, self.dim),
        )

    def test_save_and_reload(self):
        bank = PrototypeMemoryBank(
            num_classes=self.num_classes,
            num_prototypes=self.num_prototypes,
            dim=self.dim,
        )
        bank.save(self.cache_path)
        self.assertTrue(os.path.exists(self.cache_path))

        reloaded = PrototypeMemoryBank(
            bank_path=self.cache_path,
            num_classes=self.num_classes,
        )
        self.assertTrue(torch.allclose(bank.bank, reloaded.bank))


if __name__ == "__main__":
    unittest.main()
