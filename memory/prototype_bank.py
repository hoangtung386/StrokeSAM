"""Prototype Memory Bank management and loading.

Manages the strictly frozen multi-scale prototype bank tensor:
    P in R^(3 x |C| x K x d)
Where:
    - 3: scale indices [0: local (288x288), 1: structural (144x144), 2: global (72x72)]
    - |C|: number of foreground categories configured for the dataset
    - K: number of prototypes per category (K=4)
    - d: feature channel dimension (d=256)
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class PrototypeMemoryBank(nn.Module):
    """Frozen visual prototype memory bank."""

    def __init__(
        self,
        bank_path: Optional[str] = None,
        num_scales: int = 3,
        num_classes: int = 1,
        num_prototypes: int = 4,
        dim: int = 256,
        metadata: Optional[Dict[str, Any]] = None,
        bank_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the PrototypeMemoryBank.

        Args:
            bank_path: Path to cached .pt prototype bank.
            num_scales: Number of representation scales (3).
            num_classes: Number of foreground categories in the dataset.
            num_prototypes: Number of prototypes per category (K=4).
            dim: Feature dimension (256).
        """
        super().__init__()
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.metadata = metadata or {}

        # Initialize memory tensor
        if bank_path and bank_tensor is not None:
            raise ValueError("Pass either bank_path or bank_tensor, not both.")
        if bank_path:
            if not os.path.isfile(bank_path):
                raise FileNotFoundError(f"Prototype bank does not exist: {bank_path}")
            print(f"[*] Loading pre-computed prototype bank from: {bank_path}")
            bank_tensor = torch.load(bank_path, map_location="cpu", weights_only=True)
            if isinstance(bank_tensor, dict):
                self.metadata = bank_tensor.get("metadata", {})
                bank_tensor = bank_tensor.get("prototypes", bank_tensor)
        elif bank_tensor is not None:
            bank_tensor = bank_tensor.detach().to(device="cpu")
        else:
            print(
                f"[*] Initializing random prototype memory bank [3, {num_classes}, "
                f"{num_prototypes}, {dim}]."
            )
            bank_tensor = torch.randn(num_scales, num_classes, num_prototypes, dim)

        expected_shape = (num_scales, num_classes, num_prototypes, dim)
        if tuple(bank_tensor.shape) != expected_shape:
            raise ValueError(
                f"Prototype bank has shape {tuple(bank_tensor.shape)}; "
                f"expected {expected_shape}."
            )

        # Register as frozen buffer (not a learnable parameter)
        self.register_buffer("bank", bank_tensor.float())

    @property
    def local_prototypes(self) -> torch.Tensor:
        """Return local prototypes for all classes of shape [|C|, K, D]."""
        return self.bank[0]

    @property
    def structural_prototypes(self) -> torch.Tensor:
        """Return structural prototypes for all classes of shape [|C|, K, D]."""
        return self.bank[1]

    @property
    def global_prototypes(self) -> torch.Tensor:
        """Return global prototypes for all classes of shape [|C|, K, D]."""
        return self.bank[2]

    def global_all(self) -> torch.Tensor:
        """Return all global prototypes as a label-independent ``[N, D]`` bank."""
        return self.global_prototypes.reshape(-1, self.dim)

    def structural_all(self) -> torch.Tensor:
        """Return all structural prototypes as a label-independent ``[N, D]`` bank."""
        return self.structural_prototypes.reshape(-1, self.dim)

    def save(
        self,
        save_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize memory bank to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        payload = {
            "prototypes": self.bank.cpu(),
            "metadata": metadata if metadata is not None else self.metadata,
        }
        torch.save(payload, save_path)
        print(f"[✓] Prototype memory bank saved to {save_path}")
