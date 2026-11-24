"""Local Detail Level Adapter for HPMA (Figure 2c).

Projects frozen local prototypes into the alignment feature space during training.
"""

import torch
import torch.nn as nn


class LocalProjectionAdapter(nn.Module):
    """Local level projection adapter.

    Structure according to Figure 2(c):
        - Local proto -> Proj. (LayerNorm + Linear)
    """

    def __init__(self, dim: int = 256) -> None:
        """Initialize LocalProjectionAdapter.

        Args:
            dim: Feature dimension (d=256).
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, local_proto: torch.Tensor) -> torch.Tensor:
        """Project local prototypes.

        Args:
            local_proto: Local prototypes of shape [..., D].

        Returns:
            Projected prototypes of shape [..., D].
        """
        return self.proj(local_proto)
