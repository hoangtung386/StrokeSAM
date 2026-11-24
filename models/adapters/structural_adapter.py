"""Retrieve structural visual prototypes from image features and adapt queries."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralQueryAdapter(nn.Module):
    """Structural level query residual adapter.

    Structure according to Figure 2(c):
        - Structural proto: LayerNorm + Linear + GELU + Linear
        - x Learnable gate -> + Object Query
    """

    def __init__(self, dim: int = 256, init_gate: float = 0.01) -> None:
        """Initialize StructuralQueryAdapter.

        Args:
            dim: Feature dimension (d=256).
            init_gate: Initial value for the learnable gate scalar.
        """
        super().__init__()
        self.dim = dim

        self.query_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.prototype_norm = nn.LayerNorm(dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.proto_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.learnable_gate = nn.Parameter(torch.tensor(init_gate))
        self.scale = dim**-0.5

    def forward(
        self,
        object_queries: torch.Tensor,
        structural_features: torch.Tensor,
        structural_proto: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate object queries with structural prototypes.

        Args:
            object_queries: Transformer Decoder queries of shape [B, num_queries, D].
            structural_features: Image feature map [B, D, H, W].
            structural_proto: Complete frozen prototype bank [N, D].

        Returns:
            Modulated object queries of shape [B, num_queries, D].
        """
        if structural_proto.ndim == 2:
            structural_proto = structural_proto.unsqueeze(0).expand(object_queries.shape[0], -1, -1)

        image_summary = F.adaptive_avg_pool2d(structural_features, 1).flatten(1)
        query = self.query_projection(image_summary).unsqueeze(1)
        projected_prototypes = self.prototype_norm(structural_proto)
        keys = self.key_projection(projected_prototypes)
        values = self.value_projection(projected_prototypes)
        scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        retrieved = torch.bmm(weights, values)
        delta_q = self.proto_proj(retrieved)

        # Inject structural bias into object queries
        modulated_queries = object_queries + self.learnable_gate * delta_q
        return modulated_queries
