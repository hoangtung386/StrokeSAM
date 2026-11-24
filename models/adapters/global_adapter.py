"""Visual token generation and global prototype retrieval adapters."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualTokenGenerator(nn.Module):
    """Convert a global image feature map into a compact token sequence."""

    def __init__(
        self,
        dim: int = 256,
        token_grid: Tuple[int, int] = (4, 4),
    ) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(global_features, self.token_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.projection(tokens)


class GlobalVisualAdapter(nn.Module):
    """Condition image-derived tokens on the complete global prototype bank."""

    def __init__(self, dim: int = 256, init_gate: float = 0.01) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.prototype_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.scale = dim**-0.5
        self.learnable_gate = nn.Parameter(torch.tensor(init_gate))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        global_prototypes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if global_prototypes.ndim == 2:
            global_prototypes = global_prototypes.unsqueeze(0).expand(
                visual_tokens.shape[0], -1, -1
            )

        queries = self.query_projection(self.query_norm(visual_tokens))
        prototypes = self.prototype_projection(global_prototypes)
        keys = self.key_projection(prototypes)
        values = self.value_projection(prototypes)

        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        context = self.output_projection(torch.bmm(attention, values))
        return visual_tokens + self.learnable_gate * context
