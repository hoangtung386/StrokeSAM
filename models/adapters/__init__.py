"""Image-conditioned adapter modules for HPMA."""

from models.adapters.global_adapter import GlobalVisualAdapter, VisualTokenGenerator
from models.adapters.local_adapter import LocalProjectionAdapter
from models.adapters.structural_adapter import StructuralQueryAdapter

__all__ = [
    "GlobalVisualAdapter",
    "VisualTokenGenerator",
    "StructuralQueryAdapter",
    "LocalProjectionAdapter",
]
