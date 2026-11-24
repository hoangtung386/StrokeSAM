"""Layer-wise AdamW configuration for image-only HPMA."""

from typing import List

import torch.nn as nn
from torch.optim import AdamW


def build_hpma_optimizer(
    model: nn.Module,
    base_lr: float = 8.0e-5,
    vision_lr: float = 2.5e-5,
    weight_decay: float = 1.0e-4,
) -> AdamW:
    """Construct AdamW optimizer with layer-wise parameter groups.

    Args:
        model: HPMAModel instance.
        base_lr: Base learning rate for newly added/fine-tuned components.
        vision_lr: Smaller rate for unfrozen vision trunk layers.
        weight_decay: Decoupled weight decay.

    Returns:
        Configured AdamW optimizer.
    """
    vision_params: List[nn.Parameter] = []
    base_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "vision_encoder" in name:
            vision_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": vision_params, "lr": vision_lr, "weight_decay": weight_decay},
    ]

    print(
        f"[*] Optimizer groups: Base ({len(base_params)} params, lr={base_lr}), "
        f"Vision ({len(vision_params)} params, lr={vision_lr})"
    )

    return AdamW(param_groups)
