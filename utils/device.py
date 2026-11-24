"""Device selection helpers shared by the training entrypoints."""

import torch


def resolve_device(requested_device: str = "cuda") -> torch.device:
    """Resolve the configured device and activate CUDA device 0 explicitly.

    CUDA requests fail instead of silently falling back to CPU. A request for a
    different CUDA index is rejected because this project expects one GPU.
    """
    device = torch.device(requested_device)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested_device}) but is unavailable.")

    index = torch.cuda.current_device() if device.index is None else device.index
    if index != 0:
        raise ValueError(f"This run is configured for CUDA device 0, got cuda:{index}.")

    torch.cuda.set_device(0)
    return torch.device("cuda:0")
