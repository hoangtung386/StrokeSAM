"""Test script to verify tensor shapes through HPMA forward pass."""

import os
import sys

import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hpma_model import HPMAModel  # noqa: E402


def verify_shapes() -> bool:
    """Run dummy forward pass to check all tensor shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running tensor shape verification on device: {device}")

    # Standard input resolutions
    dummy_img = torch.randn(1, 3, 1008, 1008, device=device)
    model = HPMAModel(use_mock_sam3=True).to(device)
    model.eval()

    with torch.no_grad():
        out = model(dummy_img)

    pred_masks = out["pred_masks"]
    f0 = out["high_res_features"]

    print(f"[✓] Predicted Mask Shape: {tuple(pred_masks.shape)}")
    print(f"[✓] High-res Feature F0 Shape: {tuple(f0.shape)}")

    assert pred_masks.shape[-2:] == (1008, 1008), "Mask output must be (1008, 1008)!"
    assert f0.shape[-2:] == (288, 288), "F0 local feature must be (288, 288)!"
    print("[✓] ALL TENSOR SHAPES MATCH FIGURE 2 SPECIFICATIONS 100%!")
    return True


if __name__ == "__main__":
    verify_shapes()
