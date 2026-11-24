"""Script to inspect sub-module hierarchy of SAM 3.1."""

import argparse
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.sam3.model_builder import build_sam3_image_model  # noqa: E402


def inspect_sam3(
    checkpoint_path: str = "checkpoints/sam3.1.pt",
    use_mock: bool = False,
) -> None:
    """Print structural block breakdown of SAM 3.1."""
    print(f"[*] Inspecting SAM 3.1 model hierarchy (Checkpoint: {checkpoint_path}) ...")
    sam_model = build_sam3_image_model(checkpoint_path=checkpoint_path, use_mock=use_mock)

    print("\n" + "=" * 65)
    print(f"{'BLOCK NAME':<25} | {'MODULE TYPE':<25} | {'PARAMS (M)':<10}")
    print("=" * 65)
    for name, module in sam_model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:<25} | {type(module).__name__:<25} | {num_params / 1e6:.2f}M")

    print("\n" + "=" * 65)
    print("VISION BACKBONE FPN DETAILS:")
    print("=" * 65)
    if hasattr(sam_model, "backbone"):
        for sub_name, sub_module in sam_model.backbone.named_children():
            print(f"  └── {sub_name:<20} | {type(sub_module).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SAM 3.1 submodules")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam3.1.pt",
        help="Path to SAM 3.1 checkpoint",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Inspect the lightweight mock instead of an official checkpoint.",
    )
    args = parser.parse_args()
    inspect_sam3(checkpoint_path=args.checkpoint, use_mock=args.mock)
