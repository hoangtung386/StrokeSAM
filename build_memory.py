"""Entrypoint script to extract and build the frozen Prototype Memory Bank."""

import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.factory import build_dataset
from memory.builder import PrototypeMemoryBuilder
from memory.prototype_bank import PrototypeMemoryBank
from models.sam3.model_builder import build_sam3_image_model
from models.sam3.features import ImageOnlySAM3Encoder
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Multi-scale Prototype Memory Bank")
    parser.add_argument(
        "--config", type=str, default="configs/isles2022.yaml", help="Dataset config"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base_config.yaml",
        help="Shared model and data configuration",
    )
    parser.add_argument(
        "--sam3_checkpoint", type=str, default=None, help="Optional SAM 3.1 checkpoint override"
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output .pt path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    requested_device = base_cfg["system"].get("device", "cuda")
    device = resolve_device(requested_device)
    use_data_parallel = base_cfg["system"].get("use_data_parallel", False)
    device_ids = list(range(torch.cuda.device_count())) if use_data_parallel else []
    if use_data_parallel and len(device_ids) < 2:
        raise RuntimeError("DataParallel is enabled, but fewer than two CUDA devices are visible.")
    print(f"[*] Building Prototype Memory Bank for {cfg['dataset']['name']} on {device}")

    # 1. Initialize dataset
    dataset = build_dataset(
        dataset_config=cfg["dataset"],
        split="train",
        target_size=tuple(base_cfg["data"]["input_resolution"]),
        seed=base_cfg["system"]["seed"],
        mean=tuple(base_cfg["data"]["mean"]),
        std=tuple(base_cfg["data"]["std"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=base_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=base_cfg["system"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # 2. Load frozen SAM 3.1 backbone
    sam_checkpoint = args.sam3_checkpoint or base_cfg["model"]["sam3_checkpoint"]
    sam_model = build_sam3_image_model(checkpoint_path=sam_checkpoint)
    vision_encoder: nn.Module = ImageOnlySAM3Encoder(sam_model.backbone)
    del sam_model
    if use_data_parallel:
        vision_encoder = nn.DataParallel(
            vision_encoder,
            device_ids=device_ids,
            output_device=0,
        )

    # 3. Build prototypes
    builder = PrototypeMemoryBuilder(
        vision_encoder=vision_encoder,
        num_classes=cfg["dataset"]["num_classes"],
        num_prototypes=base_cfg["model"]["num_prototypes_k"],
        dim=base_cfg["model"]["feature_dim"],
        device=str(device),
    )

    print(f"[*] Extracting FPN features across {len(dataset)} samples ...")
    for batch in dataloader:
        images = batch["image"]
        masks = batch["mask"]
        labels = batch["label"]
        builder.collect_features_from_batch(images, masks, labels)

    bank_tensor = builder.build_bank()

    # 4. Save to bank cache
    save_path = args.output_path or cfg["memory"]["bank_path"]
    memory_bank = PrototypeMemoryBank(
        bank_tensor=bank_tensor,
        num_classes=cfg["dataset"]["num_classes"],
        num_prototypes=builder.num_prototypes,
        dim=builder.dim,
    )
    metadata = {
        "dataset": cfg["dataset"]["name"],
        "split": "train",
        "split_seed": base_cfg["system"]["seed"],
        "validation_fraction": cfg["dataset"].get("validation_fraction", 0.10),
        "test_fraction": cfg["dataset"].get("test_fraction", 0.10),
        "case_ids": sorted(dataset.cases) if hasattr(dataset, "cases") else [],
        "modalities": list(getattr(dataset, "modalities", cfg["dataset"].get("channels", ()))),
        "input_resolution": list(base_cfg["data"]["input_resolution"]),
        "num_prototypes": builder.num_prototypes,
        "feature_dim": builder.dim,
        "sam_checkpoint": sam_checkpoint,
    }
    memory_bank.save(save_path, metadata=metadata)
    print(f"[✓] Prototype bank built and cached at {save_path}")


if __name__ == "__main__":
    main()
