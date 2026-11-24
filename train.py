"""Training entrypoint for image-only 2D HPMA segmentation."""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.factory import build_dataset
from losses.total_loss import HPMATotalLoss
from memory.prototype_bank import PrototypeMemoryBank
from models.hpma_model import HPMAModel
from utils.device import resolve_device
from utils.logger import SimpleLogger
from utils.metrics import SegmentationMetrics
from utils.optimizer import build_hpma_optimizer


def train_one_epoch(
    model: HPMAModel,
    dataloader: DataLoader,
    criterion: HPMATotalLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    accumulation_steps: int = 1,
    mixed_precision: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive.")
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        gt_masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=torch.device(device).type,
            enabled=mixed_precision and torch.device(device).type == "cuda",
        ):
            outputs = model(images)
            loss_dict = criterion(
                outputs["pred_masks"],
                outputs["high_res_features"],
                gt_masks,
                labels,
                model.prototype_bank.local_prototypes,
            )
        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(dataloader) - group_start)
        loss = loss_dict["loss_total"] / group_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict["loss_total"].item()

    return total_loss / max(len(dataloader), 1)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_memory_bank(
    memory_bank: PrototypeMemoryBank,
    dataset_config: dict,
    base_config: dict,
    train_dataset,
) -> None:
    """Reject stale banks that were not built from this run's training split."""
    metadata = memory_bank.metadata
    if not metadata:
        raise ValueError("Prototype bank has no provenance metadata; rebuild it before training.")

    expected = {
        "dataset": dataset_config["name"],
        "split": "train",
        "split_seed": base_config["system"]["seed"],
        "input_resolution": list(base_config["data"]["input_resolution"]),
        "num_prototypes": base_config["model"]["num_prototypes_k"],
        "feature_dim": base_config["model"]["feature_dim"],
        "sam_checkpoint": base_config["model"]["sam3_checkpoint"],
    }
    if hasattr(train_dataset, "cases"):
        expected["case_ids"] = sorted(train_dataset.cases)
    if "isles" in dataset_config["name"].lower():
        expected["modalities"] = dataset_config["channels"]
        expected["validation_fraction"] = dataset_config["validation_fraction"]
        expected["test_fraction"] = dataset_config["test_fraction"]

    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prototype bank provenance does not match this run: {mismatches}")


def checkpoint_metadata(dataset_config: dict, base_config: dict, train_dataset) -> dict:
    """Build serializable metadata tying checkpoints to their data split and model."""
    return {
        "dataset": dataset_config["name"],
        "split_seed": base_config["system"]["seed"],
        "training_case_ids": sorted(train_dataset.cases) if hasattr(train_dataset, "cases") else [],
        "case_splits": getattr(train_dataset, "case_splits", {}),
        "input_resolution": list(base_config["data"]["input_resolution"]),
        "num_prototypes": base_config["model"]["num_prototypes_k"],
        "feature_dim": base_config["model"]["feature_dim"],
        "sam_checkpoint": base_config["model"]["sam3_checkpoint"],
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: HPMAModel,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
) -> dict[str, float]:
    """Evaluate validation samples without passing labels or masks to the model."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        predictions = model(images)["pred_masks"]
        metrics.update(predictions, masks, labels)
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HPMA Model")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--data_config", type=str, default="configs/isles2022.yaml")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.data_config, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    seed = base_cfg["system"]["seed"]
    set_random_seed(seed)
    requested_device = base_cfg["system"].get("device", "cuda")
    device = resolve_device(requested_device)
    print(f"[*] Starting HPMA Training pipeline on device: {device}")

    # Datasets
    train_dataset = build_dataset(
        dataset_config=data_cfg["dataset"],
        split="train",
        target_size=tuple(base_cfg["data"]["input_resolution"]),
        seed=base_cfg["system"]["seed"],
        mean=tuple(base_cfg["data"]["mean"]),
        std=tuple(base_cfg["data"]["std"]),
    )
    case_sampler = (
        train_dataset.grouped_sampler(seed=base_cfg["system"]["seed"])
        if hasattr(train_dataset, "grouped_sampler")
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_cfg["training"]["batch_size"],
        shuffle=case_sampler is None,
        sampler=case_sampler,
        num_workers=base_cfg["system"]["num_workers"],
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    validation_dataset = build_dataset(
        dataset_config=data_cfg["dataset"],
        split="val",
        target_size=tuple(base_cfg["data"]["input_resolution"]),
        seed=seed,
        mean=tuple(base_cfg["data"]["mean"]),
        std=tuple(base_cfg["data"]["std"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_cfg["system"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # Memory Bank
    memory_bank = PrototypeMemoryBank(
        bank_path=data_cfg["memory"]["bank_path"],
        num_classes=data_cfg["dataset"]["num_classes"],
        num_prototypes=base_cfg["model"]["num_prototypes_k"],
        dim=base_cfg["model"]["feature_dim"],
    )
    validate_memory_bank(memory_bank, data_cfg["dataset"], base_cfg, train_dataset)

    # Model
    model = HPMAModel(
        sam3_checkpoint_path=base_cfg["model"]["sam3_checkpoint"],
        feature_dim=base_cfg["model"]["feature_dim"],
        num_classes=data_cfg["dataset"]["num_classes"],
        input_channels=base_cfg["model"].get("input_channels", 3),
        visual_token_grid=tuple(base_cfg["model"].get("visual_token_grid", [4, 4])),
        prototype_bank=memory_bank,
        unfreeze_vision_blocks=base_cfg["model"].get("unfreeze_vision_blocks", 4),
        unfreeze_fusion_layers=base_cfg["model"].get("unfreeze_fusion_layers", 2),
    ).to(device)

    # Loss & Optimizer
    criterion = HPMATotalLoss(
        lambda_align=base_cfg["training"]["loss_weights"]["lambda_align"],
        local_adapter=model.local_adapter,
    ).to(device)

    optimizer = build_hpma_optimizer(
        model=model,
        base_lr=base_cfg["training"]["learning_rate"]["base"],
        vision_lr=base_cfg["training"]["learning_rate"]["vision_backbone"],
        weight_decay=base_cfg["training"]["weight_decay"],
    )
    mixed_precision = base_cfg["training"].get("mixed_precision", False)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision and torch.device(device).type == "cuda",
    )

    epochs = base_cfg["training"]["epochs"]
    logger = SimpleLogger(log_path="outputs/training.log")

    print(f"[*] Training for {epochs} epochs ...")
    best_validation_dice = float("-inf")
    output_stem = f"checkpoints/hpma_{data_cfg['dataset']['name'].lower()}"
    model_metadata = checkpoint_metadata(data_cfg["dataset"], base_cfg, train_dataset)
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(1, epochs + 1):
        if case_sampler is not None:
            case_sampler.set_epoch(epoch)
        loss_val = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=base_cfg["training"]["gradient_accumulation_steps"],
            mixed_precision=mixed_precision,
            scaler=scaler,
        )
        validation_metrics = evaluate_one_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            num_classes=data_cfg["dataset"]["num_classes"],
        )
        logger.log(
            f"[Epoch {epoch:02d}/{epochs}] Train Loss: {loss_val:.4f} | "
            f"Val Dice: {validation_metrics['mdice']:.2f}% | "
            f"Val IoU: {validation_metrics['iou']:.2f}%"
        )
        if validation_metrics["mdice"] > best_validation_dice:
            best_validation_dice = validation_metrics["mdice"]
            torch.save(
                {"state_dict": model.state_dict(), "metadata": model_metadata},
                f"{output_stem}_best.pt",
            )

    # Save final model
    save_path = f"{output_stem}_last.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": model_metadata}, save_path)
    print(f"[✓] Training complete. Model weights saved to {save_path}")


if __name__ == "__main__":
    main()
