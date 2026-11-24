"""Evaluate an image-only HPMA checkpoint.

Computes 2D segmentation metrics without passing labels to the model.
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from data.factory import build_dataset
from memory.prototype_bank import PrototypeMemoryBank
from models.hpma_model import HPMAModel
from utils.metrics import SegmentationMetrics


@torch.no_grad()
def evaluate_dataset(
    model: HPMAModel,
    dataloader: DataLoader,
    device: str,
    num_classes: int = 1,
) -> dict:
    model.eval()
    metric_calc = SegmentationMetrics(num_classes=num_classes)

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images)
        pred_masks = outputs["pred_masks"]

        metric_calc.update(pred_masks, gt_masks, labels, threshold=0.5)

    return metric_calc.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HPMA Model")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--data_config", type=str, default="configs/isles2022.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.data_config, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    requested_device = base_cfg["system"].get("device", "cuda")
    device = requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    print(
        f"[*] Evaluating HPMA on {data_cfg['dataset']['name']} "
        f"split={args.split} ({device}) ..."
    )

    evaluation_dataset = build_dataset(
        dataset_config=data_cfg["dataset"],
        split=args.split,
        target_size=tuple(base_cfg["data"]["input_resolution"]),
        seed=base_cfg["system"]["seed"],
        mean=tuple(base_cfg["data"]["mean"]),
        std=tuple(base_cfg["data"]["std"]),
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_cfg["system"]["num_workers"],
    )

    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("Expected a checkpoint dictionary with model and provenance metadata.")
    checkpoint_metadata = checkpoint_payload.get("metadata")
    checkpoint = checkpoint_payload.get("state_dict", checkpoint_payload.get("model"))
    if not isinstance(checkpoint, dict) or "prototype_bank.bank" not in checkpoint:
        raise ValueError("The checkpoint must contain the model state and prototype memory bank.")
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("The checkpoint has no run metadata; retrain with the current pipeline.")

    expected_metadata = {
        "dataset": data_cfg["dataset"]["name"],
        "split_seed": base_cfg["system"]["seed"],
        "input_resolution": list(base_cfg["data"]["input_resolution"]),
        "num_prototypes": base_cfg["model"]["num_prototypes_k"],
        "feature_dim": base_cfg["model"]["feature_dim"],
        "sam_checkpoint": base_cfg["model"]["sam3_checkpoint"],
    }
    if "isles" in data_cfg["dataset"]["name"].lower():
        expected_metadata["case_splits"] = evaluation_dataset.case_splits
    mismatches = {
        key: (checkpoint_metadata.get(key), value)
        for key, value in expected_metadata.items()
        if checkpoint_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint metadata does not match this evaluation: {mismatches}")

    memory_bank = PrototypeMemoryBank(
        bank_tensor=checkpoint["prototype_bank.bank"],
        num_classes=data_cfg["dataset"]["num_classes"],
        num_prototypes=base_cfg["model"]["num_prototypes_k"],
        dim=base_cfg["model"]["feature_dim"],
    )

    model = HPMAModel(
        sam3_checkpoint_path=base_cfg["model"]["sam3_checkpoint"],
        feature_dim=base_cfg["model"]["feature_dim"],
        num_classes=data_cfg["dataset"]["num_classes"],
        input_channels=base_cfg["model"].get("input_channels", 3),
        visual_token_grid=tuple(base_cfg["model"].get("visual_token_grid", [4, 4])),
        prototype_bank=memory_bank,
        num_prototypes=base_cfg["model"]["num_prototypes_k"],
        unfreeze_vision_blocks=base_cfg["model"].get("unfreeze_vision_blocks", 4),
        unfreeze_fusion_layers=base_cfg["model"].get("unfreeze_fusion_layers", 2),
    ).to(device)
    model.load_state_dict(checkpoint)

    results = evaluate_dataset(
        model=model,
        dataloader=evaluation_loader,
        device=device,
        num_classes=data_cfg["dataset"]["num_classes"],
    )

    print("\n" + "=" * 50)
    print("EVALUATION METRICS:")
    print("=" * 50)
    print(f"IoU:           {results.get('iou', 0.0):.2f}%")
    print(f"mc IoU:        {results.get('mc_iou', 0.0):.2f}%")
    print(f"mDice:         {results.get('mdice', 0.0):.2f}%")
    print(f"Precision:     {results.get('precision', 0.0):.2f}%")
    print(f"Recall:        {results.get('recall', 0.0):.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
