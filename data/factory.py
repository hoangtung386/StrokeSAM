"""Dataset construction helpers used by training, evaluation, and memory building."""

from typing import Any, Callable, Dict, Optional

from torch.utils.data import Dataset

from data.dataset_endovis import EndoVisDataset
from data.dataset_isles2d import ISLES2DDataset
from data.transforms import ImageMaskTransform


def build_dataset(
    dataset_config: Dict[str, Any],
    split: str,
    target_size: tuple[int, int] = (1008, 1008),
    seed: int = 42,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    transform: Optional[Callable] = None,
) -> Dataset:
    """Build a supported dataset from its YAML ``dataset`` configuration."""
    dataset_name = dataset_config["name"].lower()
    data_root = dataset_config["data_root"]
    if "isles" in dataset_name:
        return ISLES2DDataset(
            data_root=data_root,
            split=split,
            target_size=target_size,
            modalities=tuple(dataset_config.get("channels", ISLES2DDataset.modalities)),
            seed=seed,
            validation_fraction=dataset_config.get("validation_fraction", 0.15),
            test_fraction=dataset_config.get("test_fraction", 0.15),
        )
    return EndoVisDataset(
        data_root=data_root,
        split=split,
        transform=transform
        or ImageMaskTransform(
            target_size=target_size,
            mean=mean,
            std=std,
        ),
    )
