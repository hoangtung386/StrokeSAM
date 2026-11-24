"""Dataset loader for EndoVis 2017 & 2018 surgical instrument datasets."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import ImageMaskTransform


class EndoVisDataset(Dataset):
    """EndoVis surgical instrument segmentation dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize EndoVisDataset.

        Args:
            data_root: Path to dataset folder.
            split: 'train' or 'val'.
            transform: Optional transform callable.
        """
        super().__init__()
        self.transform = transform or ImageMaskTransform()

        self.samples = self._index_dataset(data_root, split)
        if not self.samples:
            raise FileNotFoundError(
                f"No image-mask pairs found for split '{split}' under '{data_root}'."
            )

    @staticmethod
    def _index_dataset(data_root: str, split: str) -> List[Dict[str, str]]:
        """Index image-mask pairs and skip mask files encountered during traversal."""
        samples = []
        if not os.path.isdir(data_root):
            return samples

        split_dir = Path(data_root) / split
        if not split_dir.is_dir():
            return samples

        mask_directory_names = {"mask", "masks", "instruments_masks", "labels"}
        for root, _, files in os.walk(split_dir):
            root_path = Path(root)
            if any(part.lower() in mask_directory_names for part in root_path.parts):
                continue
            for filename in sorted(files):
                if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                image_path = root_path / filename
                image_text = str(image_path)
                mask_text = image_text.replace("images", "masks").replace(
                    "frames", "instruments_masks"
                )
                mask_path = Path(mask_text)
                if mask_path != image_path and mask_path.is_file():
                    samples.append({"image": str(image_path), "mask": str(mask_path)})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        sample_id = sample["image"]
        img = cv2.imread(sample["image"])
        if img is None:
            raise OSError(f"Could not read image: {sample['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Could not read mask: {sample['mask']}")

        binary_mask = (mask > 0).astype(np.uint8)

        img_t, mask_t = self.transform(img, binary_mask)
        return {
            "image": img_t,
            "mask": mask_t,
            "label": torch.zeros(1, dtype=torch.long),
            "sample_id": sample_id,
        }
