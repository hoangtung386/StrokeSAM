"""Case-wise ISLES-2022 loader that exposes registered 2D MRI slices."""

import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.transforms import resize_and_pad


class CaseGroupedSampler(Sampler[int]):
    """Shuffle cases while keeping each case's slices together in the epoch."""

    def __init__(self, groups: Dict[str, List[int]], seed: int = 42) -> None:
        self.groups = groups
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        case_ids = list(self.groups)
        rng.shuffle(case_ids)
        for case_id in case_ids:
            indices = self.groups[case_id].copy()
            rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.groups.values())


class ISLES2DDataset(Dataset):
    """Read DWI, ADC, FLAIR and lesion masks as case-isolated 2D samples."""

    modalities = ("dwi", "adc", "FLAIR")

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (1008, 1008),
        modalities: Tuple[str, ...] = ("dwi", "adc", "FLAIR"),
        seed: int = 42,
        validation_fraction: float = 0.10,
        test_fraction: float = 0.10,
        cache_cases: int = 1,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', or 'test'.")
        if validation_fraction <= 0 or test_fraction <= 0:
            raise ValueError("Validation and test fractions must be positive.")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("Validation and test fractions must sum to less than one.")
        if len(modalities) != 3 or set(modalities) != set(self.modalities):
            raise ValueError("ISLES-2022 requires the dwi, adc, and FLAIR channels exactly once.")

        self.data_root = Path(data_root)
        self.target_size = target_size
        self.modalities = modalities
        self.cache_cases = max(cache_cases, 1)
        self._volume_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.incomplete_cases: List[str] = []

        cases = self._index_cases()
        if self.incomplete_cases:
            logging.warning(
                "Skipped %d ISLES cases missing one or more modalities/masks (examples: %s).",
                len(self.incomplete_cases),
                ", ".join(self.incomplete_cases[:5]),
            )
        if not cases:
            raise FileNotFoundError(f"No complete ISLES-2022 cases found under '{data_root}'.")

        case_ids = sorted(cases)
        random.Random(seed).shuffle(case_ids)
        test_count = max(1, round(len(case_ids) * test_fraction))
        val_count = max(1, round(len(case_ids) * validation_fraction))
        if test_count + val_count >= len(case_ids):
            raise ValueError("Not enough cases for non-empty train, validation, and test splits.")
        splits = {
            "test": case_ids[:test_count],
            "val": case_ids[test_count : test_count + val_count],
            "train": case_ids[test_count + val_count :],
        }
        self.case_splits = {key: sorted(value) for key, value in splits.items()}
        self.cases = {case_id: cases[case_id] for case_id in splits[split]}

        self.samples: List[Tuple[str, int]] = []
        self.case_groups: Dict[str, List[int]] = {}
        for case_id, paths in self.cases.items():
            reference = nib.as_closest_canonical(nib.load(str(paths["dwi"])))
            indices = []
            for slice_index in range(reference.shape[2]):
                indices.append(len(self.samples))
                self.samples.append((case_id, slice_index))
            self.case_groups[case_id] = indices

    def _index_cases(self) -> Dict[str, Dict[str, Path]]:
        cases: Dict[str, Dict[str, Path]] = {}
        for dwi_path in self.data_root.glob("sub-*/ses-*/dwi/*_dwi.nii.gz"):
            case_id = dwi_path.parents[2].name
            session_id = dwi_path.parents[1].name
            dwi_stem = dwi_path.name.removesuffix("_dwi.nii.gz")
            adc_path = dwi_path.with_name(f"{dwi_stem}_adc.nii.gz")
            case_derivatives = self.data_root / "derivatives" / case_id / session_id
            mask_path = case_derivatives / f"{dwi_stem}_msk.nii.gz"
            flair_path = dwi_path.parents[1] / "anat" / f"{case_id}_{session_id}_FLAIR.nii.gz"
            paths = {
                "dwi": dwi_path,
                "adc": adc_path,
                "FLAIR": flair_path,
                "mask": mask_path,
            }
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                self.incomplete_cases.append(f"{case_id} ({', '.join(missing)})")
                continue
            cases[case_id] = paths
        return cases

    @staticmethod
    def _same_grid(
        left: nib.spatialimages.SpatialImage,
        right: nib.spatialimages.SpatialImage,
    ) -> bool:
        return left.shape == right.shape and np.allclose(left.affine, right.affine, atol=1e-4)

    @classmethod
    def _load_aligned(
        cls,
        path: Path,
        reference: nib.spatialimages.SpatialImage,
        order: int,
    ) -> np.ndarray:
        image = nib.as_closest_canonical(nib.load(str(path)))
        if not cls._same_grid(image, reference):
            from nibabel.processing import resample_from_to

            image = resample_from_to(image, reference, order=order)
        return np.asarray(image.dataobj, dtype=np.float32)

    @staticmethod
    def _normalize_volume(volume: np.ndarray) -> np.ndarray:
        finite = np.isfinite(volume)
        foreground = finite & (volume != 0)
        if not np.any(foreground):
            return np.zeros_like(volume, dtype=np.float32)
        values = volume[foreground]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            std = 1.0
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[foreground] = (volume[foreground] - mean) / std
        return np.clip(normalized, -5.0, 5.0).astype(np.float32) / 5.0

    def _load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if case_id in self._volume_cache:
            self._volume_cache.move_to_end(case_id)
            return self._volume_cache[case_id]

        paths = self.cases[case_id]
        reference = nib.as_closest_canonical(nib.load(str(paths["dwi"])))
        modalities = []
        for name in self.modalities:
            if name == "dwi":
                volume = np.asarray(reference.dataobj, dtype=np.float32)
            else:
                volume = self._load_aligned(paths[name], reference, order=1)
            modalities.append(self._normalize_volume(volume))

        mask = self._load_aligned(paths["mask"], reference, order=0)
        images = np.stack(modalities, axis=0)
        masks = (mask > 0).astype(np.float32)
        self._volume_cache[case_id] = (images, masks)
        self._volume_cache.move_to_end(case_id)
        while len(self._volume_cache) > self.cache_cases:
            self._volume_cache.popitem(last=False)
        return images, masks

    def grouped_sampler(self, seed: int = 42) -> CaseGroupedSampler:
        """Return a sampler that prevents random slice access across cases."""
        return CaseGroupedSampler(self.case_groups, seed=seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        case_id, slice_index = self.samples[index]
        images, masks = self._load_case(case_id)

        image = torch.from_numpy(images[:, :, :, slice_index].copy())
        mask = torch.from_numpy(masks[:, :, slice_index].copy()).unsqueeze(0)
        image = resize_and_pad(image, self.target_size, mode="bilinear")
        mask = resize_and_pad(mask, self.target_size, mode="nearest")
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor([0], dtype=torch.long),
            "sample_id": f"{case_id}_z{slice_index:03d}",
        }
