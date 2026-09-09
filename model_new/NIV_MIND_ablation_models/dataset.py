"""Validated data pipeline shared by all NIV-MIND ablation groups."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import ExperimentSpec, transform_fine


FAILURE_FILENAME = "data_failure.pkl"
SUCCESS_FILENAME = "data_success.pkl"


def _read(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("rb") as handle:
        values = pickle.load(handle)
    if not isinstance(values, list) or not all(
        isinstance(value, Mapping) for value in values
    ):
        raise TypeError(f"{path} must contain a list of sample mappings")
    return values


def load_labeled_samples(
    data_dir: str | Path,
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    root = Path(data_dir)
    failure = _read(root / FAILURE_FILENAME)
    success = _read(root / SUCCESS_FILENAME)
    labels = np.concatenate(
        [
            np.ones(len(failure), dtype=np.int64),
            np.zeros(len(success), dtype=np.int64),
        ]
    )
    return failure + success, labels


def _numeric_fine(sample: Mapping[str, Any], index: int) -> np.ndarray:
    if "fine" not in sample:
        raise KeyError(f"Sample {index} has no fine field")
    fine = np.asarray(sample["fine"])
    if fine.ndim != 2 or fine.shape[0] not in (600, 1200):
        raise ValueError(
            f"Sample {index} fine must contain 600 or 1200 rows, got {fine.shape}"
        )
    if fine.shape[1] not in (8, 9):
            raise ValueError(
                f"Sample {index} fine must contain 8 numeric columns and an optional timestamp"
            )
    fine = fine[:, :8]
    try:
        fine = np.asarray(fine, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Sample {index} fine contains non-numeric values") from error
    if not np.isfinite(fine).all():
        raise ValueError(f"Sample {index} fine contains NaN or infinity")
    return fine


def _numeric_coarse(sample: Mapping[str, Any], index: int) -> np.ndarray:
    if "coarse" not in sample:
        raise KeyError(f"Sample {index} has no coarse field")
    coarse = np.asarray(sample["coarse"], dtype=np.float32).squeeze()
    if coarse.shape != (20,):
        raise ValueError(f"Sample {index} coarse must be [20], got {coarse.shape}")
    if not np.isfinite(coarse).all():
        raise ValueError(f"Sample {index} coarse contains NaN or infinity")
    return coarse


class AblationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        labels: Sequence[int] | np.ndarray,
        spec: ExperimentSpec,
        *,
        training: bool = False,
        fine_noise_std: float = 0.01,
        coarse_feature_dropout: float = 0.1,
    ) -> None:
        if len(samples) != len(labels):
            raise ValueError("samples and labels must have equal lengths")
        raw_fine = np.stack(
            [_numeric_fine(sample, i) for i, sample in enumerate(samples)]
        )
        fine = transform_fine(raw_fine, spec)
        coarse = np.stack(
            [_numeric_coarse(sample, i) for i, sample in enumerate(samples)]
        )
        self.fine = torch.from_numpy(fine)
        self.coarse = torch.from_numpy(coarse)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if not torch.all((self.labels == 0) | (self.labels == 1)):
            raise ValueError("labels must contain only 0 and 1")
        self.training = training
        self.fine_noise_std = fine_noise_std
        self.coarse_feature_dropout = coarse_feature_dropout

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = self.fine[index]
        coarse = self.coarse[index]
        if self.training:
            fine = fine.clone()
            coarse = coarse.clone()
            if self.fine_noise_std > 0:
                fine[:, 3:] += torch.randn_like(fine[:, 3:]) * self.fine_noise_std
            if self.coarse_feature_dropout > 0:
                keep = torch.rand_like(coarse) >= self.coarse_feature_dropout
                coarse *= keep
        return fine, coarse, self.labels[index]


__all__ = ["AblationDataset", "load_labeled_samples"]
