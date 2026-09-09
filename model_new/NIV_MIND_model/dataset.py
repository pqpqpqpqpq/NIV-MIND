"""Data loading for end-to-end NIV-MIND training."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


FAILURE_FILENAME = "data_failure.pkl"
SUCCESS_FILENAME = "data_success.pkl"


def _load_pickle(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, list):
        raise TypeError(f"Expected a list in {path}, got {type(samples).__name__}")
    if not all(isinstance(sample, Mapping) for sample in samples):
        raise TypeError(f"Every sample in {path} must be a mapping")
    return samples


def load_labeled_samples(
    data_dir: str | Path,
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    """Load failure samples as class 1 and success samples as class 0."""
    root = Path(data_dir)
    failure = _load_pickle(root / FAILURE_FILENAME)
    success = _load_pickle(root / SUCCESS_FILENAME)
    samples = failure + success
    labels = np.concatenate(
        [
            np.ones(len(failure), dtype=np.int64),
            np.zeros(len(success), dtype=np.int64),
        ]
    )
    return samples, labels


def _fine_tensor(sample: Mapping[str, Any], index: int) -> torch.Tensor:
    if "fine" not in sample:
        raise KeyError(f"Sample {index} has no 'fine' field")
    fine = np.asarray(sample["fine"])
    if fine.ndim != 2 or fine.shape[0] != 600 or fine.shape[1] not in (8, 9):
        raise ValueError(
            f"Sample {index} fine must be [600,8] or [600,9], got {fine.shape}"
        )
    if fine.shape[1] == 9:
        fine = fine[:, :8]
    try:
        fine = np.asarray(fine, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Sample {index} fine contains non-numeric values") from error
    if not np.isfinite(fine).all():
        raise ValueError(f"Sample {index} fine contains NaN or infinity")
    return torch.from_numpy(np.ascontiguousarray(fine))


def _coarse_tensor(sample: Mapping[str, Any], index: int) -> torch.Tensor:
    if "coarse" not in sample:
        raise KeyError(f"Sample {index} has no 'coarse' field")
    coarse = np.asarray(sample["coarse"], dtype=np.float32).squeeze()
    if coarse.shape != (20,):
        raise ValueError(f"Sample {index} coarse must be [20], got {coarse.shape}")
    if not np.isfinite(coarse).all():
        raise ValueError(f"Sample {index} coarse contains NaN or infinity")
    return torch.from_numpy(np.ascontiguousarray(coarse))


class NIVMINDDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Validated fixed-length fine/coarse sample pairs."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        labels: Sequence[int] | np.ndarray,
        *,
        training: bool = False,
        fine_noise_std: float = 0.01,
        coarse_feature_dropout: float = 0.1,
    ) -> None:
        if len(samples) != len(labels):
            raise ValueError("samples and labels must have the same length")
        if fine_noise_std < 0:
            raise ValueError("fine_noise_std must be non-negative")
        if not 0.0 <= coarse_feature_dropout < 1.0:
            raise ValueError("coarse_feature_dropout must be in [0, 1)")

        self.fine = [_fine_tensor(sample, i) for i, sample in enumerate(samples)]
        self.coarse = [
            _coarse_tensor(sample, i) for i, sample in enumerate(samples)
        ]
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


__all__ = [
    "FAILURE_FILENAME",
    "NIVMINDDataset",
    "SUCCESS_FILENAME",
    "load_labeled_samples",
]
