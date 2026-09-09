"""Cross-validation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold


def stratified_folds(
    samples: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=np.int64)
    if len(samples) != labels.size:
        raise ValueError("samples and labels must have equal lengths")
    indices = np.arange(labels.size)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(indices, labels))


__all__ = ["stratified_folds"]
