"""Window eligibility, missing-value handling, and fold standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _interpolate(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    indices = np.arange(values.size)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("a channel contains no observed values")
    values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])
    return values


def _nearest_fill(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    indices = np.arange(values.size)
    valid_indices = indices[np.isfinite(values)]
    if valid_indices.size == 0:
        raise ValueError("a channel contains no observed values")
    for index in indices[~np.isfinite(values)]:
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        values[index] = values[nearest]
    return values


def clean_ventilator_window(
    fine: np.ndarray,
    *,
    max_missing_fraction: float = 0.15,
    setting_channels: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray | None:
    """Return a cleaned ``[T,8]`` window, or ``None`` if it is ineligible."""
    values = np.asarray(fine, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"fine must be [T,8], got {values.shape}")
    missing = np.mean(~np.isfinite(values), axis=0)
    if np.any(missing > max_missing_fraction):
        return None
    cleaned = values.copy()
    for channel in range(cleaned.shape[1]):
        cleaner = _nearest_fill if channel in setting_channels else _interpolate
        cleaned[:, channel] = cleaner(cleaned[:, channel])
    return cleaned.astype(np.float32)


@dataclass
class FoldStandardizer:
    fine_mean: np.ndarray
    fine_std: np.ndarray
    coarse_mean: np.ndarray
    coarse_std: np.ndarray

    @classmethod
    def fit(cls, fine: np.ndarray, coarse: np.ndarray) -> "FoldStandardizer":
        fine = np.asarray(fine, dtype=np.float64)
        coarse = np.asarray(coarse, dtype=np.float64)
        if fine.ndim != 3 or fine.shape[2] != 8:
            raise ValueError(f"fine must be [N,T,8], got {fine.shape}")
        if coarse.ndim != 2 or coarse.shape[0] != fine.shape[0]:
            raise ValueError("coarse must be [N,C] and match fine")
        fine_mean = np.nanmean(fine, axis=(0, 1))
        fine_std = np.nanstd(fine, axis=(0, 1))
        coarse_mean = np.nanmean(coarse, axis=0)
        coarse_std = np.nanstd(coarse, axis=0)
        fine_std[fine_std < 1e-8] = 1.0
        coarse_std[coarse_std < 1e-8] = 1.0
        return cls(fine_mean, fine_std, coarse_mean, coarse_std)

    def transform_fine(self, fine: np.ndarray) -> np.ndarray:
        values = np.asarray(fine, dtype=np.float64)
        values = np.where(np.isfinite(values), values, self.fine_mean)
        return ((values - self.fine_mean) / self.fine_std).astype(np.float32)

    def transform_coarse(self, coarse: np.ndarray) -> np.ndarray:
        values = np.asarray(coarse, dtype=np.float64)
        values = np.where(np.isfinite(values), values, self.coarse_mean)
        return ((values - self.coarse_mean) / self.coarse_std).astype(np.float32)


def interval_subsample(
    fine: np.ndarray,
    *,
    source_interval_seconds: float,
    target_interval_seconds: float,
) -> np.ndarray:
    values = np.asarray(fine)
    if values.ndim not in (2, 3):
        raise ValueError("fine must be [T,C] or [N,T,C]")
    if target_interval_seconds < source_interval_seconds:
        raise ValueError("target interval cannot be finer than the source interval")
    ratio = target_interval_seconds / source_interval_seconds
    length = values.shape[-2]
    target_length = int(round(length / ratio))
    indices = np.round(np.arange(target_length) * ratio).astype(np.int64)
    indices = np.clip(indices, 0, length - 1)
    return np.take(values, indices, axis=-2)


__all__ = [
    "FoldStandardizer",
    "clean_ventilator_window",
    "interval_subsample",
]
