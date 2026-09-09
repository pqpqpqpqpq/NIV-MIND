"""Experiment registry and model factory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch.nn as nn

from .compact import (
    ARCHITECTURE_VARIANTS,
    SAMPLING_VARIANTS,
    WINDOW_VARIANTS,
    CompactAblationModel,
)


@dataclass
class ExperimentSpec:
    group: str
    variant: str
    family: str
    seq_len: int
    num_frames: int
    description: str

    @property
    def relative_weight_dir(self) -> Path:
        return Path(self.group) / self.variant


def _build_specs() -> dict[str, dict[str, ExperimentSpec]]:
    groups: dict[str, dict[str, ExperimentSpec]] = {
        "architecture": {},
        "sampling": {},
        "window": {},
        "compact_baseline": {},
    }
    for variant in ARCHITECTURE_VARIANTS:
        groups["architecture"][variant] = ExperimentSpec(
            "architecture",
            variant,
            "compact",
            600,
            60,
            f"Compact architecture variant: {variant}",
        )
    for variant, length in SAMPLING_VARIANTS.items():
        groups["sampling"][variant] = ExperimentSpec(
            "sampling",
            variant,
            "compact",
            length,
            max(length // 10, 4),
            f"Uniformly sample the 600-point sequence to {length} points",
        )
    for variant, length in WINDOW_VARIANTS.items():
        groups["window"][variant] = ExperimentSpec(
            "window",
            variant,
            "compact",
            length,
            max(length // 10, 10),
            f"Use the final {length} points of the observation window",
        )
    groups["compact_baseline"]["full"] = ExperimentSpec(
        "compact_baseline",
        "full",
        "compact",
        600,
        60,
        "Full compact architecture at 600 points",
    )
    return groups


EXPERIMENTS = _build_specs()


def get_spec(group: str, variant: str) -> ExperimentSpec:
    if group not in EXPERIMENTS:
        raise ValueError(f"Unknown group {group!r}; choose from {tuple(EXPERIMENTS)}")
    if variant not in EXPERIMENTS[group]:
        raise ValueError(
            f"Unknown variant {variant!r} for {group}; "
            f"choose from {tuple(EXPERIMENTS[group])}"
        )
    return EXPERIMENTS[group][variant]


def build_model(group: str, variant: str) -> nn.Module:
    spec = get_spec(group, variant)
    architecture_variant = variant if group == "architecture" else "full"
    return CompactAblationModel(
        seq_len=spec.seq_len,
        num_frames=spec.num_frames,
        architecture_variant=architecture_variant,
        backend_dropout=0.2 if group == "architecture" else 0.1,
    )


def transform_fine(fine: np.ndarray, spec: ExperimentSpec) -> np.ndarray:
    """Apply the study-specific temporal transformation."""
    fine = np.asarray(fine, dtype=np.float32)
    if fine.ndim != 3 or fine.shape[2] != 8:
        raise ValueError(f"fine must be [N,T,8], got {fine.shape}")
    if spec.group == "sampling":
        if spec.seq_len == 1200:
            if fine.shape[1] != 1200:
                raise ValueError(
                    "sample_1200 requires source data with 1200 one-second points"
                )
            return np.ascontiguousarray(fine)
        if fine.shape[1] != 600:
            raise ValueError("sampling variants below 600 points require 600-point input")
        step = fine.shape[1] / spec.seq_len
        indices = np.round(np.arange(spec.seq_len) * step).astype(np.int64)
        indices = np.clip(indices, 0, fine.shape[1] - 1)
        return np.ascontiguousarray(fine[:, indices, :])
    if spec.group == "window":
        if fine.shape[1] != 600:
            raise ValueError("window variants require 600-point input")
        return np.ascontiguousarray(fine[:, -spec.seq_len :, :])
    if fine.shape[1] != 600:
        raise ValueError("this variant requires 600-point input")
    return np.ascontiguousarray(fine)


__all__ = [
    "EXPERIMENTS",
    "ExperimentSpec",
    "build_model",
    "get_spec",
    "transform_fine",
]
