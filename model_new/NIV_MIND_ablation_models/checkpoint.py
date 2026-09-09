"""Checkpoint loading for the ablation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass
class CheckpointReport:
    path: str
    loaded_keys: int
    target_keys: int
    coverage: float
    ready_for_evaluation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrictCheckpointMixin:
    """Load a tensor state dictionary for the selected variant."""

    def load_final_checkpoint(
        self: nn.Module,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointReport:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location, weights_only=True)
        if (
            not isinstance(state, Mapping)
            or not state
            or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in state.items()
            )
        ):
            raise TypeError("Checkpoint must be a non-empty tensor-only state dictionary")
        self.load_state_dict(dict(state), strict=True)
        target_keys = len(self.state_dict())
        return CheckpointReport(
            path=str(path),
            loaded_keys=len(state),
            target_keys=target_keys,
            coverage=len(state) / target_keys,
            ready_for_evaluation=True,
        )

    @torch.inference_mode()
    def predict_proba(
        self: nn.Module,
        fine: torch.Tensor,
        coarse: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(self(fine, coarse), dim=1)


__all__ = ["CheckpointReport", "StrictCheckpointMixin"]
