"""NIV-MIND architecture used by training and inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from einops import rearrange

try:
    from .backbone import (
        Enhanced_STAltFormer_3D_NoMonitor,
        TrainableSTGCN_FeaturePreserving,
    )
    from .fusion import CoarseGrainedEncoder, CrossModalAttentionFusion
except ImportError:
    from backbone import (  # type: ignore
        Enhanced_STAltFormer_3D_NoMonitor,
        TrainableSTGCN_FeaturePreserving,
    )
    from fusion import CoarseGrainedEncoder, CrossModalAttentionFusion  # type: ignore


@dataclass
class CheckpointReport:
    path: str
    loaded_keys: int
    target_keys: int
    coverage: float
    ready_for_evaluation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NIV_MIND(nn.Module):
    """NIV-MIND network.

    ``fine`` must be ``[batch, 600, 8]`` and ``coarse`` must be
    ``[batch, 20]``. The output contains two logits per sample: class 0 is
    NIV success and class 1 is NIV failure.
    """

    def __init__(
        self,
        *,
        num_features: int = 8,
        feature_dim: int = 64,
        seq_len: int = 600,
        num_frames: int = 120,
        embed_dim: int = 256,
        fine_embed_dim: int = 512,
        coarse_input_dim: int = 20,
        coarse_embed_dim: int = 128,
        fusion_dim: int = 256,
        num_classes: int = 2,
        num_attn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features != 8:
            raise ValueError("NIV-MIND requires exactly 8 fine channels.")

        self.num_features = num_features
        self.seq_len = seq_len
        self.num_frames = num_frames
        self.coarse_input_dim = coarse_input_dim

        self.fine_feature_extractor = TrainableSTGCN_FeaturePreserving(
            num_features=num_features,
            feature_dim=feature_dim,
            seq_len=seq_len,
            num_blocks=3,
            adj_mode="residual",
        )
        self.fine_aggregator = Enhanced_STAltFormer_3D_NoMonitor(
            num_frame=num_frames,
            num_features=num_features,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            output_dim=fine_embed_dim,
        )
        self.coarse_encoder = CoarseGrainedEncoder(
            input_dim=coarse_input_dim,
            embed_dim=coarse_embed_dim,
            drop_rate=dropout,
        )
        self.fusion_layer = CrossModalAttentionFusion(
            fine_dim=fine_embed_dim,
            coarse_dim=coarse_embed_dim,
            fusion_dim=fusion_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def _validate_inputs(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
    ) -> None:
        if fine.ndim != 3 or tuple(fine.shape[1:]) != (
            self.seq_len,
            self.num_features,
        ):
            raise ValueError(
                f"fine must have shape [batch, {self.seq_len}, "
                f"{self.num_features}], got {tuple(fine.shape)}"
            )
        if (
            coarse.ndim != 2
            or coarse.shape[0] != fine.shape[0]
            or coarse.shape[1] != self.coarse_input_dim
        ):
            raise ValueError(
                f"coarse must have shape [batch, {self.coarse_input_dim}] "
                f"with the same batch size as fine, got {tuple(coarse.shape)}"
            )

    def _pad_and_pool(self, features: torch.Tensor) -> torch.Tensor:
        batch, length, channels, dim = features.shape
        if length % self.num_frames != 0:
            pad_len = (
                self.num_frames * math.ceil(length / self.num_frames) - length
            )
            features = torch.cat(
                [
                    features,
                    torch.zeros(
                        batch,
                        pad_len,
                        channels,
                        dim,
                        device=features.device,
                        dtype=features.dtype,
                    ),
                ],
                dim=1,
            )
            length = features.shape[1]
        points_per_frame = length // self.num_frames
        frames = rearrange(
            features,
            "b (t p) f d -> b t p f d",
            t=self.num_frames,
            p=points_per_frame,
        )
        return frames.mean(dim=2)

    def forward(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        self._validate_inputs(fine, coarse)
        fine_global = self.encode_fine(fine)
        coarse_global = self.encode_coarse(coarse)
        fused = self.fusion_layer(fine_global, coarse_global)
        return self.classifier(fused)

    def encode_fine(self, fine: torch.Tensor) -> torch.Tensor:
        """Encode a fine-grained sequence into its 512-dimensional vector."""
        if fine.ndim != 3 or tuple(fine.shape[1:]) != (
            self.seq_len,
            self.num_features,
        ):
            raise ValueError(
                f"fine must have shape [batch, {self.seq_len}, "
                f"{self.num_features}], got {tuple(fine.shape)}"
            )
        fine_local = self.fine_feature_extractor(fine)
        fine_frames = self._pad_and_pool(fine_local)
        return self.fine_aggregator(fine_frames)

    def encode_coarse(self, coarse: torch.Tensor) -> torch.Tensor:
        """Encode coarse clinical inputs into their feature vector."""
        if coarse.ndim != 2 or coarse.shape[1] != self.coarse_input_dim:
            raise ValueError(
                f"coarse must have shape [batch, {self.coarse_input_dim}], "
                f"got {tuple(coarse.shape)}"
            )
        return self.coarse_encoder(coarse)

    @torch.inference_mode()
    def predict_proba(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[P(success), P(failure)]`` for each sample."""
        return torch.softmax(self(fine, coarse), dim=1)

    def load_final_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointReport:
        """Strictly load a raw, tensor-only NIV-MIND state dictionary."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Final checkpoint not found: {path}")

        raw = torch.load(path, map_location=map_location, weights_only=True)
        if (
            not isinstance(raw, Mapping)
            or not raw
            or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in raw.items()
            )
        ):
            raise TypeError(
                "Final checkpoint must be a non-empty, tensor-only state dictionary."
            )

        state_dict = dict(raw)
        self.load_state_dict(state_dict, strict=True)
        target_keys = len(self.state_dict())
        return CheckpointReport(
            path=str(path),
            loaded_keys=len(state_dict),
            target_keys=target_keys,
            coverage=len(state_dict) / target_keys,
            ready_for_evaluation=True,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["NIV_MIND", CheckpointReport]:
        """Construct, load, move to a device, and enter evaluation mode."""
        target_device = torch.device(device)
        model = cls()
        report = model.load_final_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        model.to(target_device).eval()
        return model, report


NIVMindModel = NIV_MIND

__all__ = ["CheckpointReport", "NIV_MIND", "NIVMindModel"]
