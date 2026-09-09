"""Neural comparator models used in the manuscript."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from model_new.NIV_MIND_ablation_models.checkpoint import StrictCheckpointMixin
from model_new.NIV_MIND_model.backbone import (
    Enhanced_STAltFormer_3D_NoMonitor,
    TrainableSTGCN_FeaturePreserving,
)


class NIVMINDC(StrictCheckpointMixin, nn.Module):
    """Clinical features combined with eight window-level ventilator means."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = 28
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_proj = nn.Linear(input_dim, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        if fine.ndim != 3 or fine.shape[2] != 8:
            raise ValueError(f"fine must be [B,T,8], got {tuple(fine.shape)}")
        if coarse.shape != (fine.shape[0], 20):
            raise ValueError(f"coarse must be [B,20], got {tuple(coarse.shape)}")
        combined = torch.cat([coarse, fine.mean(dim=1)], dim=1)
        encoded = self.encoder(combined) + self.residual_proj(combined)
        return self.classifier(encoded)


class NIVMINDF(StrictCheckpointMixin, nn.Module):
    """Fine-grained ST-GCN–AltFormer branch without clinical inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fine_feature_extractor = TrainableSTGCN_FeaturePreserving(
            num_features=8,
            feature_dim=64,
            seq_len=600,
            num_blocks=3,
            adj_mode="residual",
        )
        self.fine_aggregator = Enhanced_STAltFormer_3D_NoMonitor(
            num_frame=120,
            num_features=8,
            feature_dim=64,
            embed_dim=256,
            output_dim=512,
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    @staticmethod
    def _pool(features: torch.Tensor, frames: int = 120) -> torch.Tensor:
        batch, length, channels, dimension = features.shape
        if length % frames:
            padding = frames * math.ceil(length / frames) - length
            features = torch.cat(
                [features, features.new_zeros(batch, padding, channels, dimension)],
                dim=1,
            )
            length += padding
        points = length // frames
        return rearrange(
            features,
            "b (t p) f d -> b t p f d",
            t=frames,
            p=points,
        ).mean(dim=2)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        del coarse
        if fine.ndim != 3 or tuple(fine.shape[1:]) != (600, 8):
            raise ValueError(f"fine must be [B,600,8], got {tuple(fine.shape)}")
        local = self.fine_feature_extractor(fine)
        global_feature = self.fine_aggregator(self._pool(local))
        return self.classifier(self.fusion_layer(global_feature))


__all__ = ["NIVMINDC", "NIVMINDF"]
