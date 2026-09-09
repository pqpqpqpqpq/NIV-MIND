"""Compact NIV-MIND variants for architecture, sampling, and window studies."""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange

from .backbone import Block, TrainableSTGCN_FeaturePreserving
from .checkpoint import StrictCheckpointMixin


ARCHITECTURE_VARIANTS = ("only_st", "only_ts", "no_altformer", "no_stgcn")
SAMPLING_VARIANTS = {
    "sample_1200": 1200,
    "sample_240": 240,
    "sample_120": 120,
    "sample_40": 40,
}
WINDOW_VARIANTS = {"window_300": 300, "window_150": 150}


class NoGraphFrontend(nn.Module):
    """Per-channel temporal frontend without graph communication."""

    def __init__(
        self,
        num_features: int = 8,
        feature_dim: int = 64,
        num_blocks: int = 3,
        temporal_kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        mid_dim = max(8, feature_dim // 4)
        self.input_projection = nn.Sequential(nn.Linear(1, mid_dim), nn.ReLU())
        pad = (temporal_kernel_size - 1) // 2
        blocks = []
        in_channels = mid_dim
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        feature_dim,
                        temporal_kernel_size,
                        padding=pad,
                    ),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(
                        feature_dim,
                        feature_dim,
                        temporal_kernel_size,
                        padding=pad,
                    ),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                )
            )
            in_channels = feature_dim
        self.tcn_blocks = nn.ModuleList(blocks)
        self.feature_enhance = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                )
                for _ in range(num_features)
            ]
        )
        self.temporal_attention = nn.MultiheadAttention(
            feature_dim,
            4,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, features = x.shape
        x = x.unsqueeze(-1).permute(0, 2, 1, 3).reshape(
            batch * features,
            length,
            1,
        )
        x = self.input_projection(x).transpose(1, 2)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2).reshape(batch, features, length, -1)
        x = x.permute(0, 2, 1, 3)
        enhanced = []
        for index in range(features):
            feature = self.feature_enhance[index](x[:, :, index, :])
            attended, _ = self.temporal_attention(feature, feature, feature)
            enhanced.append(self.output_norm(feature + 0.5 * attended))
        return torch.stack(enhanced, dim=2)


class TokenAltFormer(nn.Module):
    """Return AltFormer time tokens directly for downstream attention."""

    def __init__(
        self,
        *,
        num_frames: int,
        branch: str = "both",
        num_features: int = 8,
        feature_dim: int = 64,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if branch not in ("both", "only_st", "only_ts"):
            raise ValueError(f"Unknown branch: {branch}")
        self.branch = branch
        self.num_frame = num_frames
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.feature_projection = nn.Linear(feature_dim, embed_dim)

        if branch in ("both", "only_st"):
            self.st_spatial_pos_embed = nn.Parameter(
                torch.zeros(1, num_features, embed_dim)
            )
            self.st_spatial_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=dropout,
                        attn_drop=dropout,
                        drop_path=0.2 * index / depth,
                        norm_layer=norm,
                    )
                    for index in range(depth // 2)
                ]
            )
            self.st_temporal_pos_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim)
            )
            self.st_temporal_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=dropout,
                        attn_drop=dropout,
                        drop_path=0.2 * index / depth,
                        norm_layer=norm,
                    )
                    for index in range(depth // 2)
                ]
            )
            self.st_norm = norm(embed_dim)
            nn.init.trunc_normal_(self.st_spatial_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.st_temporal_pos_embed, std=0.02)

        if branch in ("both", "only_ts"):
            self.ts_temporal_pos_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim)
            )
            self.ts_temporal_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=dropout,
                        attn_drop=dropout,
                        drop_path=0.2 * index / depth,
                        norm_layer=norm,
                    )
                    for index in range(depth // 2)
                ]
            )
            self.ts_spatial_pos_embed = nn.Parameter(
                torch.zeros(1, num_features, embed_dim)
            )
            self.ts_spatial_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=dropout,
                        attn_drop=dropout,
                        drop_path=0.2 * index / depth,
                        norm_layer=norm,
                    )
                    for index in range(depth // 2)
                ]
            )
            self.ts_norm = norm(embed_dim)
            nn.init.trunc_normal_(self.ts_temporal_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.ts_spatial_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)
        input_dim = embed_dim * 2 if branch == "both" else embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            norm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if branch == "both":
            self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)

    def _st_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        spatial = rearrange(x, "b t f e -> (b t) f e")
        spatial = self.pos_drop(spatial + self.st_spatial_pos_embed)
        for block in self.st_spatial_blocks:
            spatial, _ = block(spatial)
        spatial = rearrange(spatial, "(b t) f e -> b t f e", b=batch)
        temporal = self.pos_drop(
            spatial.mean(dim=2) + self.st_temporal_pos_embed
        )
        for block in self.st_temporal_blocks:
            temporal, _ = block(temporal)
        return self.st_norm(temporal)

    def _ts_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time = x.shape[:2]
        temporal = rearrange(x, "b t f e -> (b f) t e")
        temporal = self.pos_drop(temporal + self.ts_temporal_pos_embed)
        for block in self.ts_temporal_blocks:
            temporal, _ = block(temporal)
        temporal = rearrange(temporal, "(b f) t e -> b t f e", b=batch)
        spatial = self.pos_drop(
            temporal.mean(dim=1) + self.ts_spatial_pos_embed
        )
        for block in self.ts_spatial_blocks:
            spatial, _ = block(spatial)
        pooled = self.ts_norm(spatial.mean(dim=1))
        return pooled.unsqueeze(1).expand(-1, time, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_projection(x)
        if self.branch == "both":
            st_tokens = self._st_forward(x)
            ts_tokens = self._ts_forward(x)
            weights = functional.softmax(self.fusion_weights, dim=0)
            tokens = torch.cat(
                [st_tokens * weights[0], ts_tokens * weights[1]], dim=-1
            )
        elif self.branch == "only_st":
            tokens = self._st_forward(x)
        else:
            tokens = self._ts_forward(x)
        return self.fusion(tokens)


class PerFrameMLPBackend(nn.Module):
    def __init__(self, num_features: int = 8, feature_dim: int = 64,
                 output_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_features * feature_dim, output_dim),
            nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, features, dim = x.shape
        return self.proj(x.reshape(batch, time, features * dim))


class CompactFineBackbone(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        num_frames: int,
        architecture_variant: str = "full",
        dropout: float = 0.2,
        backend_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if architecture_variant == "no_stgcn":
            self.frontend = NoGraphFrontend()
        else:
            self.frontend = TrainableSTGCN_FeaturePreserving(
                num_features=8,
                feature_dim=64,
                seq_len=seq_len,
                num_blocks=3,
                adj_mode="residual",
            )
        if architecture_variant == "no_altformer":
            self.backend = PerFrameMLPBackend(dropout=backend_dropout)
        else:
            branch = {
                "only_st": "only_st",
                "only_ts": "only_ts",
            }.get(architecture_variant, "both")
            self.backend = TokenAltFormer(
                num_frames=num_frames,
                branch=branch,
                dropout=backend_dropout,
            )
        self.num_frames = num_frames
        self.out_channels = 256
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.frontend(x)
        batch, length, channels, dim = features.shape
        if length % self.num_frames:
            pad = self.num_frames * math.ceil(length / self.num_frames) - length
            features = torch.cat(
                [features, features.new_zeros(batch, pad, channels, dim)], dim=1
            )
            length += pad
        points = length // self.num_frames
        frames = rearrange(
            features,
            "b (t p) f d -> b t p f d",
            t=self.num_frames,
            p=points,
        ).mean(dim=2)
        return self.dropout(self.backend(frames))


class CoarseEncoder(nn.Module):
    def __init__(self, in_dim: int = 20, hidden: int = 64,
                 out_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PostHocFiLM(nn.Module):
    def __init__(self, context_dim: int = 64, channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(context_dim, channels * 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(context).chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class CoarseQueryAttention(nn.Module):
    def __init__(self, fine_dim: int = 256, coarse_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.q_proj = nn.Linear(coarse_dim, fine_dim)
        self.attn = nn.MultiheadAttention(
            fine_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fine_dim)

    def forward(self, tokens: torch.Tensor,
                coarse: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(coarse).unsqueeze(1)
        output, _ = self.attn(query, tokens, tokens)
        return self.norm(output.squeeze(1))


class CompactAblationModel(StrictCheckpointMixin, nn.Module):
    """Compact fusion model used by architecture and temporal experiments."""

    def __init__(
        self,
        *,
        seq_len: int = 600,
        num_frames: int = 60,
        architecture_variant: str = "full",
        dropout: float = 0.2,
        backend_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if architecture_variant not in ("full",) + ARCHITECTURE_VARIANTS:
            raise ValueError(f"Unknown architecture variant: {architecture_variant}")
        self.seq_len = seq_len
        self.architecture_variant = architecture_variant
        self.coarse_encoder = CoarseEncoder(dropout=dropout)
        self.fine_backbone = CompactFineBackbone(
            seq_len=seq_len,
            num_frames=num_frames,
            architecture_variant=architecture_variant,
            dropout=dropout,
            backend_dropout=backend_dropout,
        )
        self.posthoc_film = PostHocFiLM()
        self.attn_pool = CoarseQueryAttention(dropout=dropout)
        self.head_fine = self._head(256, dropout)
        self.head_coarse = self._head(64, dropout)
        self.head_fused = self._head(320, dropout)
        self.gate = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )

    @staticmethod
    def _head(input_dim: int, dropout: float) -> nn.Sequential:
        hidden = max(input_dim // 2, 32)
        return nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    def forward_details(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if fine.ndim != 3 or tuple(fine.shape[1:]) != (self.seq_len, 8):
            raise ValueError(
                f"fine must be [batch,{self.seq_len},8], got {tuple(fine.shape)}"
            )
        if coarse.ndim != 2 or coarse.shape != (fine.shape[0], 20):
            raise ValueError(f"coarse must be [batch,20], got {tuple(coarse.shape)}")
        coarse_embedding = self.coarse_encoder(coarse)
        tokens = self.fine_backbone(fine)
        tokens = self.posthoc_film(tokens, coarse_embedding)
        fine_embedding = self.attn_pool(tokens, coarse_embedding)
        fine_logits = self.head_fine(fine_embedding)
        coarse_logits = self.head_coarse(coarse_embedding)
        fused_logits = self.head_fused(
            torch.cat([fine_embedding, coarse_embedding], dim=-1)
        )
        alpha = torch.sigmoid(self.gate(coarse_embedding))
        logits = alpha * fused_logits + (1.0 - alpha) * coarse_logits
        return {
            "logits": logits,
            "fine_logits": fine_logits,
            "coarse_logits": coarse_logits,
            "fused_logits": fused_logits,
        }

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return self.forward_details(fine, coarse)["logits"]

    def training_loss(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        auxiliary_weight: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        details = self.forward_details(fine, coarse)
        loss = criterion(details["logits"], labels)
        loss += auxiliary_weight * criterion(details["fine_logits"], labels)
        loss += auxiliary_weight * criterion(details["coarse_logits"], labels)
        return loss, details["logits"]


__all__ = [
    "ARCHITECTURE_VARIANTS",
    "SAMPLING_VARIANTS",
    "WINDOW_VARIANTS",
    "CompactAblationModel",
]
