"""Coarse encoder and cross-modal fusion layers for final NIV-MIND."""

from __future__ import annotations

import torch
import torch.nn as nn


class CoarseGrainedEncoder(nn.Module):
    """Encode the 20 coarse clinical features into a 128-D representation."""

    def __init__(self, input_dim: int = 20, embed_dim: int = 128,
                 drop_rate: float = 0.1) -> None:
        super().__init__()
        hidden_dim = embed_dim * 2
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder_blocks = nn.ModuleList([
            self._make_block(hidden_dim, hidden_dim, drop_rate),
            self._make_block(hidden_dim, hidden_dim, drop_rate),
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.residual_proj = (
            nn.Linear(input_dim, embed_dim)
            if input_dim != embed_dim
            else nn.Identity()
        )

    @staticmethod
    def _make_block(in_dim: int, out_dim: int, drop_rate: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        for block in self.encoder_blocks:
            hidden = hidden + block(hidden)
        return self.output_proj(hidden) + self.residual_proj(x)


class CrossModalAttentionFusion(nn.Module):
    """Bidirectional cross-attention followed by adaptive gated fusion."""

    def __init__(self, fine_dim: int, coarse_dim: int, fusion_dim: int,
                 num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.fine_proj = nn.Sequential(
            nn.Linear(fine_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.coarse_proj = nn.Sequential(
            nn.Linear(coarse_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.c2f_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.f2c_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.c2f_ffn = self._make_ffn(fusion_dim, dropout)
        self.f2c_ffn = self._make_ffn(fusion_dim, dropout)
        self.c2f_norm1 = nn.LayerNorm(fusion_dim)
        self.c2f_norm2 = nn.LayerNorm(fusion_dim)
        self.f2c_norm1 = nn.LayerNorm(fusion_dim)
        self.f2c_norm2 = nn.LayerNorm(fusion_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)

    @staticmethod
    def _make_ffn(dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fine_feat: torch.Tensor,
                coarse_feat: torch.Tensor) -> torch.Tensor:
        fine = self.fine_proj(fine_feat)
        coarse = self.coarse_proj(coarse_feat)
        fine_seq = fine.unsqueeze(1)
        coarse_seq = coarse.unsqueeze(1)

        coarse_from_fine, _ = self.c2f_attn(
            query=coarse_seq,
            key=fine_seq,
            value=fine_seq,
        )
        coarse_from_fine = self.c2f_norm1(coarse_seq + coarse_from_fine)
        coarse_from_fine = self.c2f_norm2(
            coarse_from_fine + self.c2f_ffn(coarse_from_fine)
        ).squeeze(1)

        fine_from_coarse, _ = self.f2c_attn(
            query=fine_seq,
            key=coarse_seq,
            value=coarse_seq,
        )
        fine_from_coarse = self.f2c_norm1(fine_seq + fine_from_coarse)
        fine_from_coarse = self.f2c_norm2(
            fine_from_coarse + self.f2c_ffn(fine_from_coarse)
        ).squeeze(1)

        gate = self.gate_net(
            torch.cat([fine_from_coarse, coarse_from_fine], dim=-1)
        )
        fused = gate * fine_from_coarse + (1.0 - gate) * coarse_from_fine
        return self.fusion_norm(fused + self.fusion_ffn(fused))


__all__ = ["CoarseGrainedEncoder", "CrossModalAttentionFusion"]
