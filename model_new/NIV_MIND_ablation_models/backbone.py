"""Shared final-model backbone layers."""

from model_new.NIV_MIND_model.backbone import (
    Block,
    TrainableSTGCN_FeaturePreserving,
)

__all__ = [
    "Block",
    "TrainableSTGCN_FeaturePreserving",
]
