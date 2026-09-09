from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from model_new.NIV_MIND_model import NIV_MIND
from model_new.NIV_MIND_model.analysis.engineered_features import (
    FEATURE_NAMES,
    extract_signal_features,
    extract_top_window_feature_matrix,
)
from model_new.NIV_MIND_model.analysis.landmarks import select_landmark_window
from model_new.NIV_MIND_model.analysis.metrics import (
    binary_metrics,
    decision_curve,
    paired_auc_difference_interval,
)
from model_new.NIV_MIND_model.analysis.preprocessing import (
    FoldStandardizer,
    clean_ventilator_window,
    interval_subsample,
)
from model_new.NIV_MIND_model.analysis.temporal_ablation import mean_replace
from model_new.NIV_MIND_model.analysis.temporal_attribution import (
    contribution_scores,
    top_windows,
)


def test_binary_metrics_and_decision_curve() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    result = binary_metrics(labels, scores)
    assert result["auc"] == 1.0
    assert result["auprc"] == 1.0
    assert len(decision_curve(labels, scores, thresholds=[0.2, 0.5])) == 2
    interval = paired_auc_difference_interval(
        labels,
        scores,
        [0.4, 0.3, 0.6, 0.5],
        samples=20,
    )
    assert interval["delta_auc"] >= 0.0


def test_all_41_signal_features_and_window_layout() -> None:
    signal = np.sin(np.linspace(0, 8 * np.pi, 600))
    values = extract_signal_features(signal)
    assert tuple(values) == FEATURE_NAMES
    assert len(values) == 41

    fine = np.tile(signal[:, None], (1, 8))[None]
    contribution = np.zeros((1, 40, 8), dtype=np.float32)
    matrix, names = extract_top_window_feature_matrix(fine, contribution)
    assert matrix.shape == (1, 2048)
    assert len(names) == 2048
    assert np.isfinite(matrix).all()


def test_window_cleaning_and_fold_standardization() -> None:
    fine = np.tile(np.arange(20, dtype=np.float32)[:, None], (1, 8))
    fine[3, 0] = np.nan
    fine[4, 3] = np.nan
    cleaned = clean_ventilator_window(fine)
    assert cleaned is not None
    assert np.isfinite(cleaned).all()
    assert cleaned[3, 0] in (2.0, 4.0)
    assert cleaned[4, 3] == pytest.approx(4.0)

    coarse = np.arange(40, dtype=np.float32).reshape(2, 20)
    standardizer = FoldStandardizer.fit(
        np.stack([cleaned, cleaned + 1]),
        coarse,
    )
    assert np.isfinite(standardizer.transform_fine(cleaned)).all()
    assert np.isfinite(standardizer.transform_coarse(coarse)).all()
    assert interval_subsample(fine, source_interval_seconds=1, target_interval_seconds=2).shape == (10, 8)


def test_landmark_selection_rules() -> None:
    start = datetime(2026, 1, 1, 0, 0)
    endpoint = start + timedelta(hours=30)
    windows = [
        {
            "niv_start": start.isoformat(),
            "niv_endpoint": endpoint.isoformat(),
            "window_end": (start + timedelta(hours=value)).isoformat(),
            "value": value,
        }
        for value in (1.6, 1.9, 5.0, 23.8, 24.2)
    ]
    initiation = select_landmark_window(
        windows,
        reference="initiation",
        landmark_hours=2,
    )
    endpoint_selected = select_landmark_window(
        windows,
        reference="endpoint",
        landmark_hours=6,
    )
    assert initiation is not None and initiation["value"] == 1.9
    assert endpoint_selected is not None and endpoint_selected["value"] == 24.2


def test_temporal_replacement_and_attribution_shape() -> None:
    fine_array = np.random.default_rng(1).normal(size=(2, 600, 8)).astype(np.float32)
    replaced = mean_replace(fine_array, 3)
    assert np.allclose(replaced[:, :, 3], fine_array[:, :, 3].mean(axis=1)[:, None])

    class SmallModel(torch.nn.Module):
        def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
            score = fine.mean(dim=(1, 2)) + coarse.mean(dim=1)
            return torch.stack([-score, score], dim=1)

    model = SmallModel().eval()
    fine = torch.from_numpy(fine_array[:1])
    coarse = torch.zeros(1, 20)
    labels = torch.ones(1, dtype=torch.long)
    scores = contribution_scores(model, fine, coarse, labels)
    assert scores.shape == (1, 40, 8)
    assert top_windows(scores, k=5).shape == (1, 5, 8)


def test_model_exposes_fine_and_coarse_representations() -> None:
    model = NIV_MIND().eval()
    with torch.inference_mode():
        fine = model.encode_fine(torch.zeros(1, 600, 8))
        coarse = model.encode_coarse(torch.zeros(1, 20))
    assert fine.shape == (1, 512)
    assert coarse.shape == (1, 128)
