from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from model_new.NIV_MIND_model import NIV_MIND
from model_new.NIV_MIND_model.dataset import NIVMINDDataset
from model_new.NIV_MIND_model.predict import NIV_MIND as PredictionModel
from model_new.NIV_MIND_model.train import NIV_MIND as TrainingModel


PACKAGE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PACKAGE_DIR / "weights"


@pytest.mark.parametrize("fold", range(5))
def test_all_fold_checkpoints_load_strictly(fold: int) -> None:
    checkpoint = WEIGHTS_DIR / f"fold_{fold}_best.pth"
    if not checkpoint.is_file():
        pytest.skip("fold checkpoint is not present in this checkout")

    model = NIV_MIND()
    report = model.load_final_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert len(saved) == 353
    assert set(saved) == set(model.state_dict())
    assert all(isinstance(value, torch.Tensor) for value in saved.values())
    assert report.loaded_keys == 353
    assert report.target_keys == 353
    assert report.coverage == 1.0
    assert report.ready_for_evaluation


def test_training_and_prediction_share_the_model_class() -> None:
    assert TrainingModel is NIV_MIND
    assert PredictionModel is NIV_MIND


def test_reloaded_model_is_numerically_identical(tmp_path: Path) -> None:
    source = NIV_MIND().eval()
    checkpoint = tmp_path / "final.pth"
    torch.save(source.state_dict(), checkpoint)
    reloaded, _ = NIV_MIND.from_checkpoint(checkpoint)
    fine = torch.randn(2, 600, 8)
    coarse = torch.randn(2, 20)

    with torch.inference_mode():
        expected = source(fine, coarse)
        actual = reloaded(fine, coarse)

    assert torch.equal(actual, expected)


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "incomplete.pth"
    torch.save({"classifier.8.bias": torch.zeros(2)}, checkpoint)
    with pytest.raises(RuntimeError):
        NIV_MIND().load_final_checkpoint(checkpoint)


def test_dataset_removes_timestamp_column() -> None:
    numeric = np.zeros((600, 8), dtype=np.float32)
    timestamps = np.full((600, 1), "2026-01-01 00:00:00", dtype=object)
    fine = np.concatenate([numeric.astype(object), timestamps], axis=1)
    samples = [{"fine": fine, "coarse": np.zeros(20, dtype=np.float32)}]
    dataset = NIVMINDDataset(samples, [1], training=False)

    fine_tensor, coarse_tensor, label = dataset[0]
    assert fine_tensor.shape == (600, 8)
    assert coarse_tensor.shape == (20,)
    assert label.item() == 1


def test_forward_shape_validation() -> None:
    model = NIV_MIND()
    with pytest.raises(ValueError, match="fine must have shape"):
        model(torch.zeros(1, 599, 8), torch.zeros(1, 20))
