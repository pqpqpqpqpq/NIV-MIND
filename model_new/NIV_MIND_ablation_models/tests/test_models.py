from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from model_new.NIV_MIND_ablation_models import EXPERIMENTS, build_model, get_spec
from model_new.NIV_MIND_ablation_models.registry import transform_fine


PACKAGE = Path(__file__).resolve().parents[1]
WEIGHTS = PACKAGE / "weights"


def test_registry_contains_all_groups_and_variants() -> None:
    assert set(EXPERIMENTS) == {
        "architecture",
        "sampling",
        "window",
        "compact_baseline",
    }
    assert sum(len(variants) for variants in EXPERIMENTS.values()) == 11


@pytest.mark.parametrize(
    ("group", "variant"),
    [
        ("architecture", "only_st"),
        ("sampling", "sample_120"),
        ("window", "window_150"),
        ("compact_baseline", "full"),
    ],
)
def test_representative_weights_load_strictly(group: str, variant: str) -> None:
    checkpoint = WEIGHTS / group / variant / "fold_0_best.pth"
    if not checkpoint.is_file():
        pytest.skip("fold checkpoint is not present in this checkout")
    model = build_model(group, variant)
    report = model.load_final_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(model.state_dict())
    assert report.loaded_keys == report.target_keys
    assert report.coverage == 1.0


def test_every_model_has_clean_keys() -> None:
    blocked = tuple(
        "".join(parts)
        for parts in (
            ("ori", "ginal"),
            ("keep", "_", "indices"),
            ("mod", "ule"),
            ("pre", "train"),
        )
    )
    for group, variants in EXPERIMENTS.items():
        for variant in variants:
            keys = build_model(group, variant).state_dict()
            assert not [
                key for key in keys
                if any(token in key.lower() for token in blocked)
            ]


def test_temporal_transformations() -> None:
    fine = np.arange(600 * 8, dtype=np.float32).reshape(1, 600, 8)
    sampled = transform_fine(fine, get_spec("sampling", "sample_120"))
    windowed = transform_fine(fine, get_spec("window", "window_150"))
    assert sampled.shape == (1, 120, 8)
    assert np.array_equal(sampled[:, 1], fine[:, 5])
    assert windowed.shape == (1, 150, 8)
    assert np.array_equal(windowed[:, 0], fine[:, 450])


def test_one_second_variant_requires_native_input() -> None:
    spec = get_spec("sampling", "sample_1200")
    with pytest.raises(ValueError, match="1200 one-second points"):
        transform_fine(np.zeros((1, 600, 8), dtype=np.float32), spec)
    native = np.zeros((1, 1200, 8), dtype=np.float32)
    assert transform_fine(native, spec).shape == (1, 1200, 8)


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "incomplete.pth"
    torch.save({"classifier.8.bias": torch.zeros(2)}, checkpoint)
    with pytest.raises(RuntimeError):
        build_model("architecture", "only_st").load_final_checkpoint(checkpoint)
