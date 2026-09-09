"""Replace one ventilator series by its sample-specific temporal mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_new.NIV_MIND_model import NIV_MIND

from .comparison import device_from_string, neural_oof
from .metrics import binary_metrics, paired_auc_difference_interval
from .splits import stratified_folds
from model_new.NIV_MIND_model.dataset import load_labeled_samples


CHANNELS = ("FiO2", "PEEPset", "PI", "I:E", "VT", "MV", "RR", "Ppeak")


def mean_replace(fine: np.ndarray, channel: int) -> np.ndarray:
    fine = np.asarray(fine, dtype=np.float32).copy()
    if fine.ndim != 3 or fine.shape[1:] != (600, 8):
        raise ValueError(f"fine must be [N,600,8], got {fine.shape}")
    if channel not in range(8):
        raise ValueError("channel must be in [0, 7]")
    fine[:, :, channel] = fine[:, :, channel].mean(axis=1, keepdims=True)
    return fine


def run(args: argparse.Namespace) -> dict:
    device = device_from_string(args.device)
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(samples, labels, n_splits=5, seed=args.seed)
    reference, _ = neural_oof(
        samples,
        labels,
        folds,
        NIV_MIND,
        args.weights,
        batch_size=args.batch_size,
        device=device,
    )
    output = {"full": binary_metrics(labels, reference), "channels": {}}
    for channel, name in enumerate(CHANNELS):
        modified_samples = []
        for sample in samples:
            copied = dict(sample)
            fine = np.asarray(sample["fine"])
            numeric = np.asarray(fine[:, :8], dtype=np.float32)[None]
            replaced = mean_replace(numeric, channel)[0]
            if fine.shape[1] == 9:
                copied["fine"] = np.column_stack([replaced, fine[:, 8]])
            else:
                copied["fine"] = replaced
            modified_samples.append(copied)
        scores, rows = neural_oof(
            modified_samples,
            labels,
            folds,
            NIV_MIND,
            args.weights,
            batch_size=args.batch_size,
            device=device,
        )
        output["channels"][name] = {
            "folds": rows,
            "pooled": binary_metrics(labels, scores),
            "delta_vs_full": paired_auc_difference_interval(
                labels,
                reference,
                scores,
                samples=args.bootstrap,
                seed=args.seed,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_new") / "NIV_MIND_model" / "weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_results") / "temporal_ablation.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["CHANNELS", "mean_replace"]
