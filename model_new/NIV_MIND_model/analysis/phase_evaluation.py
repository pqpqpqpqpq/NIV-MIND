"""Apply fold-specific endpoint models to another observation phase."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model_new.NIV_MIND_model import NIV_MIND

from .comparison import device_from_string
from .metrics import binary_metrics, bootstrap_auc_interval, decision_curve


def run(args: argparse.Namespace) -> dict:
    with np.load(args.input, allow_pickle=False) as archive:
        fine = np.asarray(archive["fine"], dtype=np.float32)
        coarse = np.asarray(archive["coarse"], dtype=np.float32)
        labels = np.asarray(archive["label"], dtype=np.int64)
        folds = np.asarray(archive["fold"], dtype=np.int64)
    if fine.ndim != 3 or fine.shape[1:] != (600, 8):
        raise ValueError(f"fine must be [N,600,8], got {fine.shape}")
    if coarse.shape != (fine.shape[0], 20):
        raise ValueError(f"coarse must be [N,20], got {coarse.shape}")
    if labels.shape != (fine.shape[0],) or folds.shape != labels.shape:
        raise ValueError("label and fold must be [N]")
    if not np.all(np.isin(folds, range(5))):
        raise ValueError("fold values must be in [0, 4]")

    device = device_from_string(args.device)
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    fold_metrics = []
    for fold in range(5):
        indices = np.flatnonzero(folds == fold)
        if indices.size == 0:
            continue
        model, _ = NIV_MIND.from_checkpoint(
            args.weights / f"fold_{fold}_best.pth",
            device=device,
        )
        parts = []
        with torch.inference_mode():
            for start in range(0, indices.size, args.batch_size):
                selected = indices[start : start + args.batch_size]
                probabilities = model.predict_proba(
                    torch.from_numpy(fine[selected]).to(device),
                    torch.from_numpy(coarse[selected]).to(device),
                )[:, 1]
                parts.append(probabilities.cpu().numpy())
        fold_scores = np.concatenate(parts)
        scores[indices] = fold_scores
        if np.unique(labels[indices]).size == 2:
            row = binary_metrics(labels[indices], fold_scores)
            row["fold"] = fold
            fold_metrics.append(row)
    if not np.isfinite(scores).all():
        raise RuntimeError("phase prediction vector is incomplete")

    output = {
        "folds": fold_metrics,
        "pooled": binary_metrics(labels, scores),
        "auc_interval": bootstrap_auc_interval(
            labels,
            scores,
            samples=args.bootstrap,
            seed=args.seed,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "fold", "label", "score"))
        writer.writerows(
            (index, int(folds[index]), int(labels[index]), float(scores[index]))
            for index in range(labels.size)
        )
    with (args.output_dir / "decision_curve.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = decision_curve(labels, scores)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_new") / "NIV_MIND_model" / "weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_results") / "phase_evaluation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
