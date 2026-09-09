"""Endpoint model comparison, OOF metrics, ROC, PR, and decision curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_new.NIV_MIND_model import NIV_MIND
from model_new.NIV_MIND_model.dataset import (
    NIVMINDDataset,
    load_labeled_samples,
)

from .metrics import (
    binary_metrics,
    bootstrap_auc_interval,
    decision_curve,
    paired_auc_difference_interval,
)
from .comparators import NIVMINDC, NIVMINDF
from .splits import stratified_folds


FINE_COLUMNS = ("FiO2", "PEEPset", "PI", "I:E", "VT", "MV", "RR", "Ppeak")
COARSE_COLUMNS = (
    "pao2",
    "paCO2",
    "pH",
    "HCO3",
    "PF",
    "temp",
    "lactic_acid",
    "sodium",
    "chloride",
    "potassium",
    "calcium",
    "methemoglobin",
    "base_excess",
    "white_blood_cell_count",
    "red_blood_cell_count",
    "hemoglobin",
    "platelet_count",
    "heart_rate",
    "spo2",
    "age",
)
FINE_INDEX = {name: index for index, name in enumerate(FINE_COLUMNS)}
COARSE_INDEX = {name: index for index, name in enumerate(COARSE_COLUMNS)}


def device_from_string(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _fine_array(sample: dict) -> np.ndarray:
    fine = np.asarray(sample["fine"])
    if fine.ndim != 2 or fine.shape[0] != 600 or fine.shape[1] not in (8, 9):
        raise ValueError(f"fine must be [600,8] or [600,9], got {fine.shape}")
    fine = np.asarray(fine[:, :8], dtype=np.float32)
    if not np.isfinite(fine).all():
        raise ValueError("fine contains NaN or infinity")
    return fine


def _coarse_array(sample: dict) -> np.ndarray:
    coarse = np.asarray(sample["coarse"], dtype=np.float32).squeeze()
    if coarse.shape != (20,) or not np.isfinite(coarse).all():
        raise ValueError("coarse must contain 20 finite values")
    return coarse


def _predict_neural_fold(
    model: torch.nn.Module,
    samples: list[dict],
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = NIVMINDDataset(
        [samples[index] for index in indices],
        labels[indices],
        training=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for fine, coarse, _ in loader:
            logits = model(fine.to(device), coarse.to(device))
            values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def neural_oof(
    samples: list[dict],
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], torch.nn.Module],
    weight_dir: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, float | int]] = []
    for fold, (_, valid_indices) in enumerate(folds):
        checkpoint = weight_dir / f"fold_{fold}_best.pth"
        model = model_factory()
        loader = getattr(model, "load_final_checkpoint", None)
        if loader is None:
            raise TypeError("model has no checkpoint loader")
        loader(checkpoint)
        fold_scores = _predict_neural_fold(
            model,
            samples,
            labels,
            valid_indices,
            batch_size=batch_size,
            device=device,
        )
        scores[valid_indices] = fold_scores
        row = binary_metrics(labels[valid_indices], fold_scores)
        row["fold"] = fold
        fold_metrics.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(scores).all():
        raise RuntimeError("OOF prediction vector is incomplete")
    return scores, fold_metrics


def _score_hr(value: float) -> int:
    return 0 if value <= 120 else 1 if value <= 140 else 2


def _score_ph(value: float) -> int:
    if value >= 7.35:
        return 0
    if value >= 7.30:
        return 2
    if value >= 7.25:
        return 3
    return 4


def _score_pf(value: float) -> int:
    if value >= 201:
        return 0
    if value >= 176:
        return 2
    if value >= 151:
        return 3
    if value >= 126:
        return 4
    if value >= 101:
        return 5
    return 6


def _score_rr(value: float) -> int:
    if value <= 30:
        return 0
    if value <= 35:
        return 1
    if value <= 40:
        return 2
    if value <= 45:
        return 3
    return 4


def clinical_arrays(
    samples: list[dict],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fine_mean = np.load(data_dir / "data_fine_mean.npy")
    fine_std = np.load(data_dir / "data_fine_std.npy")
    coarse_mean = np.load(data_dir / "data_coarse_mean.npy")
    coarse_std = np.load(data_dir / "data_coarse_std.npy")
    coarse_features = []
    hacor_values = []
    rox_values = []
    for sample in samples:
        fine = _fine_array(sample)
        coarse = _coarse_array(sample)
        raw_fine = fine * fine_std + fine_mean
        raw_coarse = coarse * coarse_std + coarse_mean
        rr = float(raw_fine[:, FINE_INDEX["RR"]].mean())
        fio2 = float(raw_fine[:, FINE_INDEX["FiO2"]].mean())
        heart_rate = float(raw_coarse[COARSE_INDEX["heart_rate"]])
        spo2 = float(raw_coarse[COARSE_INDEX["spo2"]])
        ph = float(raw_coarse[COARSE_INDEX["pH"]])
        pf = float(raw_coarse[COARSE_INDEX["PF"]])
        hacor_values.append(
            _score_hr(heart_rate) + _score_ph(ph) + _score_pf(pf) + _score_rr(rr)
        )
        rox_values.append((spo2 / max(fio2, 1e-8)) / max(rr, 1e-8))
        coarse_features.append(coarse)
    return (
        np.asarray(coarse_features, dtype=np.float64),
        np.asarray(hacor_values, dtype=np.float64),
        np.asarray(rox_values, dtype=np.float64),
    )


def _calibrated_score_oof(
    values: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for fold, (train_indices, valid_indices) in enumerate(folds):
        median = float(np.nanmedian(values[train_indices]))
        train_values = np.nan_to_num(values[train_indices], nan=median)
        valid_values = np.nan_to_num(values[valid_indices], nan=median)
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(train_values[:, None], labels[train_indices])
        fold_scores = calibrator.predict_proba(valid_values[:, None])[:, 1]
        scores[valid_indices] = fold_scores
        row = binary_metrics(labels[valid_indices], fold_scores)
        row["fold"] = fold
        rows.append(row)
    return scores, rows


def comparator_oof(
    samples: list[dict],
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    data_dir: Path,
    *,
    seed: int,
) -> dict[str, tuple[np.ndarray, list[dict[str, float | int]]]]:
    coarse, hacor_values, rox_values = clinical_arrays(samples, data_dir)
    lgbm_scores = np.full(labels.size, np.nan, dtype=np.float64)
    lgbm_rows: list[dict[str, float | int]] = []
    for fold, (train_indices, valid_indices) in enumerate(folds):
        classifier = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.10,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=seed,
            verbosity=-1,
        )
        classifier.fit(coarse[train_indices], labels[train_indices])
        fold_scores = classifier.predict_proba(coarse[valid_indices])[:, 1]
        lgbm_scores[valid_indices] = fold_scores
        row = binary_metrics(labels[valid_indices], fold_scores)
        row["fold"] = fold
        lgbm_rows.append(row)
    hacor_scores, hacor_rows = _calibrated_score_oof(
        hacor_values,
        labels,
        folds,
    )
    rox_scores, rox_rows = _calibrated_score_oof(
        -rox_values,
        labels,
        folds,
    )
    return {
        "LightGBM": (lgbm_scores, lgbm_rows),
        "HACOR": (hacor_scores, hacor_rows),
        "ROX": (rox_scores, rox_rows),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_prediction_table(
    path: Path,
    sample_count: int,
) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("prediction table is empty")
    columns = set(rows[0])
    result: dict[str, np.ndarray] = {}
    if {"global_index", "model", "score"} <= columns:
        for row in rows:
            model = row["model"]
            result.setdefault(
                model,
                np.full(sample_count, np.nan, dtype=np.float64),
            )[int(row["global_index"])] = float(row["score"])
    elif ({"record_index", "label"} <= columns) or ({"index", "label"} <= columns):
        index_column = "record_index" if "record_index" in columns else "index"
        excluded = {index_column, "fold", "label"}
        for model in columns - excluded:
            values = np.full(sample_count, np.nan, dtype=np.float64)
            for row in rows:
                values[int(row[index_column])] = float(row[model])
            result[model] = values
    else:
        raise ValueError("prediction table must use long or wide OOF columns")
    incomplete = [name for name, values in result.items() if not np.isfinite(values).all()]
    if incomplete:
        raise ValueError(f"incomplete prediction columns: {incomplete}")
    return result


def _plot_curves(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    colors = {
        "NIV-MIND": "#C44E52",
        "NIV-MIND-C": "#8172B3",
        "NIV-MIND-F": "#CCB974",
        "LightGBM": "#55A868",
        "HACOR": "#DD8452",
        "ROX": "#4C72B0",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, scores in predictions.items():
        color = colors.get(name)
        false_positive, true_positive, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(
            false_positive,
            true_positive,
            label=f"{name} ({roc_auc_score(labels, scores):.3f})",
            color=color,
        )
        axes[1].plot(
            recall,
            precision,
            label=f"{name} ({average_precision_score(labels, scores):.3f})",
            color=color,
        )
        curve = decision_curve(labels, scores)
        axes[2].plot(
            [row["threshold"] for row in curve],
            [row["model"] for row in curve],
            label=name,
            color=color,
        )
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--")
    axes[1].axhline(labels.mean(), color="0.6", linestyle="--")
    reference_curve = decision_curve(labels, next(iter(predictions.values())))
    axes[2].plot(
        [row["threshold"] for row in reference_curve],
        [row["treat_all"] for row in reference_curve],
        color="0.5",
        linestyle="--",
        label="Treat all",
    )
    axes[2].axhline(0.0, color="0.2", linestyle=":", label="Treat none")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    axes[2].set(xlabel="Threshold probability", ylabel="Net benefit")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_comparison.png", dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    device = device_from_string(args.device)
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(
        samples,
        labels,
        n_splits=5,
        seed=args.seed,
    )
    predictions: dict[str, np.ndarray] = {}
    fold_results: dict[str, list[dict[str, float | int]]] = {}

    model_specs = {
        "NIV-MIND": (
            NIV_MIND,
            args.main_weights,
        ),
        "NIV-MIND-C": (
            NIVMINDC,
            args.comparator_weights / "niv_mind_c",
        ),
        "NIV-MIND-F": (
            NIVMINDF,
            args.comparator_weights / "niv_mind_f",
        ),
    }
    for name, (factory, weight_dir) in model_specs.items():
        scores, rows = neural_oof(
            samples,
            labels,
            folds,
            factory,
            weight_dir,
            batch_size=args.batch_size,
            device=device,
        )
        predictions[name] = scores
        fold_results[name] = rows

    for name, (scores, rows) in comparator_oof(
        samples,
        labels,
        folds,
        args.data_dir,
        seed=args.seed,
    ).items():
        predictions[name] = scores
        fold_results[name] = rows

    if args.prediction_table is not None:
        overrides = load_prediction_table(args.prediction_table, labels.size)
        for name, scores in overrides.items():
            if name not in predictions:
                continue
            predictions[name] = scores
            rows = []
            for fold, (_, valid_indices) in enumerate(folds):
                row = binary_metrics(labels[valid_indices], scores[valid_indices])
                row["fold"] = fold
                rows.append(row)
            fold_results[name] = rows

    summary = {}
    reference = predictions["NIV-MIND"]
    for name, scores in predictions.items():
        fold_auc = [float(row["auc"]) for row in fold_results[name]]
        fold_auprc = [float(row["auprc"]) for row in fold_results[name]]
        summary[name] = {
            "folds": fold_results[name],
            "fold_auc_mean": float(np.mean(fold_auc)),
            "fold_auc_sd": float(np.std(fold_auc)),
            "fold_auprc_mean": float(np.mean(fold_auprc)),
            "fold_auprc_sd": float(np.std(fold_auprc)),
            "pooled": binary_metrics(labels, scores),
            "auc_interval": bootstrap_auc_interval(
                labels,
                scores,
                samples=args.bootstrap,
                seed=args.seed,
            ),
        }
        if name != "NIV-MIND":
            summary[name]["delta_vs_niv_mind"] = paired_auc_difference_interval(
                labels,
                reference,
                scores,
                samples=args.bootstrap,
                seed=args.seed,
            )

    fold_for_index = np.empty(labels.size, dtype=np.int64)
    for fold, (_, valid_indices) in enumerate(folds):
        fold_for_index[valid_indices] = fold
    prediction_rows = []
    for index in range(labels.size):
        row = {
            "record_index": index,
            "fold": int(fold_for_index[index]),
            "label": int(labels[index]),
        }
        row.update({name: float(scores[index]) for name, scores in predictions.items()})
        prediction_rows.append(row)

    decision_rows = []
    for name, scores in predictions.items():
        for row in decision_curve(labels, scores):
            decision_rows.append({"model": name, **row})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "oof_predictions.csv", prediction_rows)
    _write_csv(args.output_dir / "decision_curve.csv", decision_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_curves(labels, predictions, args.output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--main-weights",
        type=Path,
        default=Path("model_new") / "NIV_MIND_model" / "weights",
    )
    parser.add_argument(
        "--comparator-weights",
        type=Path,
        default=Path("weights") / "comparators",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_results") / "endpoint_comparison",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prediction-table",
        type=Path,
        help="Optional long or wide OOF table used to reproduce archived curves.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    for name, values in summary.items():
        pooled = values["pooled"]
        print(f"{name}: AUC={pooled['auc']:.4f}, AUPRC={pooled['auprc']:.4f}")


if __name__ == "__main__":
    main()
