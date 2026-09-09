"""Engineered ventilator features and greedy forward selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

from model_new.NIV_MIND_model.dataset import load_labeled_samples

from .splits import stratified_folds


CHANNELS = ("FiO2", "PEEPset", "PI", "I:E", "VT", "MV", "RR", "Ppeak")
FEATURE_NAMES = (
    "std",
    "cv",
    "minimum",
    "maximum",
    "range",
    "median",
    "iqr",
    "skewness",
    "kurtosis",
    "p10",
    "p25",
    "p75",
    "p90",
    "trend_slope",
    "trend_r2",
    "mean_absolute_difference",
    "maximum_absolute_difference",
    "number_of_changes",
    "lag1_autocorrelation",
    "lag5_autocorrelation",
    "window_mean_std",
    "window_std_mean",
    "early_mean",
    "late_mean",
    "early_late_difference",
    "shannon_entropy",
    "permutation_entropy",
    "dominant_frequency",
    "spectral_entropy",
    "low_band_energy_ratio",
    "high_band_energy_ratio",
    "total_power",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "rms",
    "crest_factor",
    "zero_crossing_rate",
    "number_of_peaks",
    "mean_peak_interval",
    "std_peak_interval",
)


def _safe_correlation(x: np.ndarray, lag: int) -> float:
    if x.size <= lag or np.std(x[:-lag]) < 1e-12 or np.std(x[lag:]) < 1e-12:
        return 0.0
    value = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _shannon_entropy(x: np.ndarray, bins: int = 20) -> float:
    counts = np.histogram(x, bins=bins)[0].astype(np.float64)
    probabilities = counts[counts > 0] / max(counts.sum(), 1.0)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    if x.size < order:
        return 0.0
    patterns: dict[tuple[int, ...], int] = {}
    for index in range(x.size - order + 1):
        pattern = tuple(np.argsort(x[index : index + order], kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    probabilities = np.asarray(list(patterns.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    value = -np.sum(probabilities * np.log2(probabilities))
    return float(value / math.log2(math.factorial(order)))


def extract_signal_features(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 3 or not np.isfinite(x).all():
        raise ValueError("signal must contain at least three finite values")
    epsilon = 1e-8
    mean = float(np.mean(x))
    standard_deviation = float(np.std(x))
    minimum = float(np.min(x))
    maximum = float(np.max(x))
    signal_range = maximum - minimum
    p10, p25, median, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])

    time = np.arange(x.size, dtype=np.float64)
    slope, intercept = np.polyfit(time, x, 1)
    fitted = slope * time + intercept
    residual_sum = float(np.sum((x - fitted) ** 2))
    total_sum = float(np.sum((x - mean) ** 2))
    trend_r2 = 1.0 - residual_sum / total_sum if total_sum > epsilon else 0.0

    differences = np.diff(x)
    window_count = min(10, x.size)
    windows = [window for window in np.array_split(x, window_count) if window.size]
    third = max(x.size // 3, 1)
    early_mean = float(np.mean(x[:third]))
    late_mean = float(np.mean(x[-third:]))

    centered = x - mean
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(x.size)
    if power.size > 1:
        candidate_power = power[1:]
        candidate_frequencies = frequencies[1:]
        dominant_frequency = float(
            candidate_frequencies[int(np.argmax(candidate_power))]
        )
    else:
        dominant_frequency = 0.0
    total_power = float(np.sum(power))
    normalized_power = power / max(total_power, epsilon)
    nonzero_power = normalized_power[normalized_power > 0]
    spectral_entropy = float(
        -np.sum(nonzero_power * np.log2(nonzero_power))
        / max(math.log2(max(normalized_power.size, 2)), epsilon)
    )
    split = max(power.size // 4, 1)
    low_ratio = float(np.sum(power[:split]) / max(total_power, epsilon))
    high_ratio = float(np.sum(power[-split:]) / max(total_power, epsilon))

    derivative = np.diff(x)
    derivative_two = np.diff(derivative)
    activity = float(np.var(x))
    mobility = math.sqrt(float(np.var(derivative)) / max(activity, epsilon))
    derivative_mobility = math.sqrt(
        float(np.var(derivative_two)) / max(float(np.var(derivative)), epsilon)
    )
    complexity = derivative_mobility / max(mobility, epsilon)
    rms = float(np.sqrt(np.mean(x**2)))
    crest_factor = float(np.max(np.abs(x)) / max(rms, epsilon))
    zero_crossing_rate = float(np.mean(centered[:-1] * centered[1:] < 0))

    peaks, _ = find_peaks(x, height=mean + 0.5 * standard_deviation)
    intervals = np.diff(peaks).astype(np.float64)
    change_threshold = 0.01 * signal_range
    values = (
        standard_deviation,
        standard_deviation / (abs(mean) + epsilon),
        minimum,
        maximum,
        signal_range,
        float(median),
        float(p75 - p25),
        float(np.nan_to_num(skew(x, bias=False), nan=0.0)),
        float(np.nan_to_num(kurtosis(x, fisher=True, bias=False), nan=0.0)),
        float(p10),
        float(p25),
        float(p75),
        float(p90),
        float(slope),
        float(trend_r2),
        float(np.mean(np.abs(differences))),
        float(np.max(np.abs(differences))),
        float(np.sum(np.abs(differences) > change_threshold)),
        _safe_correlation(x, 1),
        _safe_correlation(x, 5),
        float(np.std([np.mean(window) for window in windows])),
        float(np.mean([np.std(window) for window in windows])),
        early_mean,
        late_mean,
        late_mean - early_mean,
        _shannon_entropy(x),
        _permutation_entropy(x),
        dominant_frequency,
        spectral_entropy,
        low_ratio,
        high_ratio,
        total_power,
        activity,
        mobility,
        complexity,
        rms,
        crest_factor,
        zero_crossing_rate,
        float(peaks.size),
        float(np.mean(intervals)) if intervals.size else 0.0,
        float(np.std(intervals)) if intervals.size else 0.0,
    )
    return dict(zip(FEATURE_NAMES, values, strict=True))


def extract_feature_matrix(fine: np.ndarray) -> tuple[np.ndarray, list[str]]:
    fine = np.asarray(fine, dtype=np.float64)
    if fine.ndim != 3 or fine.shape[2] != 8:
        raise ValueError(f"fine must be [N,T,8], got {fine.shape}")
    names = [
        f"{channel}__{feature}"
        for channel in CHANNELS
        for feature in FEATURE_NAMES
    ]
    matrix = np.empty((fine.shape[0], len(names)), dtype=np.float64)
    for sample_index in range(fine.shape[0]):
        offset = 0
        for channel_index in range(8):
            values = extract_signal_features(fine[sample_index, :, channel_index])
            for feature in FEATURE_NAMES:
                matrix[sample_index, offset] = values[feature]
                offset += 1
    return matrix, names


def extract_top_window_feature_matrix(
    fine: np.ndarray,
    contribution: np.ndarray,
    *,
    window_size: int = 15,
    top_k: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Build the 2,048 window-level candidates used in the staged analysis."""
    fine = np.asarray(fine, dtype=np.float64)
    contribution = np.asarray(contribution, dtype=np.float64)
    if fine.ndim != 3 or fine.shape[2] != 8:
        raise ValueError(f"fine must be [N,T,8], got {fine.shape}")
    if fine.shape[1] % window_size:
        raise ValueError("sequence length must be divisible by window_size")
    windows = fine.shape[1] // window_size
    if contribution.shape != (fine.shape[0], windows, 8):
        raise ValueError(
            f"contribution must be [N,{windows},8], got {contribution.shape}"
        )
    if top_k != 5:
        raise ValueError("the manuscript candidate layout uses top_k=5")

    names: list[str] = []
    for channel in CHANNELS:
        for rank in range(1, top_k + 1):
            names.extend(
                f"{channel}__top{rank}__{feature}" for feature in FEATURE_NAMES
            )
            names.append(f"{channel}__top{rank}__normalized_position")
            names.append(f"{channel}__top{rank}__contribution")
        names.extend(f"{channel}__top5_mean__{feature}" for feature in FEATURE_NAMES)
    matrix = np.empty((fine.shape[0], len(names)), dtype=np.float64)

    for sample_index in range(fine.shape[0]):
        offset = 0
        for channel_index in range(8):
            top_indices = np.argsort(
                contribution[sample_index, :, channel_index]
            )[-top_k:][::-1]
            feature_rows = []
            for window_index in top_indices:
                start = window_index * window_size
                stop = start + window_size
                features = extract_signal_features(
                    fine[sample_index, start:stop, channel_index]
                )
                feature_rows.append([features[name] for name in FEATURE_NAMES])
                for name in FEATURE_NAMES:
                    matrix[sample_index, offset] = features[name]
                    offset += 1
                matrix[sample_index, offset] = window_index / max(windows - 1, 1)
                offset += 1
                matrix[sample_index, offset] = contribution[
                    sample_index,
                    window_index,
                    channel_index,
                ]
                offset += 1
            means = np.mean(feature_rows, axis=0)
            matrix[sample_index, offset : offset + len(FEATURE_NAMES)] = means
            offset += len(FEATURE_NAMES)
    if matrix.shape[1] != 2048:
        raise RuntimeError(f"expected 2048 window candidates, got {matrix.shape[1]}")
    return matrix, names


def greedy_forward_selection(
    baseline_train: np.ndarray,
    baseline_valid: np.ndarray,
    candidates_train: np.ndarray,
    candidates_valid: np.ndarray,
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
    candidate_names: list[str],
    *,
    seed: int = 42,
    min_gain: float = 0.0,
    max_features: int | None = None,
) -> list[dict[str, float | int | str]]:
    selected: list[int] = []
    remaining = list(range(candidates_train.shape[1]))
    result: list[dict[str, float | int | str]] = []
    def evaluate(train: np.ndarray, valid: np.ndarray) -> float:
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
        classifier.fit(train, train_labels)
        return float(
            roc_auc_score(valid_labels, classifier.predict_proba(valid)[:, 1])
        )

    current_auc = evaluate(baseline_train, baseline_valid)
    while remaining and (max_features is None or len(selected) < max_features):
        best_index = -1
        best_auc = current_auc
        for candidate in remaining:
            columns = selected + [candidate]
            train = np.column_stack(
                [baseline_train, candidates_train[:, columns]]
            )
            valid = np.column_stack(
                [baseline_valid, candidates_valid[:, columns]]
            )
            auc = evaluate(train, valid)
            if auc > best_auc:
                best_auc = float(auc)
                best_index = candidate
        if best_index < 0 or best_auc - current_auc <= min_gain:
            break
        gain = best_auc - current_auc
        selected.append(best_index)
        remaining.remove(best_index)
        current_auc = best_auc
        result.append(
            {
                "rank": len(selected),
                "feature": candidate_names[best_index],
                "auc": current_auc,
                "delta_auc": gain,
            }
        )
    return result


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    samples, labels = load_labeled_samples(args.data_dir)
    fine = np.stack(
        [np.asarray(sample["fine"])[:, :8].astype(np.float32) for sample in samples]
    )
    coarse = np.stack(
        [np.asarray(sample["coarse"], dtype=np.float32) for sample in samples]
    )
    engineered, names = extract_feature_matrix(fine)
    means = fine.mean(axis=1)
    baseline = np.column_stack([means, coarse])
    folds = stratified_folds(samples, labels, n_splits=5, seed=args.seed)
    train_indices, valid_indices = folds[args.fold]
    selected = greedy_forward_selection(
        baseline[train_indices],
        baseline[valid_indices],
        engineered[train_indices],
        engineered[valid_indices],
        labels[train_indices],
        labels[valid_indices],
        names,
        seed=args.seed,
        min_gain=args.min_gain,
        max_features=args.max_features,
    )
    phase_two = []
    if args.contribution is not None:
        with np.load(args.contribution, allow_pickle=False) as archive:
            contribution = np.asarray(archive["contribution"], dtype=np.float64)
        window_features, window_names = extract_top_window_feature_matrix(
            fine,
            contribution,
        )
        selected_indices = [names.index(str(row["feature"])) for row in selected]
        phase_one_baseline = np.column_stack(
            [baseline, engineered[:, selected_indices]]
        )
        phase_two = greedy_forward_selection(
            phase_one_baseline[train_indices],
            phase_one_baseline[valid_indices],
            window_features[train_indices],
            window_features[valid_indices],
            labels[train_indices],
            labels[valid_indices],
            window_names,
            seed=args.seed,
            min_gain=args.min_gain,
            max_features=args.max_window_features,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "selected_features.csv", selected)
    payload = {
        "fold": args.fold,
        "candidate_features": len(names),
        "selected": selected,
        "window_candidate_features": 2048 if args.contribution is not None else 0,
        "window_selected": phase_two,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_results") / "engineered_features",
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--contribution", type=Path)
    parser.add_argument("--max-window-features", type=int)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CHANNELS",
    "FEATURE_NAMES",
    "extract_feature_matrix",
    "extract_signal_features",
    "extract_top_window_feature_matrix",
    "greedy_forward_selection",
]
