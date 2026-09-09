"""Five-fold training for registered ablation variants."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

try:
    from .compact import CompactAblationModel
    from .dataset import AblationDataset, load_labeled_samples
    from .registry import EXPERIMENTS, build_model, get_spec
    from model_new.NIV_MIND_model.analysis.splits import stratified_folds
except ImportError:
    workspace_root = Path(__file__).resolve().parents[2]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from model_new.NIV_MIND_ablation_models.compact import CompactAblationModel
    from model_new.NIV_MIND_ablation_models.dataset import (
        AblationDataset,
        load_labeled_samples,
    )
    from model_new.NIV_MIND_ablation_models.registry import (
        EXPERIMENTS,
        build_model,
        get_spec,
    )
    from model_new.NIV_MIND_model.analysis.splits import stratified_folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Both classes must occur in each training fold")
    values = labels.size / (2.0 * counts)
    return torch.tensor(values, dtype=torch.float32, device=device)


def metrics(
    labels: list[int],
    probabilities: list[float],
    loss: float,
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities_array >= 0.5).astype(np.int64)
    return {
        "loss": float(loss),
        "auc": float(roc_auc_score(labels_array, probabilities_array)),
        "auprc": float(
            average_precision_score(labels_array, probabilities_array)
        ),
        "accuracy": float(accuracy_score(labels_array, predicted)),
        "f1": float(f1_score(labels_array, predicted, zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    amp_enabled = training and scaler is not None and scaler.is_enabled()
    if training:
        optimizer.zero_grad(set_to_none=True)
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    total_loss = 0.0

    for step, (fine, coarse, labels) in enumerate(loader):
        fine = fine.to(device, non_blocking=True)
        coarse = coarse.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                if isinstance(model, CompactAblationModel) and training:
                    batch_loss, logits = model.training_loss(
                        fine,
                        coarse,
                        labels,
                        criterion,
                    )
                else:
                    logits = model(fine, coarse)
                    batch_loss = criterion(logits, labels)
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(batch_loss / accumulation_steps).backward()
                should_step = (
                    (step + 1) % accumulation_steps == 0
                    or step + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(batch_loss.detach()) * labels.size(0)
        probability = torch.softmax(logits.detach(), dim=1)[:, 1]
        labels_all.extend(labels.cpu().tolist())
        probabilities_all.extend(probability.cpu().tolist())
    if not labels_all:
        raise RuntimeError("DataLoader produced no batches")
    return metrics(labels_all, probabilities_all, total_loss / len(labels_all))


def loader(
    dataset: AblationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def train_fold(
    fold: int,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    samples: list[Any],
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed + fold)
    spec = get_spec(args.group, args.variant)
    train_dataset = AblationDataset(
        [samples[index] for index in train_indices],
        labels[train_indices],
        spec,
        training=True,
        fine_noise_std=args.fine_noise_std,
        coarse_feature_dropout=args.coarse_feature_dropout,
    )
    valid_dataset = AblationDataset(
        [samples[index] for index in valid_indices],
        labels[valid_indices],
        spec,
    )
    train_loader = loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(args.group, args.variant).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(labels[train_indices], device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and not args.no_amp,
    )
    output_dir = args.output_dir / args.group / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"fold_{fold}_best.pth"
    metrics_path = output_dir / f"fold_{fold}_metrics.json"
    history: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_epoch = -1
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        valid_result = run_epoch(model, valid_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train": train_result, "valid": valid_result}
        )
        print(
            f"{args.group}/{args.variant} fold {fold} epoch {epoch:03d}: "
            f"valid_auc={valid_result['auc']:.4f}"
        )
        if valid_result["auc"] > best_auc + args.min_delta:
            best_auc = valid_result["auc"]
            best_epoch = epoch
            stale = 0
            state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(state, checkpoint)
        else:
            stale += 1
        result = {
            "group": args.group,
            "variant": args.variant,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_valid_auc": best_auc,
            "checkpoint": str(checkpoint),
            "history": history,
        }
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if stale >= args.patience:
            break
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a registered NIV-MIND ablation variant."
    )
    parser.add_argument("--group", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("model_new")
            / "NIV_MIND_ablation_models"
            / "training_output"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--fine-noise-std", type=float, default=0.0)
    parser.add_argument("--coarse-feature-dropout", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0 if os.name == "nt" else 4,
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    get_spec(args.group, args.variant)
    if args.data_dir is None:
        args.data_dir = Path("data")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(
        samples,
        labels,
        n_splits=args.folds,
        seed=args.seed,
    )
    results = []
    for fold, (train_indices, valid_indices) in enumerate(folds):
        if args.fold is not None and fold != args.fold:
            continue
        results.append(
            train_fold(
                fold,
                train_indices,
                valid_indices,
                samples,
                labels,
                args,
                device,
            )
        )
    output = args.output_dir / args.group / args.variant
    summary = {
        "group": args.group,
        "variant": args.variant,
        "mean_valid_auc": float(
            np.mean([result["best_valid_auc"] for result in results])
        ),
        "results": results,
    }
    (output / "cv_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
