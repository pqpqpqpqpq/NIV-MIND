"""Five-fold training for NIV-MIND."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

try:
    from .analysis.splits import stratified_folds
    from .dataset import NIVMINDDataset, load_labeled_samples
    from .model import NIV_MIND
except ImportError:
    from analysis.splits import stratified_folds
    from dataset import NIVMINDDataset, load_labeled_samples
    from model import NIV_MIND


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
        raise RuntimeError("CUDA was requested but is not available")
    return device


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Each training fold must contain both classes")
    values = labels.size / (2.0 * counts)
    return torch.tensor(values, dtype=torch.float32, device=device)


def metric_values(
    labels: list[int],
    probabilities: list[float],
    mean_loss: float,
) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probability_array >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        label_array,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "loss": float(mean_loss),
        "auc": float(roc_auc_score(label_array, probability_array)),
        "auprc": float(
            average_precision_score(label_array, probability_array)
        ),
        "accuracy": float(accuracy_score(label_array, predictions)),
        "precision": float(
            precision_score(label_array, predictions, zero_division=0)
        ),
        "sensitivity": float(
            recall_score(label_array, predictions, zero_division=0)
        ),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
    }


def run_epoch(
    model: NIV_MIND,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    amp_enabled = training and scaler is not None and scaler.is_enabled()
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    labels_all: list[int] = []
    probabilities_all: list[float] = []

    for step, (fine, coarse, labels) in enumerate(loader):
        fine = fine.to(device, non_blocking=True)
        coarse = coarse.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(fine, coarse)
                batch_loss = criterion(logits, labels)

            if training:
                assert optimizer is not None
                assert scaler is not None
                scaler.scale(batch_loss / accumulation_steps).backward()
                should_step = (
                    (step + 1) % accumulation_steps == 0
                    or step + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(batch_loss.detach()) * labels.size(0)
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        labels_all.extend(labels.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())

    if not labels_all:
        raise RuntimeError("DataLoader produced no batches")
    return metric_values(
        labels_all,
        probabilities_all,
        total_loss / len(labels_all),
    )


def cpu_state_dict(model: NIV_MIND) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def make_loader(
    dataset: NIVMINDDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
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
    fold_seed = args.seed + fold
    set_seed(fold_seed)

    train_samples = [samples[index] for index in train_indices]
    valid_samples = [samples[index] for index in valid_indices]
    train_labels = labels[train_indices]
    valid_labels = labels[valid_indices]

    train_dataset = NIVMINDDataset(
        train_samples,
        train_labels,
        training=True,
        fine_noise_std=args.fine_noise_std,
        coarse_feature_dropout=args.coarse_feature_dropout,
    )
    valid_dataset = NIVMINDDataset(
        valid_samples,
        valid_labels,
        training=False,
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = NIV_MIND().to(device)
    weights = class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_path = args.output_dir / f"fold_{fold}_best.pth"
    metrics_path = args.output_dir / f"fold_{fold}_metrics.json"
    history: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
        )
        scheduler.step()
        epoch_result = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(epoch_result)
        print(
            f"Fold {fold} epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_auc={valid_metrics['auc']:.4f}"
        )

        if valid_metrics["auc"] > best_auc + args.min_delta:
            best_auc = valid_metrics["auc"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(cpu_state_dict(model), checkpoint_path)
        else:
            stale_epochs += 1

        fold_result = {
            "fold": fold,
            "seed": fold_seed,
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
            "class_weights": weights.detach().cpu().tolist(),
            "best_epoch": best_epoch,
            "best_valid_auc": best_auc,
            "checkpoint": str(checkpoint_path),
            "history": history,
        }
        metrics_path.write_text(
            json.dumps(fold_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if stale_epochs >= args.patience:
            print(f"Fold {fold}: early stopping at epoch {epoch}")
            break

    return fold_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NIV-MIND with stratified cross-validation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_new") / "NIV_MIND_model" / "training_output",
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
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    for name in ("epochs", "patience", "batch_size", "accumulation_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed)
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(
        samples,
        labels,
        n_splits=args.folds,
        seed=args.seed,
    )

    print(
        f"Loaded {len(samples)} samples: "
        f"success={int((labels == 0).sum())}, "
        f"failure={int((labels == 1).sum())}; device={device}"
    )
    results: list[dict[str, Any]] = []
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

    summary = {
        "model": "NIV-MIND",
        "device": str(device),
        "folds_completed": len(results),
        "mean_valid_auc": float(
            np.mean([result["best_valid_auc"] for result in results])
        ),
        "settings": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "results": results,
    }
    summary_path = args.output_dir / "cv_metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
