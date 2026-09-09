"""Five-fold training for NIV-MIND-C and NIV-MIND-F."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_new.NIV_MIND_model.dataset import NIVMINDDataset, load_labeled_samples
from model_new.NIV_MIND_model.train import (
    class_weights,
    make_loader,
    resolve_device,
    run_epoch,
    set_seed,
)

from .comparators import NIVMINDC, NIVMINDF
from .splits import stratified_folds


def build(name: str) -> nn.Module:
    if name == "niv_mind_c":
        return NIVMINDC()
    if name == "niv_mind_f":
        return NIVMINDF()
    raise ValueError(name)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(samples, labels, n_splits=5, seed=args.seed)
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for fold, (train_indices, valid_indices) in enumerate(folds):
        if args.fold is not None and fold != args.fold:
            continue
        set_seed(args.seed + fold)
        train_dataset = NIVMINDDataset(
            [samples[index] for index in train_indices],
            labels[train_indices],
            training=False,
        )
        valid_dataset = NIVMINDDataset(
            [samples[index] for index in valid_indices],
            labels[valid_indices],
            training=False,
        )
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        valid_loader = make_loader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = build(args.model).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights(labels[train_indices], device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=device.type == "cuda" and not args.no_amp,
        )
        best_auc = float("-inf")
        best_epoch = -1
        stale = 0
        history = []
        checkpoint = output_dir / f"fold_{fold}_best.pth"
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                accumulation_steps=args.accumulation_steps,
            )
            valid_metrics = run_epoch(model, valid_loader, criterion, device)
            scheduler.step()
            history.append(
                {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
            )
            if valid_metrics["auc"] > best_auc + args.min_delta:
                best_auc = valid_metrics["auc"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    checkpoint,
                )
            else:
                stale += 1
            if stale >= args.patience:
                break
        result = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_valid_auc": best_auc,
            "history": history,
        }
        fold_results.append(result)
        (output_dir / f"fold_{fold}_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "cv_metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "mean_valid_auc": float(
                    np.mean([row["best_valid_auc"] for row in fold_results])
                ),
                "folds": fold_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("niv_mind_c", "niv_mind_f"), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_results") / "comparator_weights",
    )
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
