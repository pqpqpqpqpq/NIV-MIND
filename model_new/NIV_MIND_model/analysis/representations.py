"""Extract fold-specific NIV-MIND feature representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_new.NIV_MIND_model import NIV_MIND
from model_new.NIV_MIND_model.dataset import NIVMINDDataset, load_labeled_samples

from .comparison import device_from_string
from .splits import stratified_folds


def run(args: argparse.Namespace) -> None:
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(samples, labels, n_splits=5, seed=args.seed)
    device = device_from_string(args.device)
    fine_embeddings = np.empty((len(samples), 512), dtype=np.float32)
    coarse_embeddings = np.empty((len(samples), 128), dtype=np.float32)
    fold_index = np.empty(len(samples), dtype=np.int64)

    for fold, (_, valid_indices) in enumerate(folds):
        dataset = NIVMINDDataset(
            [samples[index] for index in valid_indices],
            labels[valid_indices],
            training=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model, _ = NIV_MIND.from_checkpoint(
            args.weights / f"fold_{fold}_best.pth",
            device=device,
        )
        fine_values = []
        coarse_values = []
        with torch.inference_mode():
            for fine, coarse, _ in loader:
                fine_values.append(model.encode_fine(fine.to(device)).cpu().numpy())
                coarse_values.append(model.encode_coarse(coarse.to(device)).cpu().numpy())
        fine_embeddings[valid_indices] = np.concatenate(fine_values)
        coarse_embeddings[valid_indices] = np.concatenate(coarse_values)
        fold_index[valid_indices] = fold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        record_index=np.arange(len(samples), dtype=np.int64),
        label=labels,
        fold=fold_index,
        fine_embedding=fine_embeddings,
        coarse_embedding=coarse_embeddings,
    )


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
        default=Path("paper_results") / "oof_representations.npz",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
