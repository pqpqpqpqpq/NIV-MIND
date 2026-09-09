"""Feature-by-window perturbation scores for NIV-MIND."""

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


@torch.inference_mode()
def contribution_scores(
    model: torch.nn.Module,
    fine: torch.Tensor,
    coarse: torch.Tensor,
    labels: torch.Tensor,
    *,
    window_size: int = 15,
) -> torch.Tensor:
    if fine.ndim != 3 or fine.shape[2] != 8:
        raise ValueError(f"fine must be [B,T,8], got {tuple(fine.shape)}")
    if fine.shape[1] % window_size:
        raise ValueError("sequence length must be divisible by window_size")
    if coarse.shape != (fine.shape[0], 20):
        raise ValueError(f"coarse must be [B,20], got {tuple(coarse.shape)}")
    if labels.shape != (fine.shape[0],):
        raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")

    model.eval()
    baseline = torch.softmax(model(fine, coarse), dim=1)
    correct_probability = baseline.gather(1, labels[:, None]).squeeze(1)
    windows = fine.shape[1] // window_size
    reference = fine.mean(dim=1, keepdim=True)
    scores = fine.new_empty(fine.shape[0], windows, 8)

    for window in range(windows):
        start = window * window_size
        stop = start + window_size
        for channel in range(8):
            modified = fine.clone()
            modified[:, start:stop, channel] = reference[:, :, channel]
            probability = torch.softmax(model(modified, coarse), dim=1)
            perturbed = probability.gather(1, labels[:, None]).squeeze(1)
            scores[:, window, channel] = correct_probability - perturbed
    return scores


def top_windows(scores: torch.Tensor, *, k: int = 5) -> torch.Tensor:
    if scores.ndim != 3:
        raise ValueError("scores must be [B,W,C]")
    if k < 1 or k > scores.shape[1]:
        raise ValueError("k must be between 1 and the number of windows")
    return torch.topk(scores, k=k, dim=1).indices


def run(args: argparse.Namespace) -> None:
    samples, labels = load_labeled_samples(args.data_dir)
    folds = stratified_folds(samples, labels, n_splits=5, seed=args.seed)
    device = device_from_string(args.device)
    all_scores = np.empty((len(samples), 40, 8), dtype=np.float32)
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
        rows = []
        for fine, coarse, batch_labels in loader:
            rows.append(
                contribution_scores(
                    model,
                    fine.to(device),
                    coarse.to(device),
                    batch_labels.to(device),
                    window_size=15,
                ).cpu().numpy()
            )
        all_scores[valid_indices] = np.concatenate(rows)
        fold_index[valid_indices] = fold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        record_index=np.arange(len(samples), dtype=np.int64),
        label=labels,
        fold=fold_index,
        contribution=all_scores,
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
        default=Path("paper_results") / "temporal_contribution.npz",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["contribution_scores", "top_windows"]
