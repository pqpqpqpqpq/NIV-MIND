"""Run NIV-MIND inference from a trained final PTH checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

try:  # python -m model_new.NIV_MIND_model.predict
    from .model import NIV_MIND
except ImportError:  # python predict.py
    from model import NIV_MIND


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _load_input(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "fine" not in data or "coarse" not in data:
            raise KeyError("Input NPZ must contain arrays named 'fine' and 'coarse'.")
        fine = np.asarray(data["fine"], dtype=np.float32)
        coarse = np.asarray(data["coarse"], dtype=np.float32)

    if fine.ndim == 2:
        fine = fine[None, ...]
    if coarse.ndim == 1:
        coarse = coarse[None, ...]
    if fine.ndim != 3 or fine.shape[1:] != (600, 8):
        raise ValueError(f"fine must be [N, 600, 8], got {fine.shape}")
    if coarse.ndim != 2 or coarse.shape[1] != 20:
        raise ValueError(f"coarse must be [N, 20], got {coarse.shape}")
    if fine.shape[0] != coarse.shape[0]:
        raise ValueError("fine and coarse must contain the same number of samples.")
    if not np.isfinite(fine).all() or not np.isfinite(coarse).all():
        raise ValueError("Input contains NaN or infinity; apply training-time preprocessing first.")
    return fine, coarse


@torch.inference_mode()
def _predict(
    model: NIV_MIND,
    fine: np.ndarray,
    coarse: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(fine), batch_size):
        stop = min(start + batch_size, len(fine))
        fine_batch = torch.from_numpy(fine[start:stop]).to(device)
        coarse_batch = torch.from_numpy(coarse[start:stop]).to(device)
        outputs.append(model.predict_proba(fine_batch, coarse_batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _write_predictions(path: Path, probabilities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_index", "predicted_class", "prob_success", "prob_failure"]
        )
        for index, (prob_success, prob_failure) in enumerate(probabilities):
            writer.writerow(
                [
                    index,
                    int(prob_failure >= prob_success),
                    f"{float(prob_success):.10f}",
                    f"{float(prob_failure):.10f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained final NIV-MIND PTH and run inference directly."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        help="NPZ containing normalized fine [N,600,8] and coarse [N,20].",
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the PTH and run one zero-input sample without an NPZ file.",
    )
    args = parser.parse_args()
    if not args.smoke_test and args.input is None:
        parser.error("provide --input or use --smoke-test")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    model, report = NIV_MIND.from_checkpoint(args.checkpoint, device=device)

    if args.smoke_test:
        fine = np.zeros((1, 600, 8), dtype=np.float32)
        coarse = np.zeros((1, 20), dtype=np.float32)
    else:
        fine, coarse = _load_input(args.input)

    probabilities = _predict(
        model,
        fine,
        coarse,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "checkpoint": report.to_dict(),
        "device": str(device),
        "samples": len(probabilities),
    }

    if args.smoke_test:
        result["probabilities"] = probabilities.tolist()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _write_predictions(args.output, probabilities)
        result["output"] = str(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
