"""Direct inference for a registered NIV-MIND ablation variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .checkpoint import StrictCheckpointMixin
    from .registry import EXPERIMENTS, build_model, get_spec, transform_fine
except ImportError:
    workspace_root = Path(__file__).resolve().parents[2]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from model_new.NIV_MIND_ablation_models.checkpoint import StrictCheckpointMixin
    from model_new.NIV_MIND_ablation_models.registry import (
        EXPERIMENTS,
        build_model,
        get_spec,
        transform_fine,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ablation checkpoint.")
    parser.add_argument("--group", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    get_spec(args.group, args.variant)
    if not args.smoke_test and args.input is None:
        parser.error("provide --input or --smoke-test")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model = build_model(args.group, args.variant)
    if not isinstance(model, StrictCheckpointMixin):
        raise TypeError("Constructed model has no strict checkpoint interface")
    report = model.load_final_checkpoint(args.checkpoint)
    model.to(device).eval()
    spec = get_spec(args.group, args.variant)
    if args.smoke_test:
        fine = np.zeros((1, 600, 8), dtype=np.float32)
        coarse = np.zeros((1, 20), dtype=np.float32)
    else:
        with np.load(args.input, allow_pickle=False) as data:
            fine = np.asarray(data["fine"], dtype=np.float32)
            coarse = np.asarray(data["coarse"], dtype=np.float32)
        if fine.ndim == 2:
            fine = fine[None]
        if coarse.ndim == 1:
            coarse = coarse[None]
    fine = transform_fine(fine, spec)
    if coarse.ndim != 2 or coarse.shape != (fine.shape[0], 20):
        raise ValueError(f"coarse must be [N,20], got {coarse.shape}")
    with torch.inference_mode():
        probabilities = model.predict_proba(
            torch.from_numpy(fine).to(device),
            torch.from_numpy(coarse).to(device),
        ).cpu()
    print(
        json.dumps(
            {
                "group": args.group,
                "variant": args.variant,
                "checkpoint": report.to_dict(),
                "probabilities": probabilities.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
