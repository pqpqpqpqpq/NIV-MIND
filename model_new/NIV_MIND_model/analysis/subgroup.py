"""Subgroup metrics from OOF predictions and anonymous metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import bootstrap_auc_interval, paired_auc_difference_interval


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def subgroup_metrics(
    prediction_rows: list[dict[str, str]],
    metadata_rows: list[dict[str, str]],
    *,
    record_column: str,
    subgroup_column: str,
    label_column: str,
    models: list[str],
    reference_model: str,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    metadata = {
        str(row[record_column]): row[subgroup_column]
        for row in metadata_rows
        if row.get(record_column) and row.get(subgroup_column)
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in prediction_rows:
        group = metadata.get(str(row[record_column]))
        if group:
            grouped.setdefault(group, []).append(row)

    output = []
    for group, rows in sorted(grouped.items()):
        labels = np.asarray([int(row[label_column]) for row in rows])
        if np.unique(labels).size != 2:
            continue
        scores = {
            model: np.asarray([float(row[model]) for row in rows])
            for model in models
        }
        group_result = {
            "subgroup": group,
            "samples": len(rows),
            "failures": int(labels.sum()),
            "successes": int((labels == 0).sum()),
            "models": {},
        }
        for model, values in scores.items():
            result = bootstrap_auc_interval(
                labels,
                values,
                samples=bootstrap,
                seed=seed,
            )
            if model != reference_model:
                result["delta_vs_reference"] = paired_auc_difference_interval(
                    labels,
                    scores[reference_model],
                    values,
                    samples=bootstrap,
                    seed=seed,
                )
            group_result["models"][model] = result
        output.append(group_result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--record-column", default="record_index")
    parser.add_argument("--subgroup-column", default="subgroup")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["NIV-MIND", "NIV-MIND-C", "LightGBM", "HACOR", "ROX"],
    )
    parser.add_argument("--reference-model", default="NIV-MIND")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_results") / "subgroup_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = subgroup_metrics(
        _read(args.predictions),
        _read(args.metadata),
        record_column=args.record_column,
        subgroup_column=args.subgroup_column,
        label_column=args.label_column,
        models=args.models,
        reference_model=args.reference_model,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = ["subgroup_metrics"]
