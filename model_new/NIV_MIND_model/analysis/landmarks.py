"""Select initiation-aligned and endpoint-aligned NIV observation windows."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def select_landmark_window(
    windows: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    landmark_hours: float,
    tolerance_hours: float = 0.5,
    start_key: str = "niv_start",
    endpoint_key: str = "niv_endpoint",
    window_end_key: str = "window_end",
) -> Mapping[str, Any] | None:
    if reference not in ("initiation", "endpoint"):
        raise ValueError("reference must be initiation or endpoint")
    candidates = []
    for window in windows:
        window_end = parse_time(window[window_end_key])
        if reference == "initiation":
            target = parse_time(window[start_key])
            elapsed = (window_end - target).total_seconds() / 3600.0
            difference = landmark_hours - elapsed
            eligible = 0.0 <= difference <= tolerance_hours
        else:
            target = parse_time(window[endpoint_key])
            remaining = (target - window_end).total_seconds() / 3600.0
            difference = abs(remaining - landmark_hours)
            eligible = difference <= tolerance_hours
        if eligible:
            candidates.append((difference, window_end, window))
    if not candidates:
        return None
    candidates.sort(key=lambda value: (value[0], -value[1].timestamp()))
    return candidates[0][2]


def select_landmarks(
    windows: Sequence[Mapping[str, Any]],
    landmarks: Sequence[float],
    *,
    reference: str,
    series_key: str = "series_index",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for window in windows:
        grouped.setdefault(str(window[series_key]), []).append(window)
    selected = []
    for series_index, series_windows in grouped.items():
        for landmark in landmarks:
            window = select_landmark_window(
                series_windows,
                reference=reference,
                landmark_hours=landmark,
                **kwargs,
            )
            if window is not None:
                selected.append(
                    {
                        "series_index": series_index,
                        "reference": reference,
                        "landmark_hours": landmark,
                        **dict(window),
                    }
                )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--reference", choices=("initiation", "endpoint"), required=True)
    parser.add_argument("--landmarks", type=float, nargs="+", required=True)
    parser.add_argument("--tolerance-hours", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.windows.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_landmarks(
        rows,
        args.landmarks,
        reference=args.reference,
        tolerance_hours=args.tolerance_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = ["select_landmark_window", "select_landmarks"]
