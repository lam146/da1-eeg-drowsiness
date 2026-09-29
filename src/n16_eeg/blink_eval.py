"""Evaluate detected blink events against manually annotated timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .blink import detect_blinks


def match_events(
    predicted_seconds: list[float] | np.ndarray,
    reference_seconds: list[float] | np.ndarray,
    tolerance_s: float = 0.20,
) -> dict[str, float | int]:
    """Greedily match each prediction to at most one reference event."""

    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    predicted = np.sort(np.asarray(predicted_seconds, dtype=float))
    reference = np.sort(np.asarray(reference_seconds, dtype=float))
    used: set[int] = set()
    errors: list[float] = []
    for prediction in predicted:
        candidates = [
            (abs(float(prediction - target)), index)
            for index, target in enumerate(reference)
            if index not in used and abs(float(prediction - target)) <= tolerance_s
        ]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            errors.append(error)
    true_positive = len(errors)
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    count_accuracy = 1.0 - abs(len(predicted) - len(reference)) / max(len(reference), 1)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "count_accuracy": float(max(0.0, count_accuracy)),
        "mean_absolute_timing_error_ms": float(np.mean(errors) * 1000) if errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate N16 blink detection")
    parser.add_argument("session", type=Path, help="Raw session CSV")
    parser.add_argument("annotations", type=Path, help="CSV containing timestamp_ms")
    parser.add_argument("--fs", type=int, default=512)
    parser.add_argument("--tolerance-ms", type=float, default=200.0)
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/blink_metrics.json"))
    args = parser.parse_args()

    session = pd.read_csv(args.session)
    annotations = pd.read_csv(args.annotations)
    if "raw_eeg" not in session or "timestamp_ms" not in annotations:
        raise SystemExit("Session needs raw_eeg and annotations need timestamp_ms")
    values = pd.to_numeric(session["raw_eeg"], errors="raise").to_numpy(float)
    predicted = [event.time_seconds for event in detect_blinks(values, fs=args.fs)]
    origin_ms = float(pd.to_numeric(session["timestamp_ms"], errors="raise").iloc[0])
    reference = (
        pd.to_numeric(annotations["timestamp_ms"], errors="raise").to_numpy(float) - origin_ms
    ) / 1000.0
    metrics = match_events(predicted, reference, args.tolerance_ms / 1000.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
