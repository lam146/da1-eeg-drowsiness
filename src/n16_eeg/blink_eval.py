"""Evaluate detected blink events against manually annotated timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .blink import detect_blinks


def match_event_pairs(
    predicted_seconds: list[float] | np.ndarray,
    reference_seconds: list[float] | np.ndarray,
    tolerance_s: float = 0.20,
) -> list[tuple[float, float]]:
    """Maximize one-to-one matches, then minimize total absolute timing error."""

    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    predicted = np.asarray(predicted_seconds, dtype=float)
    reference = np.asarray(reference_seconds, dtype=float)
    if (predicted.ndim != 1 or reference.ndim != 1
            or not np.isfinite(predicted).all() or not np.isfinite(reference).all()):
        raise ValueError("Event times must be finite one-dimensional arrays")
    predicted, reference = np.sort(predicted), np.sort(reference)
    # Ordered matching prevents a nearby prediction from consuming a reference
    # needed by a later prediction. Costs prioritize cardinality over time error.
    costs = [[(0, 0.0)] * (len(reference) + 1) for _ in range(len(predicted) + 1)]
    steps = np.zeros((len(predicted) + 1, len(reference) + 1), dtype=np.uint8)
    for i, prediction in enumerate(predicted, 1):
        for j, target in enumerate(reference, 1):
            options = [(costs[i - 1][j], 1), (costs[i][j - 1], 2)]
            error = abs(float(prediction - target))
            if error <= tolerance_s:
                count, total_error = costs[i - 1][j - 1]
                options.append(((count - 1, total_error + error), 3))
            costs[i][j], steps[i, j] = min(options, key=lambda item: item[0])
    pairs = []
    i, j = len(predicted), len(reference)
    while i and j:
        step = steps[i, j]
        if step == 3:
            pairs.append((float(predicted[i - 1]), float(reference[j - 1])))
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def match_events(
    predicted_seconds: list[float] | np.ndarray,
    reference_seconds: list[float] | np.ndarray,
    tolerance_s: float = 0.20,
) -> dict[str, float | int]:
    """Score event matches using a maximum-cardinality one-to-one assignment."""
    pairs = match_event_pairs(predicted_seconds, reference_seconds, tolerance_s)
    errors = [abs(prediction - reference) for prediction, reference in pairs]
    true_positive = len(pairs)
    predicted, reference = predicted_seconds, reference_seconds
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
    parser.add_argument("--openvibe-video-offset-ms", type=float,
                        help="Video time corresponding to the first sample of an OpenViBE CSV")
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/blink_metrics.json"))
    args = parser.parse_args()

    session = pd.read_csv(args.session)
    annotations = pd.read_csv(args.annotations)
    if "timestamp_ms" not in annotations:
        raise SystemExit("Annotations need a timestamp_ms column")
    if args.openvibe_video_offset_ms is not None:
        if not {"Electrode", "Time:512Hz"}.issubset(session.columns) or session.empty:
            raise SystemExit("OpenViBE session needs Electrode and Time:512Hz")
        timeline = pd.to_numeric(session["Time:512Hz"], errors="raise").to_numpy(float)
        if args.fs != 512 or not np.isfinite(timeline).all() or not np.allclose(
            np.diff(timeline), 1 / 512, rtol=0, atol=1e-7
        ):
            raise SystemExit("OpenViBE blink evaluation requires a continuous 512 Hz sample grid")
        signal_column = "Electrode"
        origin_ms = args.openvibe_video_offset_ms
    else:
        if not {"raw_eeg", "timestamp_ms"}.issubset(session.columns) or session.empty:
            raise SystemExit("Session needs raw_eeg and timestamp_ms")
        signal_column = "raw_eeg"
        origin_ms = float(pd.to_numeric(session["timestamp_ms"], errors="raise").iloc[0])
    values = pd.to_numeric(session[signal_column], errors="raise").to_numpy(float)
    reference = (
        pd.to_numeric(annotations["timestamp_ms"], errors="raise").to_numpy(float) - origin_ms
    ) / 1000.0
    if not np.isfinite(origin_ms) or not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise SystemExit("Signal, time origin and annotations must be finite")
    if args.openvibe_video_offset_ms is not None and (
        (reference < 0).any() or (reference >= len(values) / args.fs).any()
    ):
        raise SystemExit("Video annotations must fall within the aligned EEG recording")
    predicted = [event.time_seconds for event in detect_blinks(values, fs=args.fs)]
    metrics = match_events(predicted, reference, args.tolerance_ms / 1000.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
