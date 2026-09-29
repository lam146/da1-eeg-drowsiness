"""Convert an annotated raw-session CSV into fixed-window ML features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .blink import blink_window_features
from .signal_features import bandpower_features


def kss_to_label(kss: float) -> int | None:
    if 1 <= kss <= 5:
        return 0
    if 7 <= kss <= 9:
        return 1
    return None


def featurize_session(
    frame: pd.DataFrame,
    fs: int = 512,
    window_s: float = 4.0,
    step_s: float = 1.0,
    poor_signal_limit: float = 50.0,
    bad_fraction_limit: float = 0.20,
) -> pd.DataFrame:
    required = {"timestamp_ms", "subject_id", "session_id", "raw_eeg", "poor_signal", "kss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = frame.sort_values("timestamp_ms").reset_index(drop=True).copy()
    ordered["kss"] = pd.to_numeric(ordered["kss"], errors="coerce").ffill(limit=fs * 5 * 60)
    window = int(window_s * fs)
    step = int(step_s * fs)
    rows: list[dict[str, object]] = []

    for start in range(0, len(ordered) - window + 1, step):
        block = ordered.iloc[start : start + window]
        poor = pd.to_numeric(block["poor_signal"], errors="coerce").fillna(255).to_numpy(float)
        if float(np.mean(poor > poor_signal_limit)) > bad_fraction_limit:
            continue
        label = kss_to_label(float(block["kss"].iloc[-1]))
        if label is None:
            continue
        samples = pd.to_numeric(block["raw_eeg"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(samples)):
            continue

        record: dict[str, object] = {
            "subject_id": str(block["subject_id"].iloc[0]),
            "session_id": str(block["session_id"].iloc[0]),
            "window_start_ms": int(block["timestamp_ms"].iloc[0]),
            "window_end_ms": int(block["timestamp_ms"].iloc[-1]),
            "label": label,
            "source_is_synthetic": bool(
                "is_synthetic" in block and block["is_synthetic"].astype(bool).all()
            ),
        }
        record.update(bandpower_features(samples, fs=float(fs)))
        record.update(blink_window_features(samples, fs=float(fs)))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract N16 EEG/blink features")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fs", type=int, default=512)
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    args = parser.parse_args()
    features = featurize_session(
        pd.read_csv(args.input), fs=args.fs, window_s=args.window_s, step_s=args.step_s
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"wrote {len(features)} windows to {args.output}")


if __name__ == "__main__":
    main()
