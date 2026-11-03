"""Explicitly map an OpenViBE signal CSV without guessing channels or units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def convert_frame(frame, time_column, electrode_column, subject, session, unit,
                  fs=512, time_unit="seconds", quality_column=None):
    if fs != 512:
        raise ValueError("This N16 import expects 512 Hz; do not relabel a different sample rate")
    if unit not in {"uV", "count"} or time_unit not in {"seconds", "milliseconds"}:
        raise ValueError("Declare exact signal and timestamp units")
    if not subject.strip() or not session.strip() or frame.empty:
        raise ValueError("Require nonempty data, subject and session")
    if len({time_column, electrode_column, quality_column} - {None}) != (3 if quality_column else 2):
        raise ValueError("Time, electrode and quality must be different columns")
    seconds = pd.to_numeric(frame[time_column], errors="raise").to_numpy(float)
    if time_unit == "milliseconds":
        seconds = seconds / 1000
    raw = pd.to_numeric(frame[electrode_column], errors="raise").to_numpy(float)
    if len(raw) < 2 or not np.isfinite(raw).all() or not np.isfinite(seconds).all():
        raise ValueError("Need at least two finite signal/time values")
    delta = np.diff(seconds)
    # A sample-wise signal CSV must have one increasing timestamp per sample.
    if np.any(delta <= 0) or not np.allclose(delta, 1 / fs, rtol=0.05, atol=1e-6):
        raise ValueError("Timestamps are not continuous sample-wise 512 Hz; check CSV signal/chunk layout")
    quality = np.full(len(raw), np.nan)
    if quality_column:
        quality = pd.to_numeric(frame[quality_column], errors="raise").to_numpy(float)
        if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 255)):
            raise ValueError("Quality must be verified ThinkGear poorSignal in [0,255]")
    return pd.DataFrame({
        "timestamp_ms": seconds * 1000, "sample_index": np.arange(len(raw)),
        "subject_id": subject, "session_id": session, "raw_eeg": raw,
        "poor_signal": quality, "kss": np.nan, "event": "",
        "signal_unit": unit, "dataset_id": "openvibe", "fs_hz": fs,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--separator", default=";")
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--electrode-column", required=True)
    parser.add_argument("--time-unit", choices=["seconds", "milliseconds"], default="seconds")
    parser.add_argument("--signal-unit", choices=["uV", "count"], required=True)
    parser.add_argument("--quality-column", help="ONLY if verified to be raw ThinkGear poorSignal, not a percent")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".source.json")
    if args.output.exists() or sidecar.exists():
        parser.error("Output exists; choose a new filename")
    result = convert_frame(pd.read_csv(args.input, sep=args.separator), args.time_column,
                           args.electrode_column, args.subject, args.session, args.signal_unit,
                           time_unit=args.time_unit, quality_column=args.quality_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    sidecar.write_text(json.dumps({"source": str(args.input.resolve()), "fs_hz": 512,
        "signal_unit": args.signal_unit, "electrode_column": args.electrode_column,
        "quality_column": args.quality_column,
        "note": "No unit conversion or labels invented. Missing quality must be annotated before featurizing."},
        indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} samples. Annotate KSS and verified quality before featurizing.")


if __name__ == "__main__":
    main()
