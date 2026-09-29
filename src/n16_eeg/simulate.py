"""Generate clearly marked synthetic EEG sessions for software-only testing.

Synthetic data are not valid evidence for model accuracy or milestone M4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_session(
    subject_id: str = "SIM01",
    session_id: str = "demo",
    fs: int = 512,
    alert_seconds: float = 20.0,
    drowsy_seconds: float = 25.0,
    seed: int = 16,
    subject_variation: float = 0.12,
) -> pd.DataFrame:
    """Create alert then drowsy EEG-like data with frontal blink transients."""

    if fs < 128 or alert_seconds <= 4 or drowsy_seconds <= 4:
        raise ValueError("fs must be >=128 and each state must last more than 4 seconds")
    rng = np.random.default_rng(seed)
    total_seconds = alert_seconds + drowsy_seconds
    count = int(total_seconds * fs)
    time_s = np.arange(count, dtype=float) / fs
    drowsy = time_s >= alert_seconds

    # Alert: stronger beta. Drowsy: stronger theta/alpha and slower drift.
    variation = 1.0 + rng.normal(0.0, subject_variation)
    signal = 5.0 * variation * np.sin(2 * np.pi * 10.0 * time_s)
    signal += np.where(drowsy, 12.0, 3.0) * variation * np.sin(2 * np.pi * 6.0 * time_s + 0.2)
    signal += np.where(drowsy, 8.0, 3.5) * variation * np.sin(2 * np.pi * 9.0 * time_s + 0.7)
    signal += np.where(drowsy, 2.0, 8.0) / variation * np.sin(2 * np.pi * 18.0 * time_s + 1.1)
    signal += np.where(drowsy, 3.0, 0.8) * np.sin(2 * np.pi * 0.8 * time_s)
    signal += rng.normal(0.0, 2.4, count)

    # Alert blinks are short; drowsy blinks are less frequent and wider.
    blink_centers = list(np.arange(3.0, alert_seconds, 4.5))
    blink_centers += list(np.arange(alert_seconds + 3.0, total_seconds, 6.0))
    for center_s in blink_centers:
        width_s = 0.14 if center_s < alert_seconds else 0.38
        half = max(2, int(width_s * fs / 2))
        center = int(center_s * fs)
        left = max(0, center - half)
        right = min(count, center + half)
        pulse = np.hanning(right - left) * (45.0 if center_s < alert_seconds else 60.0)
        signal[left:right] += pulse

    timestamp_ms = np.rint(time_s * 1000).astype(np.int64)
    kss = np.where(drowsy, 8, 3)
    event = np.full(count, "", dtype=object)
    event[0] = "SIMULATION_START"
    event[min(count - 1, int(alert_seconds * fs))] = "STATE_DROWSY"
    return pd.DataFrame(
        {
            "timestamp_ms": timestamp_ms,
            "subject_id": subject_id,
            "session_id": session_id,
            "raw_eeg": np.rint(signal).astype(np.int16),
            "poor_signal": np.zeros(count, dtype=np.uint8),
            "attention": np.where(drowsy, 35, 70),
            "meditation": np.where(drowsy, 65, 40),
            "kss": kss,
            "event": event,
            "is_synthetic": True,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic N16 EEG demo data")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/demo_session.csv"))
    parser.add_argument("--subject", default="SIM01")
    parser.add_argument("--session", default="demo")
    parser.add_argument("--alert-seconds", type=float, default=20.0)
    parser.add_argument("--drowsy-seconds", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    frame = generate_session(
        subject_id=args.subject,
        session_id=args.session,
        alert_seconds=args.alert_seconds,
        drowsy_seconds=args.drowsy_seconds,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} synthetic samples to {args.output}")
    print("WARNING: synthetic data are for software testing only, not experimental results.")


if __name__ == "__main__":
    main()
