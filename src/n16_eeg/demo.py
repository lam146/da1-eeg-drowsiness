"""Software-only real-time replay demo with an explainable heuristic baseline."""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .alarm import AlarmPolicy
from .blink import blink_window_features
from .model import LogisticModel, load_model, risk_score
from .rbf_model import RbfModel
from .signal_features import bandpower_features


def _beep(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import winsound

        winsound.Beep(1250, 250)
    except (ImportError, RuntimeError):
        print("\a", end="", flush=True)


def replay(
    frame: pd.DataFrame,
    output_log: Path,
    fs: int = 512,
    window_s: float = 4.0,
    step_s: float = 1.0,
    speed: float = 0.0,
    threshold: float = 0.85,
    consecutive: int = 2,
    sound: bool = False,
    model: LogisticModel | RbfModel | None = None,
    verbose: bool = True,
    release_consecutive: int = 1,
) -> pd.DataFrame:
    required = {"timestamp_ms", "raw_eeg", "poor_signal"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if speed < 0:
        raise ValueError("speed must be >= 0")

    window = int(window_s * fs)
    step = int(step_s * fs)
    if window < 8 or step < 1 or consecutive < 1:
        raise ValueError("Invalid replay window, step, or confirmation count")
    baseline_ratios: deque[float] = deque(maxlen=10)
    policy = AlarmPolicy(threshold, consecutive, release_consecutive)
    rows: list[dict[str, object]] = []
    output_log.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(frame) - window + 1, step):
        started_ns = time.perf_counter_ns()
        block = frame.iloc[start : start + window]
        poor = pd.to_numeric(block["poor_signal"], errors="coerce").fillna(255).to_numpy(float)
        if float(np.mean(poor > 50)) > 0.20:
            policy.reset()
            if speed > 0:
                time.sleep(step_s / speed)
            continue
        samples = pd.to_numeric(block["raw_eeg"], errors="coerce").to_numpy(float)
        spectral = bandpower_features(samples, fs=float(fs))
        blink = blink_window_features(samples, fs=float(fs))
        if len(baseline_ratios) < baseline_ratios.maxlen:
            baseline_ratios.append(spectral["theta_alpha"])
        baseline = float(np.median(baseline_ratios))
        combined_features = {**spectral, **blink}
        risk = (
            model.predict_feature_dict(combined_features)
            if model is not None
            else risk_score(spectral, blink, baseline)
        )
        was_active = policy.active
        alarm = policy.update(risk)
        if alarm and not was_active:
            _beep(sound)
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        kss = block["kss"].iloc[-1] if "kss" in block else ""
        state = "CANH_BAO" if alarm else ("NGUY_CO" if risk >= threshold else "TINH")
        row = {
            "window_end_ms": int(block["timestamp_ms"].iloc[-1]),
            "risk": round(risk, 4),
            "state": state,
            "alarm": int(alarm),
            "theta_alpha": round(spectral["theta_alpha"], 4),
            "blink_count": int(blink["blink_count"]),
            "long_blink_count": int(blink["long_blink_count"]),
            "latency_ms": round(latency_ms, 3),
            "kss_reference": kss,
            "model_mode": model.model_type if model is not None else "heuristic_demo",
        }
        rows.append(row)
        if verbose:
            print(
                f"t={row['window_end_ms']/1000:6.1f}s  risk={risk:0.2f}  "
                f"theta/alpha={spectral['theta_alpha']:0.2f}  "
                f"blink={int(blink['blink_count'])}  {state}"
            )
        if speed > 0:
            time.sleep(step_s / speed)

    result = pd.DataFrame(rows)
    result.to_csv(output_log, index=False, quoting=csv.QUOTE_MINIMAL)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay EEG CSV as a real-time N16 demo")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-log", type=Path, default=Path("output/demo/replay_log.csv"))
    parser.add_argument("--speed", type=float, default=0.0, help="0=no wait, 1=real time, 10=10x")
    parser.add_argument("--window-s", type=float, default=4.0, help="Use the window length from model training")
    parser.add_argument("--step-s", type=float, default=1.0, help="Seconds between predictions")
    parser.add_argument("--consecutive", type=int, default=2, help="Risk predictions required to alert")
    parser.add_argument("--release-consecutive", type=int, default=1, help="Low-risk predictions required to clear the alert")
    parser.add_argument(
        "--threshold", type=float, help="Override saved model threshold (heuristic default: 0.85)"
    )
    parser.add_argument("--sound", action="store_true")
    parser.add_argument("--model", type=Path, help="Saved JSON logistic model")
    args = parser.parse_args()
    model = load_model(args.model) if args.model else None
    threshold = (
        args.threshold
        if args.threshold is not None
        else (model.threshold if model is not None else 0.85)
    )
    result = replay(
        pd.read_csv(args.input),
        output_log=args.output_log,
        window_s=args.window_s,
        step_s=args.step_s,
        consecutive=args.consecutive,
        release_consecutive=args.release_consecutive,
        speed=args.speed,
        threshold=threshold,
        sound=args.sound,
        model=model,
    )
    if result.empty:
        raise SystemExit("No valid windows were produced")
    print(f"wrote {len(result)} replay windows to {args.output_log}")
    print(f"alarms={int(result['alarm'].sum())}; latency_p95_ms={result['latency_ms'].quantile(.95):.3f}")
    if model is None:
        print("NOTE: heuristic replay is an engineering demo, not the final trained classifier.")


if __name__ == "__main__":
    main()
