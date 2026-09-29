"""Live NeuroSky inference with rolling windows and audible warnings."""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np

from .blink import blink_window_features
from .demo import risk_score
from .model import LogisticModel
from .signal_features import bandpower_features
from .thinkgear import ThinkGearParser


class LiveDetector:
    """Convert a stream of raw samples into periodic drowsiness decisions."""

    def __init__(
        self,
        model: LogisticModel | None,
        fs: int = 512,
        window_s: float = 4.0,
        step_s: float = 1.0,
        threshold: float | None = None,
        consecutive: int = 2,
    ) -> None:
        if fs <= 90 or window_s <= 0 or step_s <= 0 or consecutive < 1:
            raise ValueError("Invalid live detector parameters")
        self.model = model
        self.fs = fs
        self.window_samples = int(window_s * fs)
        self.step_samples = int(step_s * fs)
        self.threshold = threshold if threshold is not None else (model.threshold if model else 0.85)
        self.consecutive = consecutive
        self.samples: deque[float] = deque(maxlen=self.window_samples)
        self.signal_quality: deque[float] = deque(maxlen=self.window_samples)
        self.baseline_ratios: deque[float] = deque(maxlen=10)
        self.samples_since_decision = 0
        self.alert_streak = 0

    def add_sample(self, sample: float, poor_signal: float) -> dict[str, object] | None:
        self.samples.append(float(sample))
        self.signal_quality.append(float(poor_signal))
        self.samples_since_decision += 1
        if len(self.samples) < self.window_samples:
            return None
        if self.samples_since_decision < self.step_samples:
            return None
        self.samples_since_decision = 0

        started_ns = time.perf_counter_ns()
        poor = np.asarray(self.signal_quality, dtype=float)
        if float(np.mean(poor > 50)) > 0.20:
            self.alert_streak = 0
            return {
                "timestamp_ms": int(time.time_ns() // 1_000_000),
                "state": "TIN_HIEU_KEM",
                "risk": "",
                "alarm": 0,
                "poor_signal_fraction": round(float(np.mean(poor > 50)), 4),
                "latency_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
            }

        values = np.asarray(self.samples, dtype=float)
        spectral = bandpower_features(values, fs=float(self.fs))
        blink = blink_window_features(values, fs=float(self.fs))
        if len(self.baseline_ratios) < self.baseline_ratios.maxlen:
            self.baseline_ratios.append(spectral["theta_alpha"])
        baseline = float(np.median(self.baseline_ratios))
        risk = (
            self.model.predict_feature_dict({**spectral, **blink})
            if self.model is not None
            else risk_score(spectral, blink, baseline)
        )
        self.alert_streak = self.alert_streak + 1 if risk >= self.threshold else 0
        alarm = self.alert_streak >= self.consecutive
        return {
            "timestamp_ms": int(time.time_ns() // 1_000_000),
            "state": "CANH_BAO" if alarm else ("NGUY_CO" if risk >= self.threshold else "TINH"),
            "risk": round(float(risk), 4),
            "alarm": int(alarm),
            "poor_signal_fraction": round(float(np.mean(poor > 50)), 4),
            "theta_alpha": round(float(spectral["theta_alpha"]), 4),
            "blink_count": int(blink["blink_count"]),
            "long_blink_count": int(blink["long_blink_count"]),
            "latency_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        }


def _beep() -> None:
    try:
        import winsound

        winsound.Beep(1250, 250)
    except (ImportError, RuntimeError):
        print("\a", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live N16 drowsiness inference")
    parser.add_argument("--port", required=True, help="Bluetooth COM port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--model", type=Path, default=Path("output/model/n16_logistic_model.json"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--duration-min", type=float, default=30.0)
    parser.add_argument("--log-output", type=Path, default=Path("output/live/live_log.csv"))
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install hardware extra first: python -m pip install -e .[hardware]") from exc

    model = LogisticModel.load(args.model)
    detector = LiveDetector(model, threshold=args.threshold)
    packet_parser = ThinkGearParser()
    poor_signal = 255
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_ms", "state", "risk", "alarm", "poor_signal_fraction",
        "theta_alpha", "blink_count", "long_blink_count", "latency_ms",
    ]
    started = time.monotonic()
    with serial.Serial(args.port, args.baud, timeout=0.25) as stream, args.log_output.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        while time.monotonic() - started < args.duration_min * 60:
            for packet in packet_parser.feed(stream.read(4096)):
                if packet.poor_signal is not None:
                    poor_signal = packet.poor_signal
                for sample in packet.raw_eeg:
                    decision = detector.add_sample(sample, poor_signal)
                    if decision is None:
                        continue
                    writer.writerow(decision)
                    handle.flush()
                    print(
                        f"risk={decision.get('risk', '')}  state={decision['state']}  "
                        f"latency={decision['latency_ms']} ms"
                    )
                    if decision["alarm"] and not args.no_sound:
                        _beep()


if __name__ == "__main__":
    main()
