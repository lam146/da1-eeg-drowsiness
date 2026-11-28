"""Live NeuroSky inference with rolling windows and audible warnings."""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from math import isfinite
from threading import Event, Thread

import numpy as np

from .alarm import AlarmPolicy
from .blink import blink_window_features
from .continuity import SERIAL_POLL_S, RawStreamWatchdog
from .model import LogisticModel, load_model, risk_score
from .rbf_model import RbfModel
from .quality import RawArtifactGuard
from .session_recording import SessionRecording
from .signal_features import bandpower_features
from .thinkgear import ThinkGearParser


class LiveDetector:
    """Convert a stream of raw samples into periodic drowsiness decisions."""

    def __init__(
        self,
        model: LogisticModel | RbfModel | None,
        fs: int = 512,
        window_s: float = 4.0,
        step_s: float = 1.0,
        threshold: float | None = None,
        consecutive: int = 2,
        release_consecutive: int = 1,
        support_threshold: float | None = None,
        support_consecutive: int = 1,
        screen_raw: bool = False,
    ) -> None:
        if fs <= 90 or window_s * fs < 8 or step_s * fs < 1 or consecutive < 1:
            raise ValueError("Invalid live detector parameters")
        self.model = model
        self.spectral_estimator = getattr(model, "spectral_estimator", "periodogram")
        if self.spectral_estimator != "periodogram" and (fs != 512 or window_s != 3):
            raise ValueError("The N16 Welch model requires a 3-second window at 512 Hz")
        self.fs = fs
        self.window_samples = int(window_s * fs)
        self.step_samples = int(step_s * fs)
        self.threshold = threshold if threshold is not None else (model.threshold if model else 0.85)
        self.policy = AlarmPolicy(self.threshold, consecutive, release_consecutive,
                                  support_threshold=support_threshold,
                                  support_consecutive=support_consecutive)
        self.samples: deque[float] = deque(maxlen=self.window_samples)
        self.signal_quality: deque[float] = deque(maxlen=self.window_samples)
        self.baseline_ratios: deque[float] = deque(maxlen=10)
        self.samples_since_decision = 0
        self.screen_raw = screen_raw
        self.raw_guard = RawArtifactGuard(self.window_samples, fs, screen_raw)

    def reset(self) -> None:
        self.samples.clear()
        self.signal_quality.clear()
        self.baseline_ratios.clear()
        self.samples_since_decision = 0
        self.policy.reset()
        self.raw_guard.reset()

    def add_sample(self, sample: float, poor_signal: float, *, raw_artifact: bool = False) -> dict[str, object] | None:
        sample = float(sample)
        poor_signal = float(poor_signal)
        if not isfinite(poor_signal) or not 0 <= poor_signal <= 255:
            poor_signal = 255
        raw_artifact = self.raw_guard.update(sample) or raw_artifact
        self.samples.append(float(sample))
        self.signal_quality.append(float(poor_signal))
        self.samples_since_decision += 1
        if len(self.samples) < self.window_samples:
            return None
        if self.samples_since_decision < self.step_samples:
            return None
        self.samples_since_decision = 0

        started_ns = time.perf_counter_ns()
        poor_fraction = float(np.mean(np.asarray(self.signal_quality, dtype=float) > 50))
        if raw_artifact or poor_signal > 50 or poor_fraction > 0.20:
            self.policy.reset()
            return {
                "timestamp_ms": int(time.time_ns() // 1_000_000),
                "state": "TIN_HIEU_KEM",
                "risk": "",
                "alarm": 0,
                "poor_signal_fraction": round(poor_fraction, 4),
                "raw_artifact": int(raw_artifact),
                "latency_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
            }

        values = np.asarray(self.samples, dtype=float)
        spectral = bandpower_features(values, fs=float(self.fs), estimator=self.spectral_estimator)
        blink = blink_window_features(values, fs=float(self.fs))
        if self.model is not None:
            risk = self.model.predict_feature_dict({**spectral, **blink})
        else:
            if len(self.baseline_ratios) < self.baseline_ratios.maxlen:
                self.baseline_ratios.append(spectral["theta_alpha"])
            baseline = float(np.median(self.baseline_ratios))
            risk = risk_score(spectral, blink, baseline)
        alarm = self.policy.update(risk)
        return {
            "timestamp_ms": int(time.time_ns() // 1_000_000),
            "state": "CANH_BAO" if alarm else ("NGUY_CO" if risk >= self.threshold else "TINH"),
            "risk": round(float(risk), 4),
            "alarm": int(alarm),
            "poor_signal_fraction": round(poor_fraction, 4),
            "raw_artifact": 0,
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


def run_live(
    port: str,
    detector: LiveDetector,
    log_output: Path,
    *,
    baud: int = 57600,
    duration_min: float | None = None,
    sound: bool | Event = True,
    stop: Event | None = None,
    on_decision: Callable[[dict[str, object]], None] | None = None,
    raw_output: Path | None = None,
    subject_id: str = 'UNASSIGNED',
    session_id: str = 'LIVE',
    observations=None,
    on_status: Callable[[dict[str, object]], None] | None = None,
    on_observation: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Read a serial connection and stream logged decisions to the caller."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Install hardware extra first: python -m pip install -e .[hardware]") from exc

    packet_parser = ThinkGearParser()
    detector.raw_guard.screen_raw = detector.screen_raw
    detector.reset()
    poor_signal = 255
    log_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_ms", "state", "risk", "alarm", "poor_signal_fraction",
        "theta_alpha", "blink_count", "long_blink_count", "latency_ms", "audio_requested_ms", "raw_artifact",
        "sample_index", "raw_segment", "episode_id",
    ]
    with ExitStack() as resources, serial.Serial(port, baud, timeout=SERIAL_POLL_S) as stream, log_output.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        handle.flush()

        def publish(decision):
            writer.writerow(decision)
            handle.flush()
            if on_decision is not None:
                on_decision(decision)

        alarming = False
        episodes = 0
        started = time.monotonic()
        watchdog = RawStreamWatchdog(started)
        quality_received = None
        raw_count, segment = 0, 0
        last_status = started
        recording = None
        if raw_output is not None:
            recording = resources.enter_context(SessionRecording(raw_output, subject_id, session_id,
                detector.fs, observations, dict(model_type=detector.model.model_type if detector.model else 'heuristic',
                    model_sha256=getattr(detector, 'model_sha256', None), threshold=detector.threshold,
                    window_samples=detector.window_samples, step_samples=detector.step_samples,
                    spectral_estimator=detector.spectral_estimator,
                    consecutive=detector.policy.consecutive, release_consecutive=detector.policy.release_consecutive,
                    support_threshold=detector.policy.support_threshold,
                    support_consecutive=detector.policy.support_consecutive, fs_hz=detector.fs,
                    feature_names=detector.model.feature_names if detector.model else [],
                    screen_raw=detector.screen_raw, quality_expiry_s=3.), on_observation))

        def publish_status(received):
            nonlocal last_status
            if received - last_status < .25:
                return
            last_status = received
            if recording:
                recording.flush()
            quality = poor_signal if quality_received is not None and received - quality_received <= 3. else 255
            if on_status:
                on_status(dict(elapsed_s=received - started, samples=raw_count, poor_signal=quality,
                               trace=list(detector.samples), segment=segment))

        while stop is None or not stop.is_set():
            if recording:
                recording.collect_observations()
            if duration_min is not None and time.monotonic() - started >= duration_min * 60:
                break
            serial_error = None
            try:
                read_size = max(1, min(stream.in_waiting, 4096))
                chunk = stream.read(read_size)
            except OSError as exc:
                serial_error = exc
                chunk = b""
            received = time.monotonic()
            packets = packet_parser.feed(chunk)
            port_disconnected = serial_error is not None and not watchdog.stalled
            if watchdog.check(received, any(packet.raw_eeg for packet in packets)) or port_disconnected:
                detector.reset()
                alarming = False
                poor_signal = 255
                quality_received = None
                segment += 1
                publish({"timestamp_ms": time.time_ns() // 1_000_000,
                         "state": "MAT_TIN_HIEU", "alarm": 0, "raw_segment": segment})
            if serial_error is not None:
                raise OSError(f"Serial read failed on {port}: {serial_error}") from serial_error
            if watchdog.stalled:
                publish_status(received)
                continue
            for packet in packets:
                if packet.poor_signal is not None:
                    poor_signal = packet.poor_signal
                    quality_received = received
                for sample in packet.raw_eeg:
                    quality = poor_signal if quality_received is not None and received - quality_received <= 3. else 255
                    if recording:
                        recording.write(sample, quality, time.time_ns() // 1_000_000, segment)
                    raw_count += 1
                    decision = detector.add_sample(sample, quality)
                    if decision is None:
                        continue
                    alarm = bool(decision["alarm"])
                    if alarm and not alarming:
                        episodes += 1
                    decision.update(sample_index=raw_count - 1, raw_segment=segment,
                                    episode_id=episodes if alarm else None)
                    sound_enabled = sound.is_set() if isinstance(sound, Event) else sound
                    if alarm and not alarming and sound_enabled:
                        decision["audio_requested_ms"] = time.time_ns() // 1_000_000
                        Thread(target=_beep, daemon=True).start()
                    alarming = alarm
                    publish(decision)
            publish_status(received)


def _print_decision(decision):
    if decision["state"] == "MAT_TIN_HIEU":
        print("state=MAT_TIN_HIEU", flush=True)
    else:
        print(f"risk={decision.get('risk', '')}  state={decision['state']}  "
              f"latency={decision['latency_ms']} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live N16 drowsiness inference")
    parser.add_argument("--port", required=True, help="Bluetooth COM port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--model", type=Path, default=Path("output/model/n16_logistic_model.json"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--window-s", type=float, default=4.0, help="Use the window length from model training")
    parser.add_argument("--step-s", type=float, default=1.0, help="Seconds between predictions")
    parser.add_argument("--consecutive", type=int, default=2, help="Risk predictions required to alert")
    parser.add_argument("--release-consecutive", type=int, default=1, help="Low-risk predictions required to clear the alert")
    parser.add_argument("--support-threshold", type=float)
    parser.add_argument("--support-consecutive", type=int, default=1)
    parser.add_argument("--screen-raw", action="store_true")
    parser.add_argument("--duration-min", type=float, default=30.0)
    parser.add_argument("--log-output", type=Path, default=Path("output/live/live_log.csv"))
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    model = load_model(args.model)
    detector = LiveDetector(model, window_s=args.window_s, step_s=args.step_s,
                            threshold=args.threshold, consecutive=args.consecutive,
                            release_consecutive=args.release_consecutive,
                            support_threshold=args.support_threshold,
                            support_consecutive=args.support_consecutive,
                            screen_raw=args.screen_raw)
    try:
        run_live(args.port, detector, args.log_output, baud=args.baud, duration_min=args.duration_min,
                 sound=not args.no_sound, on_decision=_print_decision)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
