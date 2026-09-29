from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_events
from n16_eeg.featurize import featurize_session
from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel, evaluate_loso, fit_logistic_model
from n16_eeg.signal_features import bandpower_features
from n16_eeg.simulate import generate_session
from n16_eeg.thinkgear import ThinkGearParser
from n16_eeg.train import select_feature_columns
from n16_eeg.workflow import build_demo_workflow


def _packet(payload: bytes) -> bytes:
    checksum = (~(sum(payload) & 0xFF)) & 0xFF
    return b"\xAA\xAA" + bytes([len(payload)]) + payload + bytes([checksum])


def test_thinkgear_raw_and_metadata() -> None:
    payload = bytes([0x02, 0x00, 0x04, 55, 0x80, 0x02, 0xFF, 0x9C])
    packets = ThinkGearParser().feed(_packet(payload))
    assert len(packets) == 1
    assert packets[0].poor_signal == 0
    assert packets[0].attention == 55
    assert packets[0].raw_eeg == (-100,)


def test_thinkgear_rejects_bad_checksum() -> None:
    assert ThinkGearParser().feed(b"\xAA\xAA\x02\x80\x00\x00") == []


def test_alpha_band_dominates_for_10_hz_tone() -> None:
    fs = 512
    time = np.arange(fs * 4) / fs
    features = bandpower_features(np.sin(2 * np.pi * 10 * time), fs)
    assert features["rel_alpha"] > 0.9
    assert features["rel_alpha"] > features["rel_theta"]


def test_detects_synthetic_blinks() -> None:
    fs = 512
    samples = np.random.default_rng(16).normal(0, 1, fs * 6)
    for center_s in (1.5, 3.5, 5.0):
        center = int(center_s * fs)
        width = int(0.12 * fs)
        samples[center - width : center + width] += 35 * np.hanning(width * 2)
    events = detect_blinks(samples, fs=fs, threshold_mad=5.0)
    assert len(events) == 3


def test_simulator_contains_both_states() -> None:
    frame = generate_session(alert_seconds=5, drowsy_seconds=5, fs=128)
    assert set(frame["kss"]) == {3, 8}
    assert frame["is_synthetic"].all()
    assert "STATE_DROWSY" in set(frame["event"])


def test_blink_event_matching() -> None:
    metrics = match_events([1.02, 2.95, 8.0], [1.0, 3.0, 5.0], tolerance_s=0.10)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert abs(float(metrics["f1"]) - 2 / 3) < 1e-9


def _synthetic_features(subjects: int = 3) -> pd.DataFrame:
    frames = []
    for subject in range(1, subjects + 1):
        session = generate_session(
            subject_id=f"T{subject:02d}",
            alert_seconds=7,
            drowsy_seconds=7,
            seed=500 + subject,
        )
        frames.append(featurize_session(session))
    return pd.concat(frames, ignore_index=True)


def test_loso_and_model_roundtrip() -> None:
    frame = _synthetic_features()
    columns = select_feature_columns(frame)
    X = frame[columns].to_numpy(float)
    y = frame["label"].to_numpy(int)
    subjects = frame["subject_id"].to_numpy(str)
    folds = evaluate_loso(X, y, subjects, columns)
    assert len(folds) == 3
    assert all(fold.balanced_accuracy >= 0.70 for fold in folds)

    model = fit_logistic_model(X, y, columns)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        model.save(path)
        restored = LogisticModel.load(path)
    assert np.allclose(model.predict_proba(X[:3]), restored.predict_proba(X[:3]))


def test_full_software_workflow() -> None:
    with TemporaryDirectory() as directory:
        summary = build_demo_workflow(
            Path(directory), subjects=3, sessions_per_subject=1, seconds_per_state=7
        )
        assert Path(str(summary["model_path"])).exists()
        assert Path(str(summary["metrics_path"])).exists()
        assert int(summary["replay_windows"]) > 0
        assert int(summary["replay_alarm_windows"]) > 0


def test_live_detector_accepts_streamed_samples() -> None:
    frame = _synthetic_features()
    columns = select_feature_columns(frame)
    model = fit_logistic_model(
        frame[columns].to_numpy(float), frame["label"].to_numpy(int), columns
    )
    session = generate_session(alert_seconds=7, drowsy_seconds=7, seed=902)
    detector = LiveDetector(model)
    decisions = []
    for row in session.itertuples(index=False):
        decision = detector.add_sample(row.raw_eeg, row.poor_signal)
        if decision is not None:
            decisions.append(decision)
    assert decisions
    assert any(int(decision["alarm"]) for decision in decisions)
