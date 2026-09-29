"""One-command software-only workflow for the N16 MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .demo import replay
from .featurize import featurize_session
from .model import LogisticModel
from .simulate import generate_session
from .train import train_from_frame


def build_demo_workflow(
    root: Path,
    subjects: int = 6,
    sessions_per_subject: int = 2,
    seconds_per_state: float = 12.0,
) -> dict[str, object]:
    if subjects < 3:
        raise ValueError("At least 3 synthetic subjects are required for LOSO")
    if sessions_per_subject < 1:
        raise ValueError("sessions_per_subject must be positive")

    data_dir = root / "data" / "synthetic"
    model_dir = root / "output" / "model"
    demo_dir = root / "output" / "demo"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    feature_frames: list[pd.DataFrame] = []
    replay_source: pd.DataFrame | None = None
    for subject_index in range(1, subjects + 1):
        subject_id = f"SIM{subject_index:02d}"
        for session_index in range(1, sessions_per_subject + 1):
            session_id = f"session_{session_index:02d}"
            frame = generate_session(
                subject_id=subject_id,
                session_id=session_id,
                alert_seconds=seconds_per_state,
                drowsy_seconds=seconds_per_state,
                seed=subject_index * 100 + session_index,
            )
            feature_frames.append(featurize_session(frame))
            if replay_source is None:
                replay_source = frame

    features = pd.concat(feature_frames, ignore_index=True)
    feature_path = data_dir / "cohort_features.csv"
    features.to_csv(feature_path, index=False)

    model_path = model_dir / "n16_logistic_model.json"
    metrics_path = model_dir / "loso_metrics.json"
    metrics = train_from_frame(features, model_path, metrics_path)

    assert replay_source is not None
    replay_path = data_dir / "demo_session.csv"
    replay_source.to_csv(replay_path, index=False)
    replay_log_path = demo_dir / "replay_model_log.csv"
    model = LogisticModel.load(model_path)
    replay_result = replay(
        replay_source,
        output_log=replay_log_path,
        speed=0,
        threshold=model.threshold,
        sound=False,
        model=model,
        verbose=False,
    )

    summary: dict[str, object] = {
        "warning": "All generated data and metrics are synthetic software checks only.",
        "subjects": subjects,
        "sessions_per_subject": sessions_per_subject,
        "feature_windows": int(len(features)),
        "mean_macro_f1": metrics["mean_macro_f1"],
        "mean_balanced_accuracy": metrics["mean_balanced_accuracy"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "features_path": str(feature_path),
        "demo_session_path": str(replay_path),
        "replay_log_path": str(replay_log_path),
        "replay_windows": int(len(replay_result)),
        "replay_alarm_windows": int(replay_result["alarm"].sum()),
        "latency_p95_ms": float(replay_result["latency_ms"].quantile(0.95)),
    }
    summary_path = demo_dir / "workflow_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify the full software-only N16 MVP")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--seconds-per-state", type=float, default=12.0)
    args = parser.parse_args()
    summary = build_demo_workflow(
        args.root.resolve(),
        subjects=args.subjects,
        sessions_per_subject=args.sessions,
        seconds_per_state=args.seconds_per_state,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WARNING: synthetic values must not be used as experimental results.")


if __name__ == "__main__":
    main()
