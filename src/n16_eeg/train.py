"""Train, evaluate and save the leakage-safe NumPy logistic baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .model import evaluate_loso, fit_logistic_model


NON_FEATURES = {
    "subject_id",
    "session_id",
    "label",
    "window_start_ms",
    "window_end_ms",
    "source_is_synthetic",
}


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in NON_FEATURES:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().all():
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns found")
    return columns


def train_from_frame(
    frame: pd.DataFrame,
    model_output: Path,
    metrics_output: Path,
) -> dict[str, object]:
    missing = {"subject_id", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    feature_columns = select_feature_columns(frame)
    X = frame[feature_columns].to_numpy(float)
    y = frame["label"].to_numpy(int)
    subjects = frame["subject_id"].to_numpy(str)
    folds = evaluate_loso(X, y, subjects, feature_columns)
    final_model = fit_logistic_model(X, y, feature_columns)
    final_model.save(model_output)

    synthetic = bool(
        frame.get("source_is_synthetic", pd.Series(False, index=frame.index))
        .astype(bool)
        .any()
    )
    auc_values = [fold.roc_auc for fold in folds if fold.roc_auc is not None]
    metrics: dict[str, object] = {
        "warning": (
            "Synthetic metrics are software checks only and must not be reported as experimental results."
            if synthetic
            else None
        ),
        "model": final_model.model_type,
        "features": feature_columns,
        "subjects": int(len(np.unique(subjects))),
        "windows": int(len(frame)),
        "mean_macro_f1": float(np.mean([fold.macro_f1 for fold in folds])),
        "mean_balanced_accuracy": float(
            np.mean([fold.balanced_accuracy for fold in folds])
        ),
        "mean_roc_auc": float(np.mean(auc_values)) if auc_values else None,
        "folds": [asdict(fold) for fold in folds],
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO evaluation and final N16 model training")
    parser.add_argument("features", type=Path)
    parser.add_argument(
        "--model-output", type=Path, default=Path("output/model/n16_logistic_model.json")
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("output/model/loso_metrics.json")
    )
    args = parser.parse_args()
    metrics = train_from_frame(
        pd.read_csv(args.features), args.model_output, args.metrics_output
    )
    if metrics["warning"]:
        print("WARNING:", metrics["warning"])
    for fold in metrics["folds"]:
        print(
            f"{fold['subject']}: macro_f1={fold['macro_f1']:.3f} "
            f"balanced_accuracy={fold['balanced_accuracy']:.3f} "
            f"roc_auc={fold['roc_auc']:.3f} samples={fold['samples']}"
        )
    print("mean_macro_f1", f"{metrics['mean_macro_f1']:.3f}")
    print("mean_balanced_accuracy", f"{metrics['mean_balanced_accuracy']:.3f}")
    print("model", args.model_output)
    print("metrics", args.metrics_output)


if __name__ == "__main__":
    main()
