"""Explicit-feature, subject-held-out scikit-learn baselines (no test tuning)."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_FEATURES = {
    "subject_id", "session_id", "label", "perclos", "kss", "sample_index",
    "timestamp_ms", "window_start_ms", "window_end_ms", "source_is_synthetic",
    "dataset_id", "signal_unit", "feature_row", "source_file", "fs_hz",
}


def make_estimator(name: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    classifiers = {
        "logistic": lambda: LogisticRegression(class_weight="balanced", max_iter=2000,
                                                solver="lbfgs", random_state=16),
        "svm": lambda: SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        "random-forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight="balanced", random_state=16, n_jobs=1),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown classifier: {name}")
    return make_pipeline(StandardScaler(), classifiers[name]())


def validate_frame(frame: pd.DataFrame, features: list[str], dataset: str):
    if not features or len(features) != len(set(features)):
        raise ValueError("Provide unique, explicit feature names")
    if set(features) & FORBIDDEN_FEATURES:
        raise ValueError("Labels, IDs, timestamps and metadata cannot be features")
    missing = {"subject_id", "label", *features} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame["subject_id"].isna().any():
        raise ValueError("Empty data or missing subject_id")
    groups = frame["subject_id"].astype(str).str.strip().to_numpy()
    if np.any(groups == "") or len(np.unique(groups)) < 2:
        raise ValueError("LOSO requires at least two identified people")
    # Validate BEFORE integer casting: label 0.5 must never become class 0.
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(float)
    if not np.isfinite(labels).all() or not set(labels) <= {0.0, 1.0}:
        raise ValueError("Expected binary labels exactly 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("Final training requires both classes")
    X = frame[features].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN/Inf")
    if "dataset_id" in frame:
        if frame["dataset_id"].isna().any() or set(frame["dataset_id"].astype(str)) != {dataset}:
            raise ValueError("Dataset provenance mismatch; do not mix datasets")
    if "signal_unit" in frame:
        units = frame["signal_unit"]
        if units.isna().any() or units.nunique() != 1 or not set(units) <= {"count", "uV"}:
            raise ValueError("Require one verified signal unit; do not mix count and microvolt")
    synthetic = False
    if "source_is_synthetic" in frame:
        flags = frame["source_is_synthetic"].astype(str).str.lower()
        if not flags.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Invalid synthetic flag")
        synthetic = bool(flags.isin(["true", "1"]).any())
    return X, labels.astype(int), groups, synthetic


def evaluate_frame(frame: pd.DataFrame, features: list[str], dataset: str,
                   classifier: str = "logistic"):
    from sklearn.base import clone
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    import sklearn

    X, y, groups, synthetic = validate_frame(frame, features, dataset)
    template = make_estimator(classifier)
    folds, predictions = [], []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        fold = {
            "subject": str(groups[test][0]), "samples": int(len(test)),
            "train_subjects": sorted(set(groups[train])),
            "test_class_counts": np.bincount(y[test], minlength=2).tolist(),
        }
        if len(np.unique(y[train])) < 2:
            fold.update(status="skipped", reason="Training people contain only one class")
            folds.append(fold)
            continue
        estimator = clone(template)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(X[train], y[train])
        pred = estimator.predict(X[test])
        if hasattr(estimator, "predict_proba"):
            score = estimator.predict_proba(X[test])[:, 1]
        else:
            score = estimator.decision_function(X[test])
        cm = confusion_matrix(y[test], pred, labels=[0, 1])
        both = len(np.unique(y[test])) == 2
        recalls = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(2)]
        fold.update(
            status="evaluated", confusion_matrix=cm.tolist(),
            accuracy=float(accuracy_score(y[test], pred)),
            macro_f1=float(f1_score(y[test], pred, labels=[0, 1], average="macro", zero_division=0)),
            balanced_accuracy=float(np.mean(recalls)) if both else None,
            roc_auc=float(roc_auc_score(y[test], score)) if both else None,
            class_recalls=recalls,
            scaler_mean=estimator.steps[0][1].mean_.tolist(),
            scaler_scale=estimator.steps[0][1].scale_.tolist(),
            note=None if both else "One-class test retained; AUC/balanced accuracy undefined here",
        )
        folds.append(fold)
        for i, prediction, value in zip(test, pred, score):
            predictions.append({"row_index": int(i), "subject_id": str(groups[i]),
                                "true_label": int(y[i]), "prediction": int(prediction),
                                "score": float(value)})
    valid = [f for f in folds if f["status"] == "evaluated"]
    if not valid:
        raise ValueError("No evaluable LOSO fold; collect both classes in training people")
    summary = {}
    for key in ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc"]:
        values = [f[key] for f in valid if f[key] is not None]
        summary[key] = {"mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values)) if values else None,
                        "people": len(values)}
    final = clone(template)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        final.fit(X, y)
    report = {
        "dataset": dataset, "classifier": classifier, "sklearn_version": sklearn.__version__,
        "features": features, "subjects": len(np.unique(groups)), "windows": len(y),
        "warning": "SYNTHETIC: software tests only" if synthetic else None,
        "split": "LeaveOneGroupOut(groups=subject_id); scaler fitted inside each fold",
        "tuning": "Fixed defaults, no model/threshold selection using outer test scores",
        "score_kind": "decision_function" if classifier == "svm" else "predict_proba_class_1",
        "macro_f1_definition": "Fixed labels [0,1], zero_division=0; one-class tests retained",
        "folds": folds, "summary_by_person": summary,
        "export_scope": "Offline only; does not create a C++ runtime model",
        "signal_unit": str(frame["signal_unit"].iloc[0]) if "signal_unit" in frame else None,
    }
    return final, report, pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--dataset", choices=["neurosky", "openvibe", "seed-vig"], required=True)
    parser.add_argument("--classifier", choices=["logistic", "svm", "random-forest"], default="logistic")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = ["pipeline.joblib", "metrics.json", "oof_predictions.csv"]
    if any((args.output_dir / name).exists() for name in names):
        parser.error("Output exists; use a new run directory to preserve previous results")
    frame = pd.read_csv(args.features_csv, dtype={"subject_id": str})
    final, report, predictions = evaluate_frame(frame, args.features, args.dataset, args.classifier)
    import joblib
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.output_dir / names[0])
    (args.output_dir / names[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                     allow_nan=False), encoding="utf-8")
    predictions.to_csv(args.output_dir / names[2], index=False)
    print(json.dumps(report["summary_by_person"], indent=2))
    print("Offline pipeline saved. Do not load untrusted joblib files or use it as C++ JSON.")


if __name__ == "__main__":
    main()
