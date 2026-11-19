"""Evaluate and export the class-balanced RBF model for the N16 Windows app."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from compare_pilot_ml_models import fit_model
from evaluate_python_alert_latency import FEATURES, onset_test
from update_openvibe_model import sha256

from n16_eeg.model import load_model
from n16_eeg.simulate import generate_session

THRESHOLD_GRID = (.55, .56, .57, .58, .59, .60)


def fit(frame):
    """Give the two classes equal total weight when fitting the SVM."""
    counts = frame.groupby("label").label.transform("size")
    weights = len(frame) / (2 * counts.to_numpy())
    return fit_model(frame, "rbf_combined", weights)


def predict(estimator, frame):
    margins = np.clip(estimator.decision_function(frame[FEATURES]), -30, 30)
    return 1 / (1 + np.exp(-margins))


def evaluate_predictions(frame, predictions):
    correct = frame[["subject_id", "label"]].assign(correct=predictions == frame.label)
    by_person = correct.groupby(["subject_id", "label"]).correct.mean()
    by_class = by_person.groupby("label").mean()
    return {
        "balanced_accuracy": float(by_class.mean()),
        "recall_awake": float(by_class.loc[0]),
        "recall_drowsy": float(by_class.loc[1]),
        "recall_by_person": by_person.unstack().astype(object)
        .where(by_person.unstack().notna(), None).to_dict(orient="index"),
    }


def evaluate(frame, threshold):
    """Measure fixed-threshold LOSO and calibrate the threshold in nested LOSO."""
    people = sorted(frame.subject_id.unique())
    inner_scores = {}
    for pair in combinations(people, 2):
        test = frame.subject_id.isin(pair)
        estimator = fit(frame.loc[~test])
        inner_scores[pair] = pd.Series(predict(estimator, frame.loc[test]), index=frame.index[test])

    scores = pd.Series(index=frame.index, dtype=float)
    nested_thresholds = pd.Series(index=frame.index, dtype=float)
    for subject in people:
        test = frame.subject_id.eq(subject)
        training = frame.loc[~test]
        scores.loc[test] = predict(fit(training), frame.loc[test])
        inner = pd.concat([values.loc[~test.loc[values.index]] for pair, values
                           in inner_scores.items() if subject in pair]).reindex(training.index)
        nested_thresholds.loc[test] = max(
            THRESHOLD_GRID,
            key=lambda cut: (evaluate_predictions(training, inner >= cut)["balanced_accuracy"],
                             -abs(cut - threshold)),
        )
    return scores, nested_thresholds


def save_model(estimator, threshold, path):
    _, scaler, classifier = (step for _, step in estimator.steps)
    payload = {
        "model_type": "numpy_rbf_svm",
        "format_version": 1,
        "feature_names": list(estimator.feature_names_in_),
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "support_vectors": classifier.support_vectors_.tolist(),
        "dual_coefficients": classifier.dual_coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "gamma": float(classifier._gamma),
        "threshold": threshold,
    }
    path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")


def check_timing(model, fixture_path, schedule):
    fixtures = [pd.read_csv(fixture_path)]
    fixtures.extend(generate_session(alert_seconds=12 + phase, drowsy_seconds=12, seed=seed)
                    for seed in (51, 71, 103, 127) for phase in (.05, .23, .47, .71, .93))
    cases = [onset_test(model, schedule, fixture) for fixture in fixtures]
    poor = fixtures[-1].copy()
    poor.loc[poor.timestamp_ms.between(8000, 18000), "poor_signal"] = 200
    return {
        "cases": cases,
        "within_3s": sum(row["onset_to_alarm_ms"] is not None
                         and row["onset_to_alarm_ms"] < 3000 for row in cases),
        "early_alarms": sum(row["pre_onset_alarm"] for row in cases),
        "poor_signal_case": onset_test(model, schedule, poor),
    }


def run(source, previous, fixture, output, threshold):
    frame = pd.read_csv(source / "predictions_LOSO_PROVISIONAL.csv")
    release = json.loads((source / "release.json").read_text(encoding="utf-8-sig"))
    schedule = tuple(release[key] for key in ("window_s", "step_s", "consecutive"))
    profile = release["variants"]["LowRepeat"]
    scores, tuned = evaluate(frame, threshold)
    frame = frame.assign(upgraded_score=scores, upgraded_prediction=(scores >= threshold).astype(int),
                         inner_threshold=tuned, nested_prediction=(scores >= tuned).astype(int))
    summary = {name: evaluate_predictions(frame, frame[f"{name}_prediction"])
               for name in ("previous_data", "updated_data", "upgraded", "nested")}
    cuts = frame.groupby("subject_id").inner_threshold.first().to_dict()

    output.mkdir(parents=True, exist_ok=False)
    estimator = fit(frame)
    path = output / "python_openvibe_rbf_3s_PROVISIONAL.json"
    save_model(estimator, threshold, path)
    model = load_model(path)
    native = [model.predict_feature_dict(row) for row in frame[FEATURES].to_dict(orient="records")]
    np.testing.assert_allclose(native, predict(estimator, frame), rtol=0, atol=1e-9)
    old_release = json.loads((previous / "release.json").read_text(encoding="utf-8-sig"))
    old_model = load_model(previous / old_release["variants"]["LowRepeat"]["file"])
    updated_model = load_model(source / profile["file"])
    timings = {name: check_timing(candidate, fixture, schedule)
               for name, candidate in (("previous_data", old_model), ("updated_data", updated_model),
                                       ("upgraded", model))}
    release.update(release=output.name, status="PROVISIONAL_OPENVIBE_RBF",
                   variants={"LowRepeat": dict(profile, file=path.name, sha256=sha256(path),
                                               threshold=threshold)})
    report = {
        "scope": "Exploratory model and threshold selection on the same five-person cohort. "
                 "Nested LOSO selects only the threshold inside training folds. "
                 "Session labels are provisional. Synthetic timing measures replay only.",
        "windows": len(frame),
        "source_predictions_sha256": sha256(source / "predictions_LOSO_PROVISIONAL.csv"),
        "model": "Class-balanced binary RBF SVM with log1p and robust scaling",
        "score": "Sigmoid of the decision margin, not a calibrated probability",
        "threshold": threshold,
        "inner_selected_thresholds": cuts,
        "class_metrics": summary,
        "synthetic_replay": timings,
    }
    for name, content in (("metrics_PROVISIONAL.json", report), ("release.json", release)):
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "sessions.json").write_bytes((source / "sessions.json").read_bytes())
    frame.to_csv(output / "predictions_LOSO_PROVISIONAL.csv", index=False)
    print({name: round(value["balanced_accuracy"], 4) for name, value in summary.items()})
    print({name: (value["within_3s"], value["early_alarms"]) for name, value in timings.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "previous", "fixture", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threshold", type=float, default=.57)
    run(**vars(parser.parse_args()))
