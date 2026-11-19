"""Refit the Windows OpenViBE model with confirmed KSS intervals and nested LOSO."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVC

from evaluate_python_alert_latency import FEATURES
from update_openvibe_model import additional_features, sha256
from upgrade_openvibe_model import evaluate_predictions, save_model
from n16_eeg.model import load_model


FEATURE_SETS = {
    "combined": FEATURES,
    "spectral": FEATURES[:-3],
    "ratios": ["theta_alpha", "theta_beta", "alpha_beta", "spectral_entropy"],
}
CANDIDATES = [(name, cost, gamma) for name in FEATURE_SETS
              for cost in (0.25, 0.5, 2.0) for gamma in ("scale", 0.05)]
CUTS = (0.44, 0.48, 0.50, 0.52, 0.55, 0.57, 0.60, 0.63)
BASELINE = ("combined", 0.5, "scale")


def fit(frame, candidate):
    feature_set, cost, gamma = candidate
    features = FEATURE_SETS[feature_set]
    counts = frame.groupby("label").label.transform("size")
    return make_pipeline(FunctionTransformer(np.log1p), RobustScaler(), SVC(C=cost, gamma=gamma)).fit(
        frame[features], frame.label, svc__sample_weight=len(frame) / (2 * counts.to_numpy())
    )


def score(model, frame):
    margins = np.clip(model.decision_function(frame[list(model.feature_names_in_)]), -30, 30)
    return 1 / (1 + np.exp(-margins))


def select(frame, scores, old_threshold):
    def rank(candidate, cut):
        metrics = evaluate_predictions(frame, scores[candidate] >= cut)
        return (metrics["balanced_accuracy"], min(metrics["recall_awake"], metrics["recall_drowsy"]),
                -abs(old_threshold - cut))

    return max(((candidate, cut) for candidate in CANDIDATES for cut in CUTS),
               key=lambda choice: rank(*choice))


def compare(frame, previous, old_threshold):
    people = sorted(frame.subject_id.unique())
    outer = {candidate: pd.Series(index=frame.index, dtype=float) for candidate in CANDIDATES}
    nested = pd.Series(False, index=frame.index)
    baseline = pd.Series(False, index=frame.index)
    folds = []
    inner_scores = {}
    for pair in combinations(people, 2):
        test = frame.subject_id.isin(pair)
        inner_scores[pair] = {candidate: pd.Series(score(fit(frame.loc[~test], candidate), frame.loc[test]),
                                                   index=frame.index[test]) for candidate in CANDIDATES}

    for person in people:
        test = frame.subject_id.eq(person)
        train, held_out = frame.loc[~test], frame.loc[test]
        inner = {candidate: pd.concat([values[candidate].drop(index=held_out.index, errors="ignore")
                                       for pair, values in inner_scores.items() if person in pair]).reindex(train.index)
                 for candidate in CANDIDATES}
        chosen, threshold = select(train, inner, old_threshold)
        for candidate in CANDIDATES:
            outer[candidate].loc[test] = score(fit(train, candidate), held_out)
        baseline.loc[test] = score(fit(previous.loc[previous.subject_id.ne(person)], BASELINE), held_out) >= old_threshold
        nested.loc[test] = outer[chosen].loc[test] >= threshold
        folds.append(dict(held_out=person, candidate=chosen, threshold=threshold))
        print(f"Held out {person}: {chosen}, threshold {threshold:.2f}", flush=True)

    chosen, threshold = select(frame, outer, old_threshold)
    return chosen, threshold, dict(
        previous_labels_fixed_config_loso=evaluate_predictions(frame, baseline),
        confirmed_intervals_fixed_config_loso=evaluate_predictions(frame, outer[BASELINE] >= old_threshold),
        nested_loso=evaluate_predictions(frame, nested),
        selection_loso=evaluate_predictions(frame, outer[chosen] >= threshold),
        nested_folds=folds,
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run(repo, preparation, original, previous_release, output):
    report = read_json(preparation)
    previous = pd.read_csv(original / "predictions_LOSO_PROVISIONAL.csv")
    release = read_json(previous_release / "release.json")
    profile = release["variants"]["LowRepeat"]
    old_model_path = previous_release / profile["file"]
    if sha256(old_model_path) != profile["sha256"]:
        raise ValueError("Previous model does not match its release hash")
    old_threshold = profile["threshold"]
    nominal = previous.loc[previous.label_basis.eq("nominal_kss")]
    raw_dir = repo / "raw" / "Raw data"
    old_report = read_json(original / "metrics_PROVISIONAL.json")
    # Exclude hashes of every preserved recording and verify the older MAIN_02 still matches.
    pilot_report = read_json(repo / "output/two_subjects_20261105_v1/experiment_PROVISIONAL.json")
    prior_hashes = {item["source_sha256"] for item in pilot_report["sessions"]}
    for item in old_report["new_recordings"]:
        if item["label_basis"] == "nominal_kss":
            if sha256(raw_dir / item["file"]) != item["sha256"]:
                raise ValueError(f"Previous labeled recording changed: {item['file']}")
            prior_hashes.add(item["sha256"])
    entries = [dict(file=Path(item["raw"]).relative_to("raw/Raw data").as_posix(),
                    annotations=item["annotations"], annotations_sha256=item["annotations_sha256"],
                    recording_sha256=item["source_sha256"], note=item["note"], kss=item["kss"])
               for item in report["sessions"]]
    parts, audit = zip(*additional_features(raw_dir, entries, prior_hashes))
    frame = pd.concat([nominal, *parts], ignore_index=True)
    if frame.duplicated(["subject_id", "session_id", "start_sample"]).any():
        raise ValueError("Duplicate windows in training input")
    old_model = load_model(old_model_path)
    previous_check = previous.iloc[::17]
    old_scores = [old_model.predict_feature_dict(row) for row in previous_check[FEATURES].to_dict("records")]
    np.testing.assert_allclose(old_scores, score(fit(previous, BASELINE), previous_check), rtol=0, atol=1e-8)
    candidate, cutoff, results = compare(frame, previous, old_threshold)

    output.mkdir(parents=True, exist_ok=False)
    estimator = fit(frame, candidate)
    model_path = output / "python_openvibe_rbf_3s.json"
    save_model(estimator, cutoff, model_path)
    model = load_model(model_path)
    native = [model.predict_feature_dict(row) for row in frame[FEATURES].to_dict("records")]
    np.testing.assert_allclose(native, score(estimator, frame), rtol=0, atol=1e-9)
    # The low-margin confirmation band keeps its existing width relative to the main cutoff.
    if "support_threshold" in profile:
        profile["support_threshold"] = round(cutoff - (old_threshold - profile["support_threshold"]), 6)
    profile.update(file=model_path.name, sha256=sha256(model_path), threshold=cutoff)
    release.update(release=output.name, status="EXPLORATORY_KSS_INTERVALS")
    results.update(candidate=candidate, threshold=cutoff, window_counts={
        "total": len(frame), "new_kss_intervals": sum(len(part) for part in parts), "older_nominal_kss": len(nominal),
    }, search=dict(candidates=CANDIDATES, thresholds=CUTS),
        preparation_sha256=sha256(preparation), session_audit=list(audit),
        scope="Five people; previously captured sessions remain in the data. Nested LOSO only; no independent validation. "
              "Balanced accuracy averages per-class, per-person recall. Window classifications precede alarm confirmation.")
    for name, payload in (("release.json", release), ("metrics.json", results)):
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    frame[["subject_id", "session_id", "start_sample", "label", "label_basis", *FEATURES]].to_csv(
        output / "training_features_3s.csv", index=False
    )
    print(json.dumps({"candidate": candidate, "threshold": cutoff, "metrics": {
        key: results[key]["balanced_accuracy"] for key in (
            "previous_labels_fixed_config_loso", "confirmed_intervals_fixed_config_loso", "nested_loso", "selection_loso"
        )}, "windows": results["window_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "preparation", "original", "previous_release", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
