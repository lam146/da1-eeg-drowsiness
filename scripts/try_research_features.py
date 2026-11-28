"""Evaluate literature-inspired features with fixed windows and held-out people."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from develop_pilot_model import FEATURES, session_weights
from evaluate_python_alert_latency import feature_vector
from improve_pilot_ml import KEY
from n16_eeg.research_features import research_features


def fit(frame, columns, family):
    classifier = (LogisticRegression(C=.1, max_iter=3000) if family == "logistic"
                  else SVC(C=.5, gamma="scale"))
    model = make_pipeline(RobustScaler(), classifier)
    model.fit(frame[columns], frame.label,
              **{f"{model.steps[-1][0]}__sample_weight": session_weights(frame)})
    return model


def metrics(truth, predicted):
    cm = confusion_matrix(truth, predicted, labels=[0, 1])
    recall = np.diag(cm) / cm.sum(axis=1)
    return dict(balanced_accuracy=float(recall.mean()), awake_recall=float(recall[0]),
                drowsy_recall=float(recall[1]), macro_f1=float(f1_score(truth, predicted, average="macro")),
                confusion_matrix=cm.tolist())


def run(pilot, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_csv(pilot / "all_subjects_features_PROVISIONAL.csv").sort_values(KEY).reset_index(drop=True)
    if set(metadata.subject_id) != {"S01", "S02"} or metadata.duplicated(KEY).any():
        raise ValueError("Expected original unique S01/S02 windows")
    manifest = json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    sources = {(row["subject_id"], row["session_id"]): row for row in manifest["sessions"]}
    records, costs = [], []
    for key, group in metadata.groupby(KEY[:2], sort=False):
        source = sources[key]
        path = Path(source["source"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["source_sha256"]:
            raise ValueError(f"Raw changed: {path}")
        raw = pd.read_csv(path, usecols=["Electrode"]).Electrode.to_numpy(float)
        for index, row in group.iterrows():
            end = round(row.window_end_ms * 512 / 1000)
            samples = raw[end - 1536:end]
            started = time.perf_counter()
            extra = research_features(samples)
            costs.append((time.perf_counter() - started) * 1000)
            records.append(dict(index=index, **dict(zip(FEATURES, feature_vector(samples))), **extra))
        print(*key, len(group), flush=True)
    feature_table = pd.DataFrame(records).set_index("index").sort_index()
    if not np.isfinite(feature_table.to_numpy()).all():
        raise ValueError("Nonfinite research features")
    frame = pd.concat([metadata[KEY + ["window_end_ms", "label", "source_file"]], feature_table], axis=1)
    frame["date"] = frame.source_file.str.extract(r"_(\d{8})_")[0]
    frame.to_csv(output / "features_PROVISIONAL.csv", index=False)
    ar = [name for name in feature_table if name.startswith("ar_")]
    entropy = [name for name in feature_table if name.startswith("sample_entropy_")]
    branches = {"baseline": FEATURES, "plus_ar": FEATURES + ar,
                "plus_entropy": FEATURES + entropy, "plus_both": FEATURES + ar + entropy,
                "ar_entropy": ar + entropy}
    configurations = {f"{branch}_{family}": (columns, family) for branch, columns in branches.items()
                      for family in ("logistic", "svm")}
    report = dict(status="DEVELOPMENT_ONLY_PROVISIONAL_KSS", windows=len(frame),
                  methods="AR order 8, Yule-Walker with 1e-6 diagonal regularization; sample entropy m=2, r=0.2, scales 1/2/4, common embeddings and +1 count smoothing; decimation to 128 Hz after FFT band limit. Inspired experiment, independently implemented parameters.",
                  features=configurations,
                  reference="https://doi.org/10.3390/s18124477",
                  feature_latency_median_ms=float(np.median(costs)),
                  feature_latency_p95_ms=float(np.quantile(costs, .95)),
                  selection={}, folds=[])
    (output / "plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = []
    for person in sorted(frame.subject_id.unique()):
        train = frame.loc[frame.subject_id.ne(person)]
        test = frame.loc[frame.subject_id.eq(person)]
        groups = train.subject_id + "/" + train.date
        splits = list(LeaveOneGroupOut().split(train, groups=groups))
        ranking = {}
        for name, (columns, family) in configurations.items():
            inner_pred = np.empty(len(train), dtype=int)
            for left, right in splits:
                inner = train.iloc[left]
                if inner.label.nunique() != 2:
                    raise ValueError("Training-day split missing a class")
                inner_pred[right] = fit(inner, columns, family).predict(train.iloc[right][columns])
            ranking[name] = float(balanced_accuracy_score(train.label, inner_pred))
        selected = max(ranking, key=ranking.get)
        report["selection"][person] = dict(chosen=selected, training_day_balanced_accuracy=ranking)
        for name, (columns, family) in configurations.items():
            model = fit(train, columns, family)
            predicted = model.predict(test[columns])
            names = [name, "selected_on_training_days"] if name == selected else [name]
            for label in names:
                report["folds"].append(dict(model=label, person=person, **metrics(test.label, predicted)))
                predictions.append(test[KEY + ["label"]].assign(model=label, predicted=predicted))
        print("held out", person, "training-day selection", selected, flush=True)
    folds = pd.DataFrame(report["folds"])
    columns = ["balanced_accuracy", "awake_recall", "drowsy_recall", "macro_f1"]
    summary = folds.groupby("model")[columns].mean().sort_values("balanced_accuracy", ascending=False)
    summary.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    folds.to_csv(output / "folds.csv", index=False)
    pd.concat(predictions).to_csv(output / "predictions_PROVISIONAL.csv", index=False)
    report["caution"] = "Repeated development on S01/S02. KSS alignment provisional. Independent S03 labels required for confirmation. Runtime/audio latency and real-time warning policy not validated for research models."
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Save the post-hoc development winner for future independent evaluation.
    best = summary.index[0]
    if best != "selected_on_training_days":
        columns, family = configurations[best]
        joblib.dump(dict(estimator=fit(frame, columns, family), feature_names=columns,
                         config=best, window_s=3, status=report["status"]), output / "candidate_OFFLINE.joblib")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.pilot, args.output)
