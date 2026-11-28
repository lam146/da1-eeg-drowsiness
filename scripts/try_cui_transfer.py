"""Use public Fp1 EEG to assist N16, selecting source weight on training days."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from develop_pilot_model import session_weights
from improve_pilot_ml import KEY
from n16_eeg.research_features import research_features
from n16_eeg.signal_features import bandpower_features


SPECTRAL = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
            "theta_alpha", "theta_beta", "alpha_beta", "spectral_entropy"]
OPTIONS = {"target_only": 0., "source_10pct": .1, "source_25pct": .25,
           "source_50pct": .5, "public_only": 1.}


def features(signal):
    return bandpower_features(signal, fs=128) | research_features(signal, fs=128)


def fit_predict(source, training, testing, columns, proportion):
    source_weights = session_weights(source)
    target_weights = session_weights(training)
    if proportion == 0:
        fitting = training
        weights = target_weights
    elif proportion == 1:
        fitting = source
        weights = source_weights / source_weights.sum() * len(training)
    else:
        fitting = pd.concat([training, source], ignore_index=True)
        weights = np.r_[(1 - proportion) * target_weights,
                        proportion * source_weights / source_weights.sum() * target_weights.sum()]
    scaler = StandardScaler().fit(fitting[columns], sample_weight=weights)
    model = LogisticRegression(C=.1, max_iter=3000)
    model.fit(scaler.transform(fitting[columns]), fitting.label, sample_weight=weights)
    return model.predict_proba(scaler.transform(testing[columns]))[:, 1]


def score(labels, probability):
    predicted = probability >= .5
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    recalls = np.diag(matrix) / matrix.sum(axis=1)
    return dict(balanced_accuracy=float(recalls.mean()), awake_recall=float(recalls[0]),
                drowsy_recall=float(recalls[1]), confusion_matrix=matrix.tolist())


def run(dataset, pilot, output):
    output.mkdir(parents=True, exist_ok=False)
    content = loadmat(dataset)
    eeg = content["EEGsample"]
    labels, people = (content[key].ravel().astype(int) for key in ("substate", "subindex"))
    if eeg.shape != (len(labels), 30, 384) or len(np.unique(people)) != 11 or set(labels) != {0, 1}:
        raise ValueError(f"Unexpected public dataset shape {eeg.shape}")
    public = pd.DataFrame([features(signal) for signal in eeg[:, 0, :]])
    public["subject_id"] = [f"CUI_{person:02d}" for person in people]
    public["session_id"] = public.subject_id
    public["label"] = labels
    public.to_csv(output / "public_fp1_features.csv", index=False)
    print("Public Fp1 extracted", len(public), flush=True)

    metadata = pd.read_csv(pilot / "all_subjects_features_PROVISIONAL.csv").sort_values(KEY).reset_index(drop=True)
    manifest = json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    sessions = {(row["subject_id"], row["session_id"]): row for row in manifest["sessions"]}
    records = []
    for key, rows in metadata.groupby(KEY[:2], sort=False):
        recording = sessions[key]
        path = Path(recording["source"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != recording["source_sha256"]:
            raise ValueError("Target recording changed")
        raw = pd.read_csv(path, usecols=["Electrode"]).Electrode.to_numpy(float)
        for index, row in rows.iterrows():
            end = round(row.window_end_ms * 512 / 1000)
            # Each 3 s window is independently resampled to match source epochs.
            reduced = resample_poly(raw[end - 1536:end], 1, 4)
            records.append(dict(index=index, **features(reduced)))
        print(*key, len(rows), flush=True)
    values = pd.DataFrame(records).set_index("index").sort_index()
    target = pd.concat([metadata[KEY + ["window_end_ms", "source_file", "label"]], values], axis=1)
    target["date"] = target.source_file.str.extract(r"_(\d{8})_")[0]
    target.to_csv(output / "target_features_PROVISIONAL.csv", index=False)
    if not np.isfinite(public[values.columns].to_numpy()).all() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite features")
    branches = {"spectral": SPECTRAL, "spectral_ar_entropy": list(values.columns)}
    folds, predictions, choices, public_folds = [], [], {}, []
    for branch, columns in branches.items():
        for person in sorted(public.subject_id.unique()):
            train = public.loc[public.subject_id.ne(person)]
            test = public.loc[public.subject_id.eq(person)]
            probability = fit_predict(train, train, test, columns, 0.)
            public_folds.append(dict(branch=branch, person=person, **score(test.label, probability)))
        for person in sorted(target.subject_id.unique()):
            training = target.loc[target.subject_id.ne(person)]
            testing = target.loc[target.subject_id.eq(person)]
            groups = training.subject_id + "/" + training.date
            inner_splits = list(LeaveOneGroupOut().split(training, groups=groups))
            ranking = {}
            for name, proportion in OPTIONS.items():
                prediction = np.empty(len(training), dtype=int)
                for left, right in inner_splits:
                    inner = training.iloc[left]
                    if inner.label.nunique() != 2:
                        raise ValueError("Missing training class in inner day split")
                    probability = fit_predict(public, inner, training.iloc[right], columns, proportion)
                    prediction[right] = probability >= .5
                ranking[name] = float(balanced_accuracy_score(training.label, prediction))
            selected = max(ranking, key=ranking.get)
            choices[f"{branch}/{person}"] = dict(chosen=selected, training_day_scores=ranking)
            for name, proportion in OPTIONS.items():
                probability = fit_predict(public, training, testing, columns, proportion)
                for choice in ([name, "selected_on_training_days"] if name == selected else [name]):
                    folds.append(dict(branch=branch, model=choice, person=person, **score(testing.label, probability)))
                    predictions.append(testing[KEY + ["label"]].assign(branch=branch, model=choice,
                                                                      probability=probability, predicted=(probability >= .5).astype(int)))
            print(branch, person, "selected", selected, flush=True)
    table = pd.DataFrame(folds)
    summary = table.groupby(["branch", "model"])[["balanced_accuracy", "awake_recall", "drowsy_recall"]].mean()
    summary.to_csv(output / "transfer_comparison.csv", encoding="utf-8-sig")
    table.to_csv(output / "transfer_folds.csv", index=False)
    pd.DataFrame(public_folds).to_csv(output / "public_loso.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "target_predictions_PROVISIONAL.csv", index=False)
    report = dict(status="DEVELOPMENT_ONLY_PROVISIONAL_KSS", source_url="https://doi.org/10.6084/m9.figshare.14273687.v3",
                  channel="Fp1, index 0 in author code channel order", source_shape=list(eeg.shape),
                  source_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  target_windows=len(target), source_weights=OPTIONS, selection=choices,
                  limits=["Source lab EEG at 128 Hz versus NeuroSky at 512 Hz resampled to 128 Hz.",
                          "Source published binary state and provisional N16 KSS labels use different protocols.",
                          "S01/S02 were repeatedly inspected during development. S03 independent labeled validation pending.",
                          "Public epochs have no continuous timeline, so cannot establish warning latency or alarm episode rates."],
                  configurations=branches, folds=folds)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)
    print("Public LOSO", pd.DataFrame(public_folds).groupby("branch").balanced_accuracy.mean().to_dict(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.dataset, args.pilot, args.output)
