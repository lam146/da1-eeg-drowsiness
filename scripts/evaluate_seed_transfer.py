"""Compare NeuroSky-only training with fixed-weight SEED-VIG assistance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from develop_pilot_model import session_weights
from evaluate_seed_vig_local import ALERT_MAX, DROWSY_MIN, load_pairs
from n16_eeg.signal_features import _periodogram


FS = 512
BANDS = ((1, 4), (4, 8), (8, 14), (14, 31))
FEATURES = ("log_delta_beta", "log_theta_beta", "log_alpha_beta")
SOURCE_WEIGHT = 0.20
C = 0.1


def log_band_ratios(power):
    if not np.isfinite(power).all() or (power < 0).any():
        raise ValueError("Expected finite, nonnegative band powers")
    log_power = np.log(np.maximum(power, np.finfo(float).tiny))
    return log_power[..., :3] - log_power[..., 3:4]


def target_features(pilot):
    path = pilot / "all_subjects_features_PROVISIONAL.csv"
    rows = pd.read_csv(path)
    if set(rows.subject_id) != {"S01", "S02"} or set(rows.label) != {0, 1} or set(rows.fs_hz) != {FS}:
        raise ValueError("Expected the reviewed two-person NeuroSky feature table")
    values = np.empty((len(rows), len(BANDS)))
    for (person, session), group in rows.groupby(["subject_id", "session_id"]):
        signal = pd.read_csv(pilot / person / session / "eeg_filtered_offline.csv",
                             usecols=["sample_index", "eeg_filtered_count"])
        if not np.array_equal(signal.sample_index, np.arange(len(signal))):
            raise ValueError(f"Unexpected sample index in {person}/{session}")
        eeg = signal.eeg_filtered_count.to_numpy(float)
        for row in group.itertuples():
            start = round(row.window_start_ms * FS / 1000)
            end = round(row.window_end_ms * FS / 1000)
            if start < 0 or end > len(eeg) or end - start != 4 * FS:
                raise ValueError(f"Unexpected window in {person}/{session}")
            freqs, psd = _periodogram(eeg[start:end], FS)
            values[row.Index] = [psd[(freqs >= lo) & (freqs < hi)].sum() * FS / (end - start)
                                 for lo, hi in BANDS]
    return log_band_ratios(values), rows, hashlib.sha256(path.read_bytes()).hexdigest()


def source_features(dataset):
    psd, rows = load_pairs(dataset)
    eligible = rows.perclos.le(ALERT_MAX) | rows.perclos.ge(DROWSY_MIN)
    selected = rows.loc[eligible].rename(columns={"prefix_group": "subject_id", "session": "session_id"}).copy()
    selected["label"] = selected.perclos.ge(DROWSY_MIN).astype(int)
    # One channel per example. Four correlated views share the weight of one time row.
    x = log_band_ratios(psd[eligible].reshape(-1, 4, 5)[..., :4]).reshape(-1, len(FEATURES))
    weights = np.repeat(session_weights(selected) / 4, 4)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    return scaler.transform(x), np.repeat(selected.label.to_numpy(), 4), weights, selected


def run(dataset, pilot, output):
    source_x, source_y, source_weights, source_rows = source_features(dataset)
    x, rows, target_hash = target_features(pilot)
    y = rows.label.to_numpy()
    folds, predictions = [], []
    for train, test in LeaveOneGroupOut().split(x, y, rows.subject_id):
        target_weights = session_weights(rows.iloc[train])
        scaler = StandardScaler().fit(x[train], sample_weight=target_weights)
        train_x, test_x = scaler.transform(x[train]), scaler.transform(x[test])
        train_y = y[train]
        variants = {
            "neurosky_only": (train_x, train_y, target_weights),
            "seed_assisted": (
                np.vstack([train_x, source_x]),
                np.concatenate([train_y, source_y]),
                np.concatenate([(1 - SOURCE_WEIGHT) * target_weights,
                                SOURCE_WEIGHT * target_weights.sum() * source_weights / source_weights.sum()]),
            ),
        }
        person = rows.subject_id.iloc[test[0]]
        for name, (fit_x, fit_y, weights) in variants.items():
            model = LogisticRegression(C=C, max_iter=3000).fit(fit_x, fit_y, sample_weight=weights)
            prob = model.predict_proba(test_x)[:, 1]
            predicted = (prob >= 0.5).astype(int)
            cm = confusion_matrix(y[test], predicted, labels=[0, 1])
            folds.append(dict(model=name, held_out=person, train_people=sorted(rows.subject_id.iloc[train].unique()),
                              windows=len(test), accuracy=accuracy_score(y[test], predicted),
                              balanced_accuracy=balanced_accuracy_score(y[test], predicted),
                              macro_f1=f1_score(y[test], predicted, average="macro"),
                              auc=roc_auc_score(y[test], prob), confusion_matrix=cm.tolist(),
                              recall_by_class=(cm.diagonal() / cm.sum(axis=1)).tolist(),
                              target_scaler_mean=scaler.mean_.tolist(),
                              coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0])))
            predictions.append(pd.DataFrame(dict(model=name, row_index=test, subject_id=person,
                                                 label=y[test], prediction=predicted, probability=prob)))
    summary = pd.DataFrame(folds).groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "auc"]].mean()
    report = dict(status="EXPLORATORY_EXISTING_PILOT",
                  target_windows=len(rows), target_table_sha256=target_hash,
                  source_sessions=int(source_rows.session_id.nunique()), source_time_rows=len(source_rows),
                  source_channel_examples=len(source_y), source_prefix_groups=int(source_rows.subject_id.nunique()),
                  source_label_rule=dict(alert_max=ALERT_MAX, drowsy_min=DROWSY_MIN),
                  bands_hz=BANDS, features=FEATURES, source_loss_weight=SOURCE_WEIGHT,
                  regularization_C=C, probability_threshold=0.5, folds=folds,
                  mean_by_person=summary.to_dict(orient="index"),
                  limitations=["Released smoothed four-channel forehead PSD versus single-channel 4 s NeuroSky spectra.",
                               "PERCLOS and provisional KSS measure different aspects of vigilance.",
                               "SEED filename prefixes are weighting groups with unverified person identity.",
                               "The two NeuroSky people were already used for development."])
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions_PROVISIONAL.csv", index=False)
    (output / "metrics_PROVISIONAL.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.pilot, args.output), indent=2))
