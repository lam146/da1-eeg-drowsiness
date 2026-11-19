"""Evaluate C++-compatible logistic weights and a threshold chosen on training days."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler

from develop_pilot_model import FEATURES, session_weights
from improve_pilot_ml import KEY, scored
from n16_eeg.model import LogisticModel, fit_logistic_model


def fit(frame):
    x = frame[FEATURES].to_numpy(float)
    scaler = RobustScaler().fit(x)
    model = LogisticRegression(C=.1, max_iter=3000).fit(
        scaler.transform(x), frame.label, sample_weight=session_weights(frame))
    compatible = LogisticModel(FEATURES, scaler.center_.tolist(), scaler.scale_.tolist(),
                               model.coef_[0].tolist(), float(model.intercept_[0]))
    np.testing.assert_allclose(compatible.predict_proba(x), model.predict_proba(scaler.transform(x))[:, 1], atol=1e-10)
    return compatible


def threshold_from_training(frame):
    groups = frame.subject_id + '/' + frame.date
    scores = np.empty(len(frame))
    for train, test in LeaveOneGroupOut().split(frame, groups=groups):
        model = fit(frame.iloc[train])
        scores[test] = model.predict_proba(frame.iloc[test][FEATURES].to_numpy(float))
    weights = session_weights(frame)
    labels = frame.label.to_numpy()
    # Fixed grid, default 0.5 preferred on ties. No held-out person enters this search.
    grid = np.arange(25, 76, 5) / 100
    return float(max(grid, key=lambda t: (np.average((scores >= t) == labels, weights=weights), -abs(t - .5))))


def run(pilot, runtime_features, output):
    offline = pd.read_csv(pilot / 'all_subjects_features_PROVISIONAL.csv').sort_values(KEY).reset_index(drop=True)
    frame = pd.read_csv(runtime_features).sort_values(KEY).reset_index(drop=True)
    if (set(frame.subject_id) != {'S01', 'S02'} or offline.duplicated(KEY).any()
            or not frame[KEY + ['label', 'window_end_ms']].equals(offline[KEY + ['label', 'window_end_ms']])):
        raise ValueError('Expected the same provisional S01/S02 windows')
    frame['date'] = offline.source_file.str.extract(r'_(\d{8})_')[0]
    if frame.date.isna().any():
        raise ValueError('Missing date needed for inner validation')
    folds, predictions = [], []
    for train, test in LeaveOneGroupOut().split(frame, groups=frame.subject_id):
        training = frame.iloc[train]
        previous = fit_logistic_model(training[FEATURES].to_numpy(), training.label.to_numpy(), FEATURES)
        revised = fit(training)
        threshold = threshold_from_training(training)
        held_out = frame.subject_id.iloc[test[0]]
        for name, model, cut in [('cpp_previous', previous, .5), ('cpp_weighted_fixed', revised, .5),
                                  ('cpp_weighted_tuned', revised, threshold)]:
            score = model.predict_proba(frame.iloc[test][FEATURES].to_numpy())
            result, rows = scored(frame, test, score, (score >= cut).astype(int))
            folds.append(dict(model=name, held_out=held_out, threshold=cut, **result))
            predictions.append(rows.assign(model=name))
    summary = pd.DataFrame(folds).groupby('model')[['accuracy', 'balanced_accuracy', 'macro_f1', 'recall_tinh',
                                                     'recall_buon_ngu', 'roc_auc']].mean()
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / 'runtime_oof_PROVISIONAL.csv', index=False)
    fixed = fit(frame)
    fixed.save(output / 'runtime_weighted_fixed_PROVISIONAL.json')
    fixed.threshold = threshold_from_training(frame)
    fixed.save(output / 'runtime_weighted_tuned_PROVISIONAL.json')
    report = dict(status='DEVELOPMENT_ONLY_PROVISIONAL_KSS', windows=len(frame), folds=folds,
                  mean_by_person=summary.to_dict(orient='index'), prototype_threshold=fixed.threshold)
    (output / 'runtime_comparison_PROVISIONAL.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('runtime_features', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.pilot, args.runtime_features, args.output)
