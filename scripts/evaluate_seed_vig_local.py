"""Evaluate released forehead EEG features with provisional filename-prefix groups."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from n16_eeg.sklearn_train import make_estimator


ALERT_MAX, DROWSY_MIN = .25, .75


def load_pairs(root):
    eeg = {p.stem: p for p in (root/'Forehead_EEG/EEG_Feature_5Bands').glob('*.mat')}
    labels = {p.stem: p for p in (root/'perclos_labels').glob('*.mat')}
    if not eeg or eeg.keys() != labels.keys():
        raise ValueError('Expected matching EEG and PERCLOS files.')
    features, metadata = [], []
    for session in sorted(eeg):
        match = re.fullmatch(r'(\d+)_\d{8}_(?:noon|night)(?:_\d+)?', session)
        if match is None:
            raise ValueError(f'Unexpected release filename: {session}')
        psd = np.asarray(loadmat(eeg[session])['psd_movingAve'], dtype=float)
        perclos = np.asarray(loadmat(labels[session])['perclos'], dtype=float)
        if (psd.shape != (4, 885, 5) or perclos.shape != (885, 1)
                or not np.isfinite(psd).all() or not np.isfinite(perclos).all()
                or not ((0 <= perclos) & (perclos <= 1)).all()):
            raise ValueError(f'Invalid release arrays: {session}')
        features.append(psd.transpose(1, 0, 2).reshape(885, 20))
        metadata.append(pd.DataFrame(dict(session=session, prefix_group=match[1],
                                          row=np.arange(885), perclos=perclos[:, 0])))
    return np.concatenate(features), pd.concat(metadata, ignore_index=True)


def run(root, output):
    x, rows = load_pairs(root)
    y = rows.perclos.to_numpy()
    extreme = (y <= ALERT_MAX) | (y >= DROWSY_MIN)
    binary = (y >= DROWSY_MIN).astype(int)
    rows['ridge'] = np.nan
    rows['train_mean'] = np.nan
    rows['logistic'] = np.nan
    folds = []
    for train, test in LeaveOneGroupOut().split(x, groups=rows.prefix_group):
        assert set(rows.prefix_group.iloc[train]).isdisjoint(rows.prefix_group.iloc[test])
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1))
        prediction = ridge.fit(x[train], y[train]).predict(x[test]).clip(0, 1)
        baseline = np.full(len(test), y[train].mean())
        rows.loc[test, 'ridge'] = prediction
        rows.loc[test, 'train_mean'] = baseline
        tr, te = train[extreme[train]], test[extreme[test]]
        model = make_estimator('logistic').fit(x[tr], binary[tr])
        predicted_class = model.predict(x[te]) if len(te) else np.array([], dtype=int)
        rows.loc[te, 'logistic'] = predicted_class
        folds.append(dict(
            group=rows.prefix_group.iloc[test[0]], rows=len(test), binary_rows=len(te),
            rmse=root_mean_squared_error(y[test], prediction),
            mae=mean_absolute_error(y[test], prediction),
            baseline_rmse=root_mean_squared_error(y[test], baseline),
            balanced_accuracy=balanced_accuracy_score(binary[te], predicted_class)
            if np.unique(binary[te]).size == 2 else None,
        ))
    assert rows.ridge.notna().all() and np.array_equal(rows.logistic.notna(), extreme)
    report = dict(sessions=int(rows.session.nunique()), groups=len(folds), rows=len(rows),
                  group_basis='Numeric filename prefix; person identity unverified',
                  features='Forehead EEG psd_movingAve (4 channels x 5 bands)',
                  alignment='Matched release stems and row order; no independent timestamp verification',
                  alert_max=ALERT_MAX, drowsy_min=DROWSY_MIN,
                  alert_rows=int((y <= ALERT_MAX).sum()), drowsy_rows=int((y >= DROWSY_MIN).sum()),
                  middle_rows=int((~extreme).sum()), folds=folds)
    for key in ['rmse', 'mae', 'baseline_rmse', 'balanced_accuracy']:
        values = [fold[key] for fold in folds if fold[key] is not None]
        report[key] = dict(mean=float(np.mean(values)) if values else None, groups=len(values))
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output/'oof_local.csv', index=False)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return {key: value for key, value in report.items() if key != 'folds'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output), indent=2))
