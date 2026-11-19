"""Select a pilot EEG model within training sessions and assess on held-out people."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from develop_pilot_model import FEATURES, estimator, session_weights


SPECTRAL = FEATURES[:-3]
CANDIDATES = {
    'spectral_c0.1': (SPECTRAL, .1),
    'combined_c0.1': (FEATURES, .1),
    'spectral_c1': (SPECTRAL, 1.0),
    'combined_c1': (FEATURES, 1.0),
}


def fit_candidate(frame, name):
    features, regularization = CANDIDATES[name]
    model = clone(estimator(weighted=True)).set_params(classifier__C=regularization)
    return model.fit(frame[features], frame.label, classifier__sample_weight=session_weights(frame))


def predict(model, frame):
    return model.predict_proba(frame[list(model.feature_names_in_)])[:, 1]


def choose_candidate(frame):
    groups = frame.subject_id + '/' + frame.session_id
    splits = list(LeaveOneGroupOut().split(frame, groups=groups))
    if any(frame.iloc[train].label.nunique() != 2 for train, _ in splits):
        raise ValueError('Session validation requires both classes in every inner training split.')
    scores = {}
    for name in CANDIDATES:
        probability = np.empty(len(frame))
        for train, valid in splits:
            model = fit_candidate(frame.iloc[train], name)
            probability[valid] = predict(model, frame.iloc[valid])
        scores[name] = float(np.average((probability >= .5) == frame.label,
                                        weights=session_weights(frame)))
    return max(scores, key=scores.get), scores


def metrics(rows):
    matrix = confusion_matrix(rows.label, rows.prediction, labels=[0, 1])
    return {
        'windows': len(rows),
        'accuracy': accuracy_score(rows.label, rows.prediction),
        'balanced_accuracy': balanced_accuracy_score(rows.label, rows.prediction),
        'macro_f1': f1_score(rows.label, rows.prediction, average='macro', labels=[0, 1]),
        'recall_tinh': matrix[0, 0] / matrix[0].sum(),
        'recall_buon_ngu': matrix[1, 1] / matrix[1].sum(),
        'confusion_matrix': matrix.tolist(),
    }


def run(source, target):
    frame = pd.read_csv(source)
    key = ['subject_id', 'session_id', 'window_start_ms']
    if (set(frame.subject_id) != {'S01', 'S02'} or set(frame.label) != {0, 1}
            or frame.duplicated(key).any() or not frame.time_alignment_verified.eq(False).all()
            or frame.label_policy.nunique() != 1
            or not np.isfinite(frame[FEATURES]).all().all() or frame[FEATURES].lt(0).any().any()):
        raise ValueError('Unexpected pilot data or label policy.')
    results, predictions = {}, []
    for person in sorted(frame.subject_id.unique()):
        train = frame.loc[frame.subject_id.ne(person)]
        test = frame.loc[frame.subject_id.eq(person)]
        selected, scores = choose_candidate(train)
        model = fit_candidate(train, selected)
        probability = predict(model, test)
        rows = test[key + ['window_end_ms', 'label']].assign(
            probability_drowsy=probability, prediction=(probability >= .5).astype(int))
        predictions.append(rows)
        # Sensitivity analysis only. Keep at least 30 s from each nominal KSS boundary.
        start = rows.window_start_ms
        end = rows.window_end_ms
        interior = start.mod(300000).ge(30000) & (end - start.floordiv(300000)*300000).le(270000)
        results[person] = dict(train_subjects=sorted(train.subject_id.unique()),
                               selected=selected, inner_scores=scores, all=metrics(rows),
                               interior=metrics(rows.loc[interior]))
    oof = pd.concat(predictions, ignore_index=True)
    assert len(oof) == len(frame) and not oof.duplicated(key).any()
    report = dict(status='DEVELOPMENT_WITH_PROVISIONAL_LABELS',
                  source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  label_policy=frame.label_policy.iloc[0], candidates=CANDIDATES, threshold=.5,
                  folds=results)
    for subset in ['all', 'interior']:
        report[subset] = {metric: float(np.mean([r[subset][metric] for r in results.values()]))
                          for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']}
    final_name, final_scores = choose_candidate(frame)
    report.update(prototype=final_name, prototype_inner_scores=final_scores)
    target.mkdir(parents=True, exist_ok=False)
    joblib.dump(fit_candidate(frame, final_name), target/'prototype_offline.joblib')
    oof.to_csv(target/'oof_PROVISIONAL.csv', index=False)
    oof.assign(correct=oof.label.eq(oof.prediction)).groupby(
        ['subject_id', 'session_id', 'label']).correct.agg(['size', 'sum', 'mean']).to_csv(
            target/'session_diagnostics.csv')
    (target/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.target), indent=2))
