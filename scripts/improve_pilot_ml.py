"""Audit and compare EEG pilot models on the same held-out people and windows."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from develop_pilot_model import FEATURES, estimator, session_weights
from refine_pilot_model import metrics


KEY = ['subject_id', 'session_id', 'window_start_ms']
SPECTRAL = FEATURES[:-3]
BANDS = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 45))
ROBUST = [f'log_{band}_beta' for band in ('delta', 'theta', 'alpha', 'gamma')] + ['entropy']
CANDIDATES = {
    'previous_combined': ('offline', FEATURES, 'previous'),
    'runtime_spectral': ('runtime', SPECTRAL, 'previous'),
    'runtime_combined': ('runtime', FEATURES, 'previous'),
    'welch_spectral': ('welch', ROBUST, 'logistic'),
    'welch_svm': ('welch', ROBUST, 'svm'),
    'welch_boost': ('welch', ROBUST, 'boost'),
    'welch_combined': ('welch', ROBUST + FEATURES[-3:], 'logistic'),
    'welch_history_20s': ('welch', [f'history_{c}' for c in ROBUST], 'logistic'),
    'welch_history_combined': ('welch', [f'history_{c}' for c in ROBUST] + FEATURES[-3:], 'logistic'),
}
CALIBRATION = ('welch', [f'opening_delta_{c}' for c in ROBUST], 'logistic')
REFERENCE = ('welch', ROBUST, 'logistic')
SAMPLE_RATE = 512
WINDOW = SAMPLE_RATE * 4
OPENING_WINDOWS = 15


def welch_features(signal):
    """Average three Hann periodograms with 50% overlap within each 4 s window."""
    n = 2 * SAMPLE_RATE
    starts = (0, n // 2, n)
    chunks = np.stack([signal[:, start:start + n] for start in starts], axis=1)
    tapered = (chunks - chunks.mean(axis=-1, keepdims=True)) * np.hanning(n)
    psd = np.square(np.abs(np.fft.rfft(tapered, axis=-1))).mean(axis=1)
    freq = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    powers = np.column_stack([psd[:, (freq >= lo) & (freq < hi)].sum(axis=1) for lo, hi in BANDS])
    log_power = np.log(np.maximum(powers, np.finfo(float).eps))
    bins = psd[:, (freq >= 1) & (freq < 45)]
    density = bins / np.maximum(bins.sum(axis=1, keepdims=True), np.finfo(float).eps)
    entropy = -(density * np.log(np.maximum(density, np.finfo(float).eps))).sum(axis=1) / np.log(bins.shape[1])
    return np.column_stack([log_power[:, [0, 1, 2, 4]] - log_power[:, 3:4], entropy])


def load_branches(pilot, runtime):
    source = pilot / 'all_subjects_features_PROVISIONAL.csv'
    frame = pd.read_csv(source).sort_values(KEY).reset_index(drop=True)
    frame['date'] = frame.source_file.str.extract(r'_(\d{8})_')[0]
    if (set(frame.subject_id) != {'S01', 'S02'} or frame.duplicated(KEY).any()
            or frame.date.isna().any() or set(frame.fs_hz) != {SAMPLE_RATE}
            or not frame.time_alignment_verified.eq(False).all()
            or not frame.source_is_synthetic.eq(False).all()):
        raise ValueError('Expected unique provisional windows from S01 and S02')
    runtime_frame = pd.read_csv(runtime).sort_values(KEY).reset_index(drop=True)
    if not runtime_frame[KEY + ['label', 'window_end_ms']].equals(frame[KEY + ['label', 'window_end_ms']]):
        raise ValueError('Runtime and offline window keys or labels differ')
    welch = frame[FEATURES[-3:]].copy()
    for (person, session), group in frame.groupby(['subject_id', 'session_id']):
        path = pilot / person / session / 'eeg_filtered_offline.csv'
        data = pd.read_csv(path, usecols=['sample_index', 'eeg_filtered_count', 'quality_screen'])
        if not np.array_equal(data.sample_index, np.arange(len(data))):
            raise ValueError(f'Unexpected sample index in {person}/{session}')
        count = len(data) // WINDOW
        signal = data.eeg_filtered_count.to_numpy(float)[:count * WINDOW].reshape(count, WINDOW)
        quality = data.quality_screen.to_numpy()[:count * WINDOW].reshape(count, WINDOW)
        valid = np.isfinite(signal).all(axis=1) & ~np.isin(quality, ['exclude_screening', 'review_filter_edge']).any(axis=1)
        values = pd.DataFrame(welch_features(np.nan_to_num(signal)), columns=ROBUST)
        indices = (group.window_start_ms / 4000).to_numpy(int)
        if not (np.array_equal(group.window_start_ms, indices * 4000) and valid[indices].all()):
            raise ValueError(f'Window audit mismatch in {person}/{session}')
        history = values.loc[valid].groupby((~valid).cumsum()[valid]).rolling(5, min_periods=1).mean()
        history = history.droplevel(0).reindex(values.index)
        opening = values.iloc[:OPENING_WINDOWS].loc[valid[:OPENING_WINDOWS]]
        reference = opening.median() if len(opening) >= 8 else pd.Series(np.nan, index=ROBUST)
        delta = values - reference
        delta.iloc[:OPENING_WINDOWS] = np.nan
        for prefix, branch in [('', values), ('history_', history), ('opening_delta_', delta)]:
            welch.loc[group.index, [prefix + c for c in ROBUST]] = branch.iloc[indices].to_numpy()
    return frame, dict(offline=frame, runtime=runtime_frame, welch=welch), hashlib.sha256(source.read_bytes()).hexdigest()


def fit_model(config, frame, branches, train):
    branch, columns, method = config
    features = branches[branch][columns]
    weights = session_weights(frame.iloc[train])
    if method == 'previous':
        model = estimator(weighted=True)
        fit_options = {'classifier__sample_weight': weights}
    elif method == 'boost':
        model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, min_samples_leaf=40,
                                               l2_regularization=2, early_stopping=False, random_state=16)
        fit_options = {'sample_weight': weights}
    else:
        classifier = (LogisticRegression(C=.1, max_iter=3000) if method == 'logistic'
                      else SVC(C=.3, gamma='scale'))
        model = make_pipeline(RobustScaler(), classifier)
        fit_options = {f'{model.steps[-1][0]}__sample_weight': weights}
    return model.fit(features.iloc[train], frame.label.iloc[train], **fit_options)


def fit_and_predict(config, frame, branches, train, test):
    branch, columns, method = config
    features = branches[branch][columns].iloc[test]
    model = fit_model(config, frame, branches, train)
    if method == 'svm':
        score = model.decision_function(features)
        return score, (score >= 0).astype(int)
    score = model.predict_proba(features)[:, 1]
    return score, (score >= .5).astype(int)


def choose_in_training(frame, branches, train):
    training_rows = frame.iloc[train]
    groups = training_rows.subject_id + '/' + training_rows.date
    splits = list(LeaveOneGroupOut().split(training_rows, groups=groups))
    if any(training_rows.iloc[fit].label.nunique() != 2 for fit, _ in splits):
        raise ValueError('Inner held-out-day validation needs both classes in each training split')
    weights = session_weights(training_rows)
    results = {}
    for name, config in CANDIDATES.items():
        predictions = np.empty(len(train), dtype=int)
        for fit, valid in splits:
            _, predictions[valid] = fit_and_predict(config, frame, branches, train[fit], train[valid])
        results[name] = float(np.average(predictions == training_rows.label.to_numpy(), weights=weights))
    return max(results, key=results.get), results


def scored(frame, test, scores, predictions):
    rows = frame.iloc[test][KEY + ['window_end_ms', 'label']].assign(score=scores, prediction=predictions)
    result = metrics(rows)
    result['roc_auc'] = roc_auc_score(rows.label, rows.score)
    return result, rows


def audit_labels(frame, pilot):
    kss = pd.read_csv(pilot / 'kss_observations.csv')
    clocks = kss.wall_clock_as_recorded.str.replace('.', ':', regex=False).str.split(':', expand=True).astype(int)
    minutes = clocks[0] * 60 + clocks[1]
    opening = minutes.groupby([kss.subject_id, kss.session_id]).transform('first')
    elapsed = (minutes - opening) % 1440
    mismatch = kss.loc[elapsed.ne(kss.planned_minute), ['subject_id', 'session_id', 'planned_minute', 'kss']]
    days = frame.groupby(['subject_id', 'date', 'label']).size().rename('windows').reset_index()
    return dict(days=days.to_dict('records'),
                sessions_used=int(frame.groupby(['subject_id', 'session_id']).ngroups),
                single_class_days=int(days.groupby(['subject_id', 'date']).label.nunique().eq(1).sum()),
                clock_mismatches=mismatch.assign(elapsed_minute=elapsed.loc[mismatch.index]).to_dict('records'))


def run(pilot, runtime, output):
    frame, branches, digest = load_branches(pilot, runtime)
    folds, predictions, selections, calibrated = [], [], {}, []
    comparison_mask = branches['welch'][CALIBRATION[1]].notna().all(axis=1).to_numpy()
    for train, test in LeaveOneGroupOut().split(frame, groups=frame.subject_id):
        held_out = frame.subject_id.iloc[test[0]]
        selected, inner_scores = choose_in_training(frame, branches, train)
        selections[held_out] = dict(selected=selected, inner_scores=inner_scores)
        for name, config in CANDIDATES.items():
            score, pred = fit_and_predict(config, frame, branches, train, test)
            result, rows = scored(frame, test, score, pred)
            folds.append(dict(model=name, held_out=held_out, **result))
            predictions.append(rows.assign(model=name))
            if name == selected:
                folds.append(dict(model='selected_inside_training', held_out=held_out, **result))
                predictions.append(rows.assign(model='selected_inside_training'))
        subtrain, subtest = train[comparison_mask[train]], test[comparison_mask[test]]
        for name, config in [('same_windows_without_calibration', REFERENCE), ('opening_60s_calibration', CALIBRATION)]:
            score, pred = fit_and_predict(config, frame, branches, subtrain, subtest)
            result, _ = scored(frame, subtest, score, pred)
            calibrated.append(dict(model=name, held_out=held_out, **result))

    # Personalized branch answers a separate question: known wearer, unseen day.
    personalized = []
    for person, rows in frame.groupby('subject_id'):
        outputs = []
        for train, test in LeaveOneGroupOut().split(rows, groups=rows.date):
            tr, te = rows.index[train].to_numpy(), rows.index[test].to_numpy()
            score, pred = fit_and_predict(CANDIDATES['previous_combined'], frame, branches, tr, te)
            outputs.append(frame.iloc[te][['label']].assign(score=score, prediction=pred))
        oof = pd.concat(outputs)
        personalized.append(dict(subject=person, **metrics(oof), roc_auc=roc_auc_score(oof.label, oof.score)))

    prediction_frame = pd.concat(predictions, ignore_index=True)
    interior = prediction_frame.window_start_ms.mod(300000).ge(30000) & (
        prediction_frame.window_end_ms - prediction_frame.window_start_ms.floordiv(300000) * 300000).le(270000)
    sensitivity = [dict(model=name, held_out=person, **metrics(rows)) for (name, person), rows in
                   prediction_frame.loc[interior].groupby(['model', 'subject_id'])
                   if name in ('previous_combined', 'selected_inside_training')]
    columns = ['accuracy', 'balanced_accuracy', 'macro_f1', 'recall_tinh', 'recall_buon_ngu', 'roc_auc']
    summary = pd.DataFrame(folds).groupby('model')[columns].mean().sort_values('balanced_accuracy', ascending=False)
    all_rows = np.arange(len(frame))
    prototype, all_inner = choose_in_training(frame, branches, all_rows)
    fitted = fit_model(CANDIDATES[prototype], frame, branches, all_rows)
    report = dict(status='DEVELOPMENT_ONLY_PROVISIONAL_KSS', source_sha256=digest, windows=len(frame),
                  label_audit=audit_labels(frame, pilot), candidate_definitions=CANDIDATES,
                  inner_selection=selections, folds=folds, mean_by_person=summary.to_dict(orient='index'),
                  opening_calibration_matched_subset=calibrated, boundary_sensitivity=sensitivity,
                  personalized_unseen_day=personalized, prototype_selection=prototype, prototype_inner_scores=all_inner,
                  limitations=['S01 and S02 were already inspected during model development.',
                               'Offline SOS features and runtime FFT features are different pipelines.',
                               '20 s aggregation uses current and previous windows, but underlying offline SOS filtering is noncausal.',
                               'Opening calibration excludes the first 60 s and uses no KSS from the held-out person.',
                               'Personalized held-out-day scores do not measure performance on a new wearer.'])
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / 'comparison.csv')
    prediction_frame.to_csv(output / 'oof_PROVISIONAL.csv', index=False)
    (output / 'metrics_PROVISIONAL.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    joblib.dump(dict(model=fitted, config=CANDIDATES[prototype], status=report['status']), output / 'prototype_offline.joblib')
    print(summary.to_string())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('runtime_features', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.pilot, args.runtime_features, args.output)
