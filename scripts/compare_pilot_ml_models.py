"""Compare fixed pilot ML candidates and audit selection on training days only."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from develop_pilot_model import FEATURES, session_weights
from evaluate_python_alert_latency import FS, feature_vector, onset_test, prepare
from improve_pilot_ml import KEY, scored
from n16_eeg.model import LogisticModel
from n16_eeg.simulate import generate_session


SPECTRAL = FEATURES[:-3]
BLINK = FEATURES[-3:]
RATIOS = ['theta_alpha', 'theta_beta', 'alpha_beta', 'spectral_entropy']
FAMILIES = {
    'logistic_combined': (FEATURES, 'linear', .1),
    'logistic_combined_strong_reg': (FEATURES, 'linear', .02),
    'logistic_combined_weak_reg': (FEATURES, 'linear', 1.),
    'logistic_spectral': (SPECTRAL, 'linear', .1),
    'logistic_ratios': (RATIOS, 'linear', .1),
    'logistic_blink': (BLINK, 'linear', .1),
    'log1p_combined': (FEATURES, 'log', .1),
    'log1p_spectral': (SPECTRAL, 'log', .1),
    'log1p_ratios_blink': (RATIOS + BLINK, 'log', .1),
    'quantile_combined': (FEATURES, 'quantile', .1),
    'quantile_spectral': (SPECTRAL, 'quantile', .1),
    'rbf_combined': (FEATURES, 'svm', .5),
    'rbf_spectral': (SPECTRAL, 'svm', .5),
    'rbf_ratios': (RATIOS, 'svm', .5),
    'forest_combined': (FEATURES, 'forest', 4),
    'forest_spectral': (SPECTRAL, 'forest', 4),
    'extra_trees_combined': (FEATURES, 'extra', 4),
    'extra_trees_spectral': (SPECTRAL, 'extra', 4),
    'boost_combined': (FEATURES, 'boost', 7),
    'boost_spectral': (SPECTRAL, 'boost', 7),
    'bayes_spectral': (SPECTRAL, 'bayes', .01),
}
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'recall_tinh', 'recall_buon_ngu', 'roc_auc']


def fit_model(frame, name, weights=None):
    columns, method, param = FAMILIES[name]
    if method in ('linear', 'log', 'quantile'):
        scale = (QuantileTransformer(n_quantiles=64, output_distribution='normal', random_state=16)
                 if method == 'quantile' else RobustScaler())
        preprocessing = [FunctionTransformer(np.log1p)] if method == 'log' else []
        model = make_pipeline(*preprocessing, scale, LogisticRegression(C=param, max_iter=3000))
    elif method == 'svm':
        model = make_pipeline(FunctionTransformer(np.log1p), RobustScaler(), SVC(C=param, gamma='scale'))
    elif method in ('forest', 'extra'):
        kind = RandomForestClassifier if method == 'forest' else ExtraTreesClassifier
        model = kind(n_estimators=80, max_depth=param, min_samples_leaf=25, random_state=16, n_jobs=1)
    elif method == 'boost':
        model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=param, min_samples_leaf=40,
                                               l2_regularization=2., early_stopping=False, random_state=16)
    else:
        model = GaussianNB(var_smoothing=param)
    if weights is None:
        weights = session_weights(frame)
    fit_params = {f'{model.steps[-1][0]}__sample_weight': weights} if hasattr(model, 'steps') else {'sample_weight': weights}
    return model.fit(frame[columns], frame.label, **fit_params)


def score(model, frame, name):
    columns, method, _ = FAMILIES[name]
    x = frame[columns]
    if method == 'svm':
        values = model.decision_function(x)
        return values, (values >= 0).astype(int)
    values = model.predict_proba(x)[:, 1]
    return values, (values >= .5).astype(int)


def choose_on_days(frame):
    groups = frame.subject_id + '/' + frame.date
    splits = list(LeaveOneGroupOut().split(frame, groups=groups))
    if any(frame.iloc[train].label.nunique() != 2 for train, _ in splits):
        raise ValueError('Every training-day split needs both classes')
    splits = [(frame.iloc[train], test, session_weights(frame.iloc[train])) for train, test in splits]
    weights = session_weights(frame)
    labels = frame.label.to_numpy()
    result = {}
    for name in FAMILIES:
        predictions = np.empty(len(frame), dtype=int)
        for train, test, fit_weights in splits:
            _, predictions[test] = score(fit_model(train, name, fit_weights), frame.iloc[test], name)
        recalls = [float(np.average(predictions[labels == k] == k, weights=weights[labels == k])) for k in (0, 1)]
        result[name] = dict(balanced_accuracy=float(np.mean(recalls)), recall_drowsy=recalls[1])
    selected = max(result, key=lambda name: (result[name]['balanced_accuracy'], result[name]['recall_drowsy']))
    return selected, result


class PythonReplayModel:
    """Adapt a candidate to the live detector without changing the live model format."""

    threshold = .5

    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    def predict_feature_dict(self, features):
        columns, method, _ = FAMILIES[self.name]
        row = pd.DataFrame([[features[name] for name in columns]], columns=columns)
        if method == 'svm':
            margin = self.estimator.decision_function(row)[0]
            return float(1 / (1 + np.exp(-np.clip(margin, -30, 30))))
        return float(self.estimator.predict_proba(row)[0, 1])


def timing_fixtures(fixture):
    return [pd.read_csv(fixture)] + [generate_session(alert_seconds=12 + phase, drowsy_seconds=12, seed=seed)
                                     for seed in (51, 71, 103, 127) for phase in (.05, .23, .47, .71, .93)]


def export_spectral_candidate(frame, output):
    estimator = fit_model(frame, 'logistic_spectral')
    scaler, classifier = (step for _, step in estimator.steps)
    model = LogisticModel(SPECTRAL, scaler.center_.tolist(), scaler.scale_.tolist(),
                          classifier.coef_[0].tolist(), float(classifier.intercept_[0]))
    x = frame[SPECTRAL]
    np.testing.assert_allclose(model.predict_proba(x.to_numpy()), estimator.predict_proba(x)[:, 1], atol=1e-9)
    model.save(output / 'python_model_3s_spectral_PROVISIONAL.json')


def run_shortlist(pilot, previous, fixture, output):
    meta, features, fragments = prepare(pilot)
    frame = pd.concat([meta[KEY[:2] + ['label']], pd.DataFrame(features[3], columns=FEATURES)], axis=1)
    configs = ('logistic_combined', 'logistic_spectral', 'rbf_combined')
    models = {name: {person: fit_model(frame.loc[frame.subject_id.ne(person)], name)
                     for person in frame.subject_id.unique()} for name in configs}
    raw_fixtures = timing_fixtures(fixture)
    recordings, timings = [], {}
    for person_session, label, raw in fragments:
        person, session = person_session
        x = pd.DataFrame([feature_vector(raw[end - 3 * FS:end])
                          for end in range(4 * FS, len(raw) + 1, FS // 4)], columns=FEATURES)
        for name in configs:
            _, high_risk = score(models[name][person], x, name)
            alarm = np.convolve(high_risk, np.ones(3, dtype=int), mode='full')[:len(high_risk)] == 3
            recordings.append(dict(model=name, person=person, session=session, label=label,
                                   alarm_fraction=float(alarm.mean())))
    for name in configs:
        model = LogisticModel.load(previous) if name == 'logistic_combined' else PythonReplayModel(fit_model(frame, name), name)
        timings[name] = [onset_test(model, (3, .25, 3), raw) for raw in raw_fixtures]
    rates = pd.DataFrame(recordings).groupby(['model', 'label', 'person']).alarm_fraction.mean()
    report = dict(selection='Post-hoc shortlist from the 3 s comparison; development only.',
                  fragments=len(fragments),
                  pilot_alarm_fractions={name: {str(label): float(rates.loc[name, label].mean()) for label in (0, 1)}
                                         for name in configs}, synthetic=timings)
    (output / 'shortlist_alarm_replay.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    export_spectral_candidate(frame, output)
    print(report['pilot_alarm_fractions'])
    for name, rows in timings.items():
        delays = [row['onset_to_alarm_ms'] for row in rows if row['onset_to_alarm_ms'] is not None]
        print(name, 'under 3s:', sum(delay < 3000 for delay in delays), '/', len(rows),
              'worst ms:', max(delays, default=None), 'early cases:', sum(row['pre_onset_alarm'] for row in rows))


def run(pilot, previous, fixture, output):
    meta, features, _ = prepare(pilot)
    meta['date'] = meta.source_file.str.extract(r'_(\d{8})_')[0]
    if meta.date.isna().any() or not meta.time_alignment_verified.eq(False).all():
        raise ValueError('Expected the original provisional label and date policy')
    prior = LogisticModel.load(previous)
    folds, predictions, inner, fitted = [], [], {}, {}
    for seconds in (3, 4):
        frame = pd.concat([meta[KEY + ['window_end_ms', 'date', 'label']],
                           pd.DataFrame(features[seconds], columns=FEATURES)], axis=1)
        for train, test in LeaveOneGroupOut().split(frame, groups=frame.subject_id):
            held_out = frame.subject_id.iloc[test[0]]
            training = frame.iloc[train]
            fit_weights = session_weights(training)
            selected, ranking = choose_on_days(training)
            inner[f'{seconds}s/{held_out}'] = dict(selected=selected, training_day_scores=ranking)
            for name in FAMILIES:
                values, pred = score(fit_model(training, name, fit_weights), frame.iloc[test], name)
                result, rows = scored(frame, test, values, pred)
                model_names = [name, 'selected_on_training_days'] if name == selected else [name]
                for model_name in model_names:
                    folds.append(dict(window_s=seconds, model=model_name, held_out=held_out, **result))
                    predictions.append(rows.assign(window_s=seconds, model=model_name))
            print(f'{seconds}s held out {held_out}: selected {selected}', flush=True)
        selected, ranking = choose_on_days(frame)
        estimator = fit_model(frame, selected)
        fitted[seconds] = (selected, estimator)
        inner[f'{seconds}s/full'] = dict(selected=selected, training_day_scores=ranking)
        if seconds == 3:
            old = fit_model(frame, 'logistic_combined')
            np.testing.assert_allclose(old.predict_proba(frame[FEATURES])[:, 1], prior.predict_proba(features[3]), atol=1e-9)

    fold_frame = pd.DataFrame(folds)
    summary = fold_frame.groupby(['window_s', 'model'])[METRICS].mean()
    summary['worst_person_balanced_accuracy'] = fold_frame.groupby(['window_s', 'model']).balanced_accuracy.min()
    summary['worst_person_drowsy_recall'] = fold_frame.groupby(['window_s', 'model']).recall_buon_ngu.min()
    summary = summary.reset_index().sort_values(['window_s', 'balanced_accuracy'], ascending=[True, False])

    # Test the 3 s training-day choice on the same alarm schedule as the current Python demo.
    name, estimator = fitted[3]
    fixtures = timing_fixtures(fixture)
    comparisons = {'current_logistic': prior, name: PythonReplayModel(estimator, name)}
    timing = {key: [onset_test(model, (3, .25, 3), raw) for raw in fixtures] for key, model in comparisons.items()}
    report = dict(status='DEVELOPMENT_ONLY_PROVISIONAL_KSS', windows=len(meta),
                  source_table_sha256=hashlib.sha256((pilot / 'all_subjects_features_PROVISIONAL.csv').read_bytes()).hexdigest(),
                  candidate_families=FAMILIES, inner_selection=inner, folds=folds,
                  alarm_replay=dict(scope='Synthetic EEG, sample time to software decision', window_s=3, step_s=.25,
                                    consecutive=3, candidate=name, results=timing))
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / 'comparison.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / 'oof_PROVISIONAL.csv', index=False)
    (output / 'metrics_PROVISIONAL.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for seconds, (name, model) in fitted.items():
        joblib.dump(dict(estimator=model, config=name, feature_names=FAMILIES[name][0], window_s=seconds,
                         threshold=0. if FAMILIES[name][1] == 'svm' else .5,
                         status=report['status']), output / f'candidate_{seconds}s_offline.joblib')
    print(summary.groupby('window_s').head(8).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'previous_3s_model', 'fixture', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--shortlist-only', action='store_true', help='Check three 3 s candidates in an existing output folder')
    args = parser.parse_args()
    benchmark = run_shortlist if args.shortlist_only else run
    benchmark(args.pilot, args.previous_3s_model, args.fixture, args.output)
