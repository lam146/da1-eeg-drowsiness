"""Compare 3 s N16 classifiers and alarm policies with a shared streaming decision rule."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_pilot_ml_models import FEATURES, KEY, fit_model
from evaluate_python_alert_latency import FS, feature_vector, prepare
from n16_eeg.alarm import AlarmPolicy
from n16_eeg.model import LogisticModel
from n16_eeg.simulate import generate_session


STEP = FS // 4
WINDOW = 3 * FS
POLICIES = {
    'current': {},
    'release_2': {'release_consecutive': 2},
    'release_3': {'release_consecutive': 3},
    'release_5': {'release_consecutive': 5},
    'hysteresis': {'release_threshold': .45},
    'hysteresis_release_2': {'release_threshold': .45, 'release_consecutive': 2},
    'vote_3_of_4': {'confirmation_window': 4},
    'confirm_4': {'consecutive': 4},
    'threshold_55': {'threshold': .55},
    'threshold_55_release_2': {'threshold': .55, 'release_consecutive': 2},
    'threshold_60': {'threshold': .60},
    'threshold_60_release_2': {'threshold': .60, 'release_consecutive': 2},
}
POLICIES.update({f'threshold_{cut}_release_{count}': {'threshold': cut/1000, 'release_consecutive': count}
                 for cut in (505, 510, 515, 520, 525, 530, 540) for count in (1, 2, 3)})
MODELS = ('combined', 'spectral', 'logit_blend')


def train(frame, config):
    pipeline = fit_model(frame, config)
    scaler, logistic = (step for _, step in pipeline.steps)
    return LogisticModel(list(pipeline.feature_names_in_), scaler.center_.tolist(), scaler.scale_.tolist(),
                         logistic.coef_[0].tolist(), float(logistic.intercept_[0]))


def train_models(frame):
    combined = train(frame, 'logistic_combined')
    spectral = train(frame, 'logistic_spectral')
    # The average of two logistic logits is another affine logit. Fold both
    # scalers into the coefficients to export the blend as a single JSON model.
    coefficients = dict.fromkeys(FEATURES, 0.)
    bias = 0.
    for model in (combined, spectral):
        weight = np.asarray(model.weights)/model.scale
        for column, value in zip(model.feature_names, weight):
            coefficients[column] += value/2
        bias += (model.bias - weight @ model.mean)/2
    blend = LogisticModel(FEATURES, [0.]*len(FEATURES), [1.]*len(FEATURES),
                          list(coefficients.values()), float(bias))
    return dict(combined=combined, spectral=spectral, logit_blend=blend)


def risk_series(raw, models, start):
    positions = np.arange(start, len(raw)+1, STEP)
    values = pd.DataFrame([feature_vector(raw[stop-WINDOW:stop]) for stop in positions], columns=FEATURES)
    return positions, {name: model.predict_proba(values[model.feature_names].to_numpy()) for name, model in models.items()}


def trace(risk, options, poor=None):
    policy = AlarmPolicy(**options)
    if poor is None:
        return np.array([policy.update(value) for value in risk])
    return np.array([policy.update(None if bad else value) for value, bad in zip(risk, poor, strict=True)])


def simulate_cases(models, seeds, phases):
    cases = []
    for seed in seeds:
        for phase in phases:
            raw = generate_session(alert_seconds=12+phase, drowsy_seconds=10, seed=seed)
            onset = int(raw.loc[raw.event.eq('STATE_DROWSY'), 'timestamp_ms'].iloc[0])
            # Append eight seconds of a second alert signal to exercise release.
            returned = generate_session(alert_seconds=10, drowsy_seconds=5, seed=seed+1)
            values = np.r_[raw.raw_eeg.to_numpy(), returned.raw_eeg.to_numpy()[:8*FS]]
            positions, risks = risk_series(values, models, WINDOW)
            millis = np.rint((positions-1)*1000/FS).astype(int)
            recovery = round(len(raw)*1000/FS)
            cases.append((onset, recovery, millis, risks))
    return cases


def score_synthetic(cases, model, options):
    results = []
    for onset, recovery, millis, risks in cases:
        alarm = trace(risks[model], options)
        starts = alarm & ~np.r_[False, alarm[:-1]]
        delay = millis[starts & (millis >= onset) & (millis < recovery)] - onset
        last = millis[alarm & (millis >= recovery)] - recovery
        results.append(dict(delay_ms=int(delay[0]) if len(delay) else None,
                            pre_onset_alarm=bool(alarm[millis < onset].any()),
                            last_alarm_after_recovery_ms=int(last[-1]) if len(last) else 0))
    return results


def summarize_pilot(rows):
    rates = rows.groupby(['label', 'person']).alarm_fraction.mean().groupby('label').mean()
    false = rows.loc[rows.label.eq(0)]
    return dict(pilot_awake_alarm_fraction=float(rates.loc[0]), pilot_drowsy_alarm_fraction=float(rates.loc[1]),
                pilot_false_episodes_per_min=float(false.alarm_episodes.sum()/false.scored_minutes.sum()))


def run(pilot, prior_path, output):
    meta, matrices, fragments = prepare(pilot, include_later_fragment=True)
    frame = pd.concat([meta[KEY[:2]+['label']], pd.DataFrame(matrices[3], columns=FEATURES)], axis=1)
    final = train_models(frame)
    prior = LogisticModel.load(prior_path)
    np.testing.assert_allclose(final['combined'].predict_proba(matrices[3]), prior.predict_proba(matrices[3]), atol=1e-9)
    held_out = {person: train_models(frame.loc[frame.subject_id.ne(person)]) for person in frame.subject_id.unique()}
    development = simulate_cases(final, (17, 23, 41, 99), (0, .125, .375, .625, .875))
    audit = simulate_cases(final, (257, 277, 293, 311, 337, 353), (.05, .29, .56, .87))
    pilot_rows, records = [], []
    seen_fragments = set()
    for (person, session), label, raw in fragments:
        key = (person, session, label)
        split = 'later' if key in seen_fragments else 'first'
        seen_fragments.add(key)
        _, risks = risk_series(raw, held_out[person], 4*FS)
        for model in MODELS:
            for name, options in POLICIES.items():
                alarm = trace(risks[model], options)
                starts = alarm & ~np.r_[False, alarm[:-1]]
                pilot_rows.append(dict(model=model, policy=name, person=person, session=session, label=label, split=split,
                                       alarm_fraction=float(alarm.mean()), alarm_episodes=int(starts.sum()),
                                       scored_minutes=(len(raw)/FS-4)/60))
    pilot_results = pd.DataFrame(pilot_rows)
    for (model, name), rows in pilot_results.loc[pilot_results.split.eq('first')].groupby(['model', 'policy']):
        cases = score_synthetic(development, model, POLICIES[name])
        delays = [r['delay_ms'] for r in cases if r['delay_ms'] is not None]
        records.append(dict(model=model, policy=name, dev_under_3s=sum(d<3000 for d in delays), dev_cases=len(cases),
                            dev_max_ms=max(delays, default=None), dev_premature=sum(r['pre_onset_alarm'] for r in cases),
                            **summarize_pilot(rows)))
    baseline = next(row for row in records if (row['model'], row['policy']) == ('combined', 'current'))
    eligible = [r for r in records if r['dev_under_3s'] == r['dev_cases'] and r['dev_premature'] == 0
                and r['pilot_awake_alarm_fraction'] <= baseline['pilot_awake_alarm_fraction']+.01
                and r['pilot_drowsy_alarm_fraction'] >= baseline['pilot_drowsy_alarm_fraction']-.02]
    selected = min(eligible, key=lambda r: (r['pilot_false_episodes_per_min'], r['dev_max_ms']), default=baseline)
    selections = {'baseline': baseline, 'selected': selected}
    check = {key: score_synthetic(audit, row['model'], POLICIES[row['policy']]) for key, row in selections.items()}
    later = pilot_results.loc[pilot_results.split.eq('later')]
    later_check = {key: summarize_pilot(later.loc[later.model.eq(row['model']) & later.policy.eq(row['policy'])])
                   for key, row in selections.items()}
    # Explicitly test that a quality dropout clears the alert immediately and
    # any new alert must satisfy the confirmation policy again.
    settings = POLICIES[selected['policy']]
    required = settings.get('consecutive', 3)
    quality = trace([.99]*(required+1)+[.99]+[.99]*required, settings,
                    [False]*(required+1)+[True]+[False]*required)
    assert quality[required] and not quality[required+1:2*required+1].any() and quality[-1]
    report = dict(scope='Pilot development and synthetic replay. Physical audio latency not measured.',
                  trials=len(records), pilot_segments=len(fragments), selection=selections, later_pilot=later_check,
                  selection_policy='All development onsets under 3 s without early alarm, at most +1 percentage point awake alarm time and -2 percentage points drowsy alarm time, then fewest false alarm episodes per minute.',
                  selected_policy_options=settings, synthetic_audit=check, quality_reset_verified=True)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).sort_values('pilot_false_episodes_per_min').to_csv(output/'comparison.csv', index=False)
    pilot_results.to_csv(output/'pilot_segments.csv', index=False)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    model = final[selected['model']]
    model.threshold = settings.get('threshold', .5)
    model.save(output/'selected_3s_PROVISIONAL.json')
    print(json.dumps(selections, indent=2))
    print('Later fragments:', later_check)
    for name, cases in check.items():
        delays = [r['delay_ms'] for r in cases if r['delay_ms'] is not None]
        print(name, sum(d<3000 for d in delays), '/', len(cases), 'max_ms', max(delays, default=None),
              'early', sum(r['pre_onset_alarm'] for r in cases),
              'max_recovery_ms', max(r['last_alarm_after_recovery_ms'] for r in cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'prior_model', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.pilot, args.prior_model, args.output)
