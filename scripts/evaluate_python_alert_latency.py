"""Compare Python alarm schedules on synthetic onsets and held-out pilot recordings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_pilot_model import FEATURES
from improve_pilot_ml import KEY
from n16_eeg.blink import blink_window_features
from n16_eeg.live import LiveDetector
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.model import LogisticModel
from n16_eeg.signal_features import bandpower_features
from n16_eeg.simulate import generate_session
from tune_runtime_pilot import fit


FS = 512
CADENCE = 128  # 0.25 s base grid, shared across comparisons.
SCHEDULES = [(4, 1., 2), (4, .5, 2), (4, .25, 2), (4, .25, 1),
             (3, .5, 2), (3, .25, 2), (3, .25, 3),
             (2, .5, 2), (2, .25, 2), (2, .25, 3)]


def feature_vector(samples):
    features = bandpower_features(samples, FS) | blink_window_features(samples, FS)
    return [features[name] for name in FEATURES]


def prepare(pilot, include_later_fragment=False):
    frame = pd.read_csv(pilot / 'all_subjects_features_PROVISIONAL.csv').sort_values(KEY).reset_index(drop=True)
    if set(frame.subject_id) != {'S01', 'S02'} or frame.duplicated(KEY).any() or set(frame.fs_hz) != {FS}:
        raise ValueError('Expected the reviewed S01/S02 pilot windows')
    manifest = json.loads((pilot / 'experiment_PROVISIONAL.json').read_text(encoding='utf-8'))
    recordings = {(item['subject_id'], item['session_id']): item for item in manifest['sessions']}
    data = {seconds: np.empty((len(frame), len(FEATURES))) for seconds in (2, 3, 4)}
    fragments = []
    for key, rows in frame.groupby(KEY[:2]):
        source = recordings[key]
        path = Path(source['source'])
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != source['source_sha256']:
                raise ValueError(f'Raw recording changed: {path.name}')
        raw = pd.read_csv(path, usecols=['Electrode']).Electrode.to_numpy(float)
        stops = np.rint(rows.window_end_ms.to_numpy() * FS / 1000).astype(int)
        for seconds, values in data.items():
            values[rows.index] = [feature_vector(raw[end - seconds * FS:end]) for end in stops]

        # Choose disjoint 60 s fragments without inspecting predictions.
        for label, labeled in rows.groupby('label'):
            starts_ms = labeled.window_start_ms.to_numpy()
            breaks = np.flatnonzero(np.diff(starts_ms) != 4000) + 1
            eligible = [block for block in np.split(starts_ms, breaks) if len(block) >= 15]
            if not eligible:
                continue
            first = eligible[0][:15]
            last = eligible[-1][-15:]
            blocks = [first]
            if include_later_fragment and last[0] >= first[-1] + 4000:
                blocks.append(last)
            for block in blocks:
                start = round(block[0] * FS / 1000)
                end = round((block[-1] + 4000) * FS / 1000)
                fragments.append((key, int(label), raw[start:end]))
    return frame, data, fragments


def score_fragments(frame, data, fragments):
    people = sorted(frame.subject_id.unique())
    output = []
    for seconds, values in data.items():
        training = pd.concat([frame[KEY[:2] + ['label']], pd.DataFrame(values, columns=FEATURES)], axis=1)
        models = {person: fit(training.loc[training.subject_id.ne(person)]) for person in people}
        for (person, session), label, raw in fragments:
            # All policies are scored after a common 4 s warm-up, with identical source segments.
            features = [feature_vector(raw[end - seconds * FS:end])
                        for end in range(4 * FS, len(raw) + 1, CADENCE)]
            risks = models[person].predict_proba(np.asarray(features))
            for window, step, count in SCHEDULES:
                if window != seconds:
                    continue
                decisions = risks[::round(step * FS / CADENCE)] >= .5
                alarms = np.convolve(decisions.astype(int), np.ones(count, dtype=int), mode='full')[:len(decisions)] == count
                starts = alarms & ~np.r_[False, alarms[:-1]]
                output.append(dict(window_s=seconds, step_s=step, consecutive=count, person=person,
                                   session=session, label=label, alarm_fraction=float(alarms.mean()),
                                   alarm_episodes=int(starts.sum()), scoring_minutes=(len(raw) / FS - 4) / 60))
    return pd.DataFrame(output)


def onset_test(model, config, raw):
    window, step, count = config
    marker = raw.loc[raw.event.eq('STATE_DROWSY'), 'timestamp_ms']
    if len(marker) != 1:
        raise ValueError('Synthetic timing cases need one onset marker')
    onset = int(marker.iloc[0])
    detector = LiveDetector(model, window_s=window, step_s=step, consecutive=count)
    first_risk, poor_alarms = None, 0
    times, alarms = [], []
    for row in raw.itertuples(index=False):
        decision = detector.add_sample(row.raw_eeg, row.poor_signal)
        if decision is None:
            continue
        times.append(row.timestamp_ms)
        alarms.append(bool(decision['alarm']))
        if decision['state'] == 'TIN_HIEU_KEM':
            poor_alarms += int(decision['alarm'])
        if row.timestamp_ms >= onset and decision['state'] in ('NGUY_CO', 'CANH_BAO') and first_risk is None:
            first_risk = int(row.timestamp_ms - onset)
    return dict(first_risk_ms=first_risk, poor_signal_alarms=poor_alarms,
                **alarm_onset_metrics(times, alarms, onset))


def run(pilot, fixture, prior, output):
    frame, data, fragments = prepare(pilot)
    real = score_fragments(frame, data, fragments)
    models = {seconds: fit(pd.concat([frame[KEY[:2] + ['label']], pd.DataFrame(values, columns=FEATURES)], axis=1))
              for seconds, values in data.items()}
    existing = LogisticModel.load(prior)
    np.testing.assert_allclose(existing.predict_proba(data[4]), models[4].predict_proba(data[4]), atol=1e-8)
    development = [pd.read_csv(fixture)] + [generate_session(alert_seconds=12 + phase, drowsy_seconds=12, seed=seed)
                                            for seed in (17, 23, 41, 99) for phase in (0, .125, .375, .625, .875)]
    summaries, synthetic_rows = [], []
    for config, subset in real.groupby(['window_s', 'step_s', 'consecutive'], sort=False):
        results = [onset_test(models[config[0]], config, raw) for raw in development]
        synthetic_rows.extend(dict(window_s=config[0], step_s=config[1], consecutive=config[2], case=i, **result)
                              for i, result in enumerate(results))
        alarm_times = [result['onset_to_alarm_ms'] for result in results if result['onset_to_alarm_ms'] is not None]
        rates = subset.groupby(['person', 'label']).alarm_fraction.mean().groupby('label').mean()
        false = subset.loc[subset.label.eq(0)]
        summaries.append(dict(window_s=config[0], step_s=config[1], consecutive=config[2],
                              demo_ms=results[0]['onset_to_alarm_ms'], max_onset_ms=max(alarm_times, default=None),
                              under_3s=sum(ms < 3000 for ms in alarm_times), cases=len(results),
                              pre_onset_alarms=sum(row['pre_onset_alarm'] for row in results),
                              pilot_alert_false_fraction=float(rates.loc[0]), pilot_drowsy_alarm_fraction=float(rates.loc[1]),
                              pilot_alarm_balanced_rate=float((1 - rates.loc[0] + rates.loc[1]) / 2),
                              pilot_false_episodes_per_min=float(false.alarm_episodes.sum() / false.scoring_minutes.sum())))
    baseline = next(row for row in summaries if (row['window_s'], row['step_s'], row['consecutive']) == (4, 1, 2))
    # Reserve some of the three-second budget for transport and audio. The pilot
    # check limits time spent alarming on alert fragments, rather than episode count.
    eligible = [row for row in summaries if row['under_3s'] == row['cases'] and row['pre_onset_alarms'] == 0
                and row['max_onset_ms'] <= 2750
                and row['pilot_alert_false_fraction'] <= baseline['pilot_alert_false_fraction']]
    selected = max(eligible, key=lambda row: (row['pilot_alarm_balanced_rate'], -row['max_onset_ms']), default=None)
    report = dict(scope='Development only. Synthetic transitions and provisional NeuroSky labels; no speaker measurement.',
                  pilot_segments=len(fragments), training_windows=len(frame), comparison=summaries, selected=selected,
                  selection_policy='All development onsets <=2750 ms, no pre-onset alarm, pilot alert alarm fraction no higher than the old schedule, then highest pilot balanced alarm rate.')
    output.mkdir(parents=True, exist_ok=False)
    real.to_csv(output / 'pilot_segments_PROVISIONAL.csv', index=False)
    pd.DataFrame(synthetic_rows).to_csv(output / 'synthetic_timing.csv', index=False)
    if selected:
        config = tuple(selected[k] for k in ('window_s', 'step_s', 'consecutive'))
        model_path = output / f'python_model_{config[0]}s_PROVISIONAL.json'
        models[config[0]].save(model_path)
        report['model_file'] = model_path.name
        checks = [generate_session(alert_seconds=12 + phase, drowsy_seconds=12, seed=seed)
                  for seed in (51, 71, 103, 127) for phase in (.05, .23, .47, .71, .93)]
        report['fresh_synthetic'] = [onset_test(models[config[0]], config, raw) for raw in checks]
        poor = checks[0].copy()
        poor.loc[poor.timestamp_ms.between(8000, 18000), 'poor_signal'] = 200
        report['poor_signal_test'] = onset_test(models[config[0]], config, poor)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Selected:', selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'fixture', 'prior_model', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.pilot, args.fixture, args.prior_model, args.output)
