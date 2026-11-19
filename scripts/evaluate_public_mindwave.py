"""Check EEG blink detectors against task cues and artifacts in Ahn–Lee MW2 EDFs."""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from compare_blink_detectors import BASELINE, FS, INPUTS, MAD, MORPH, peaks, read_session, suppress, transformed
from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_events


METHODS = [BASELINE, 'mad_4_0.35_0.08', 'mad_8_0.5_0.08', 'mad_10_0.5_0.08',
           'mad_6_0.5_0.1', 'peak_8_5_0.5_0.055', 'peak_8_8_0.5_0.055', 'peak_8_12_0.5_0.08',
           'peak_12_5_0.5_0.055', 'peak_12_8_0.5_0.055', 'peak_12_12_0.5_0.055', 'peak_12_8_0.35_0.08']
FEATURES = ['z', 'prominence', 'width', 'symmetry', 'sign']
NEGATIVE_TASKS = ('BT', 'MVC')


def read_recording(path):
    import edfio

    subject, task = re.fullmatch(r'(sub-\d+)_task-(EB|BT|MVC|MVO)_acq-MW2_eeg\.edf', path.name).groups()
    edf = edfio.read_edf(path)
    signal, = edf.signals
    if signal.sampling_frequency != FS:
        raise ValueError(f'Expected 512 Hz: {path.name}')
    annotations = [(event.onset, event.text.rsplit('_', 1)[-1]) for event in edf.annotations]
    start, = [t for t, code in annotations if code == '01']
    stop, = [t for t, code in annotations if code == '03']
    cues = np.array([t for t, code in annotations if code == '02' and start <= t < stop])
    if len(cues) < 10 or not 0 <= start < stop <= edf.duration:
        raise ValueError(f'Unexpected task markers: {path.name}')
    left, right = round(start*FS), round(stop*FS)
    x = signal.data[left:right]
    candidates = {hi: peaks(x, hi) for hi in (8, 12)}
    result = {}
    for name in METHODS:
        if name in MAD:
            z, gap, width = MAD[name]
            result[name] = np.array([event.time_seconds+start for event in detect_blinks(
                x, FS, threshold_mad=z, refractory_s=gap, min_width_s=width)])
        else:
            hi, z, gap, width = MORPH[name]
            p = candidates[hi]
            p = p.loc[p.z.ge(z) & p.width.ge(width) & p.prominence.ge(z*.55)]
            result[name] = suppress(p['sample'].to_numpy()/FS+start, p.z.to_numpy(), gap)
    return dict(subject=subject, task=task, cues=cues, start=start, stop=stop,
                peaks=candidates[12], predictions=result)


def cue_counts(times, cues):
    return ((times[:, None] >= cues[None, :]-.25) & (times[:, None] < cues[None, :]+.75)).sum(axis=0)


def train_shape(records):
    train = []
    for record in records:
        p = record['peaks']
        times = p['sample'].to_numpy()/FS + record['start']
        if record['task'] == 'EB':
            selected = []
            for cue in record['cues']:
                near = p.loc[(times >= cue-.25) & (times < cue+.75)]
                if len(near):
                    selected.append(near.z.idxmax())
            train.append(p.loc[selected, FEATURES].assign(label=1))
        elif record['task'] in NEGATIVE_TASKS:
            train.append(p[FEATURES].assign(label=0))
    frame = pd.concat(train, ignore_index=True)
    model = make_pipeline(RobustScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=2000))
    return model.fit(frame[FEATURES], frame.label), frame.label.value_counts().to_dict()


def results_for(record, model):
    ml = predict_shape(model, record['peaks'], record['start'])
    predictions = dict(record['predictions'], **hybrids(record['predictions'][BASELINE], ml))
    for name, times in predictions.items():
        counts = cue_counts(times, record['cues'])
        yield dict(subject=record['subject'], task=record['task'], model=name,
                   cues=len(counts), cues_hit=int((counts > 0).sum()), cues_with_multiple=int((counts > 1).sum()),
                   detections=len(times), minutes=(record['stop']-record['start'])/60)


def predict_shape(model, candidates, origin):
    probabilities = model.predict_proba(candidates[FEATURES])[:, 1] if len(candidates) else np.array([])
    active = probabilities >= .5
    return suppress(candidates['sample'].to_numpy()[active]/FS+origin, probabilities[active], .5)


def hybrids(default, ml):
    distance = np.abs(default[:, None] - ml[None, :])
    consensus = default[(distance <= .15).any(axis=1)]
    complement = np.sort(np.r_[default, ml[~(distance < .35).any(axis=0)]])
    return dict(cue_trained_logistic=ml, consensus=consensus, complement=complement)


def transfer_to_n16(model):
    rows = []
    for name in INPUTS:
        session = read_session(name)
        x, origin = session['x'], session['origin']
        default = np.array([e.time_seconds+origin for e in detect_blinks(x, FS)])
        detectors = dict(default=default, **hybrids(default, predict_shape(model, peaks(x, 12), origin)))
        for detector, times in detectors.items():
            for alignment in session['alignments']:
                score = match_events(transformed(times, session, alignment), session['reference'], .20)
                rows.append(dict(session=name, detector=detector, alignment=alignment, **score))
    return rows


def summarize(frame):
    group = frame.groupby('task')[['cues', 'cues_hit', 'cues_with_multiple', 'detections', 'minutes']].sum()
    positive = float(group.loc['EB', 'cues_hit']/group.loc['EB', 'cues'])
    negative = group.loc[list(NEGATIVE_TASKS)]
    artifact = float(negative.cues_hit.sum()/negative.cues.sum())
    return dict(blink_cue_hit_rate=positive, artifact_cue_activation=artifact,
                proxy_balanced_score=(positive+1-artifact)/2,
                artifact_detections_per_min=float(negative.detections.sum()/negative.minutes.sum()),
                task_totals=group.to_dict(orient='index'))


def run(dataset, output):
    manifest = json.loads((dataset/'source_manifest.json').read_text(encoding='utf-8'))
    records = [read_recording(dataset/item['file']) for item in manifest['files']]
    # Predeclared deterministic subject split, independent of cue responses.
    train = [row for row in records if int(row['subject'].split('-')[-1]) % 3 != 0]
    test = [row for row in records if int(row['subject'].split('-')[-1]) % 3 == 0]
    model, weak_label_counts = train_shape(train)
    dev = pd.DataFrame(row for rec in train for row in results_for(rec, model))
    validation = pd.DataFrame(row for rec in test for row in results_for(rec, model))
    ranking = {name: summarize(rows) for name, rows in dev.groupby('model') if name in METHODS}
    choice = max(ranking, key=lambda name: (ranking[name]['proxy_balanced_score'], ranking[name]['blink_cue_hit_rate']))
    scores = {name: summarize(rows) for name, rows in validation.groupby('model')}
    report = dict(dataset=manifest['source'], recordings=len(records), configurations=len(METHODS)+3,
                  development_people=sorted(dev.subject.unique()), held_out_people=sorted(validation.subject.unique()),
                  cue_window_s=[-.25, .75], negative_tasks=NEGATIVE_TASKS, selected_fixed=choice,
                  weak_label_counts=weak_label_counts, development_fixed=ranking, held_out=scores,
                  scope='Prompt-level sensitivity and motion response, not video-ground-truth blink accuracy. MVO eyes-open motion can contain true blinks.')
    transfer = transfer_to_n16(model)
    report['n16_transfer'] = transfer
    output.mkdir(parents=True, exist_ok=False)
    pd.concat([dev.assign(split='development'), validation.assign(split='held_out')]).to_csv(output/'file_scores.csv', index=False)
    pd.DataFrame(transfer).to_csv(output/'n16_transfer.csv', index=False)
    joblib.dump(dict(estimator=model, features=FEATURES, source=manifest['source'],
                     scope='Offline weak cue labels; no software runtime validation'), output/'public_blink_candidate.joblib')
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name in dict.fromkeys((BASELINE, 'peak_12_8_0.5_0.055', choice, 'cue_trained_logistic', 'consensus', 'complement')):
        print(name, {k: v for k, v in scores[name].items() if k != 'task_totals'})
    print(pd.DataFrame(transfer)[['session', 'detector', 'alignment', 'f1']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.dataset, args.output)
