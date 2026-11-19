"""Compare blink detectors across the two reviewed S02 sessions."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_event_pairs, match_events
from evaluate_s02_blink_review import in_range


FS = 512
TOLERANCE = .20
ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'mad_6_0.5_0.08'
MAD = {f'mad_{z}_{gap}_{width}': (z, gap, width)
       for z, gap, width in product((4, 6, 8, 10), (.25, .35, .5), (.05, .08, .10))}
MORPH = {f'peak_{hi}_{z}_{gap}_{width}': (hi, z, gap, width)
         for hi, z, gap, width in product((8, 12), (5, 8, 12), (.25, .35, .5), (.055, .08))}
INPUTS = {
    'BLINK_01': ('S02_06112026_Blink.csv', 'blink_s02_20261106', 'video_evaluation_v2/video_blink_metrics.json'),
    'BLINK_02': ('S02_07112026_Blink02.csv', 'blink02_s02_20261107', 'video_blink_metrics.json'),
}


def read_session(name):
    file, directory, metrics = INPUTS[name]
    raw = pd.read_csv(ROOT / 'raw/Raw data/S02' / file)
    stored = json.loads((ROOT / 'output' / directory / metrics).read_text(encoding='utf-8'))
    ref = pd.read_csv(ROOT / 'output' / directory / 'video_blinks_confirmed.csv').timestamp_ms.to_numpy(float) / 1000
    x = raw.Electrode.to_numpy(float)
    clock = raw['Time:512Hz'].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(clock).all() or not np.allclose(np.diff(clock), 1/FS, atol=1e-7, rtol=0):
        raise ValueError('Expected finite EEG on a 512 Hz time grid')
    limits = tuple(stored['holdout_video_s'])
    device_fit = stored['video_time_formula']
    scale, offset = device_fit['scale'], device_fit['offset_s']
    raw_fit = stored.get('alternative_raw_anchor_formula')
    if raw_fit is None:
        # For BLINK_01, re-fit on the existing early interval only, starting
        # from the device alignment and pairing the frozen default raw peaks.
        cal = tuple(stored['calibration_video_s'])
        eeg = np.array([event.time_seconds for event in detect_blinks(x, FS)])
        pairs = match_event_pairs(in_range(eeg*scale+offset, (cal[0]-.35, cal[1]+.35)), in_range(ref, cal), .35)
        if len(pairs) < 4:
            raise ValueError('Insufficient early anchors')
        estimated, video = np.asarray(pairs).T
        a, b = np.polyfit((estimated-offset)/scale, video, 1)
        raw_fit = dict(scale=float(a), offset_s=float(b), early_anchors=len(pairs))
    alignments = dict(device_anchors=device_fit, raw_anchors=raw_fit)
    starts, ends = zip(*[((limits[0]-p['offset_s'])/p['scale'], (limits[1]-p['offset_s'])/p['scale'])
                        for p in alignments.values()])
    start = max(0, int(np.floor(min(starts)*FS)))
    stop = min(len(x), int(np.ceil(max(ends)*FS)))
    device = raw.loc[pd.to_numeric(raw['Event Id'], errors='coerce').eq(33025), 'Event Date'].to_numpy(float) - clock[0]
    return dict(x=x[start:stop], origin=start/FS, reference=in_range(ref, limits), alignments=alignments,
                limits=limits, device=device, calibration_s=stored['calibration_video_s'])


def peaks(x, high):
    filtered = sosfiltfilt(butter(2, (.6, high), btype='bandpass', fs=FS, output='sos'), x)
    filtered -= np.median(filtered)
    scale = max(1.4826*np.median(np.abs(filtered)), np.finfo(float).eps)
    magnitude = np.abs(filtered) / scale
    indices, prop = find_peaks(magnitude, height=3.5, prominence=2.5, width=(.04*FS, .5*FS), wlen=FS)
    left = indices - prop['left_ips']
    right = prop['right_ips'] - indices
    return pd.DataFrame(dict(sample=indices, z=prop['peak_heights'], prominence=prop['prominences'],
                             width=prop['widths']/FS, symmetry=np.minimum(left, right)/np.maximum(left, right),
                             sign=np.sign(filtered[indices])))


def suppress(times, strength, gap):
    kept = []
    for index in np.argsort(-np.asarray(strength), kind='stable'):
        time = times[index]
        if all(abs(time-other) >= gap for other in kept):
            kept.append(time)
    return np.array(sorted(kept))


def transformed(times, session, alignment, shift=0):
    fit = session['alignments'][alignment]
    return in_range(times*fit['scale'] + fit['offset_s'] + shift, session['limits'])


def fixed_predictions(session):
    x, origin = session['x'], session['origin']
    result = {name: np.array([e.time_seconds+origin for e in detect_blinks(x, FS, threshold_mad=z,
                                                                        refractory_s=gap, min_width_s=width)])
              for name, (z, gap, width) in MAD.items()}
    extracted = {hi: peaks(x, hi) for hi in (8, 12)}
    for name, (hi, z, gap, width) in MORPH.items():
        p = extracted[hi]
        p = p.loc[p.z.ge(z) & p.width.ge(width) & p.prominence.ge(z*.55)]
        result[name] = suppress(p['sample'].to_numpy()/FS+origin, p.z.to_numpy(), gap)
    return result, extracted[12]


def transfer_logistic(training, test, train_peaks, test_peaks):
    cols = ['z', 'prominence', 'width', 'symmetry', 'sign']
    fit = training['alignments']['raw_anchors']
    times = (train_peaks['sample'].to_numpy()/FS+training['origin'])*fit['scale']+fit['offset_s']
    valid = (times >= training['limits'][0]) & (times < training['limits'][1])
    matched = {p for p, _ in match_event_pairs(times[valid], training['reference'], TOLERANCE)}
    y = np.isin(times[valid], list(matched))
    model = make_pipeline(RobustScaler(), LogisticRegression(C=.1, class_weight='balanced'))
    model.fit(train_peaks.loc[valid, cols], y)
    probability = model.predict_proba(test_peaks[cols])[:, 1]
    active = probability >= .5
    return suppress(test_peaks['sample'].to_numpy()[active]/FS+test['origin'], probability[active], .35), int(y.sum())


def run(output):
    sessions = {name: read_session(name) for name in INPUTS}
    computed = {name: fixed_predictions(s) for name, s in sessions.items()}
    rows, details, robustness, matches, selected = [], [], [], [], {}
    for name, s in sessions.items():
        for model, times in computed[name][0].items():
            for alignment in s['alignments']:
                rows.append(dict(session=name, detector=model, alignment=alignment,
                                 **match_events(transformed(times, s, alignment), s['reference'], TOLERANCE)))
    fixed = pd.DataFrame(rows)
    for held_out, s in sessions.items():
        train = next(name for name in sessions if name != held_out)
        stats = fixed.loc[fixed.session.eq(train)].groupby('detector').agg(
            mean_f1=('f1', 'mean'), worst_f1=('f1', 'min'), mean_precision=('precision', 'mean'))
        choice = stats.sort_values(['mean_f1', 'worst_f1', 'mean_precision'], ascending=False).index[0]
        selected[held_out] = dict(trained_on=train, fixed_choice=choice)
        predicted, positive = transfer_logistic(sessions[train], s, computed[train][1], computed[held_out][1])
        selected[held_out]['ml_training_positive_candidates'] = positive
        finalists = {'default': computed[held_out][0][BASELINE], 'selected_on_other_session': computed[held_out][0][choice],
                     'shape_logistic': predicted, 'device_log': s['device']}
        for model, times in finalists.items():
            for alignment in s['alignments']:
                aligned = transformed(times, s, alignment)
                summary = match_events(aligned, s['reference'], TOLERANCE)
                details.append(dict(session=held_out, detector=model, alignment=alignment, **summary))
                paired = match_event_pairs(aligned, s['reference'], TOLERANCE)
                used_predictions = {p for p, _ in paired}
                used_references = {r for _, r in paired}
                key = dict(session=held_out, detector=model, alignment=alignment)
                matches.extend(dict(**key, status='matched', prediction_video_s=p, reference_video_s=r,
                                    error_ms=(p-r)*1000) for p, r in paired)
                matches.extend(dict(**key, status='prediction_only', prediction_video_s=p)
                               for p in aligned if p not in used_predictions)
                matches.extend(dict(**key, status='reference_only', reference_video_s=r)
                               for r in s['reference'] if r not in used_references)
                for shift in (-.10, -.05, 0, .05, .10):
                    shifted = transformed(times, s, alignment, shift)
                    robustness.append(dict(session=held_out, detector=model, alignment=alignment, shift_ms=shift*1000,
                                           **match_events(shifted, s['reference'], TOLERANCE)))
    report = dict(scope='Exploratory offline replay, two already-inspected sessions from S02. Estimated alignment only.',
                  tolerance_ms=TOLERANCE*1000, fixed_configurations=len(MAD)+len(MORPH),
                  mad_parameters=MAD, peak_parameters=MORPH,
                  session={name: dict(references=len(s['reference']), evaluation_video_s=s['limits'],
                                      calibration_video_s=s['calibration_s'], alignment=s['alignments'])
                           for name, s in sessions.items()},
                  selected=selected, transferred_results=details)
    output.mkdir(parents=True, exist_ok=False)
    fixed.to_csv(output / 'all_fixed_scores.csv', index=False)
    pd.DataFrame(details).to_csv(output / 'transfer_comparison.csv', index=False)
    pd.DataFrame(matches).to_csv(output / 'event_matches.csv', index=False)
    pd.DataFrame(robustness).to_csv(output / 'sync_sensitivity.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(pd.DataFrame(details)[['session', 'detector', 'alignment', 'true_positive', 'false_positive', 'false_negative', 'f1']].to_string(index=False))
    print(selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
