"""Score frozen N16 and Ahn–Lee blink detectors against reviewed BLINK_03 video."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from compare_blink_detectors import FS, peaks
from evaluate_public_mindwave import predict_shape
from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_event_pairs, match_events


CAL_END = 80.
VIDEO_END = 240.33
TOLERANCE = .20


def between(times, start, stop):
    return times[(times >= start) & (times < stop)]


def run(raw_file, annotations, model_file, output):
    raw = pd.read_csv(raw_file)
    signal = raw.Electrode.to_numpy(float)
    clock = raw['Time:512Hz'].to_numpy(float)
    if (not len(clock) or not np.isfinite(signal).all() or not np.isfinite(clock).all()
            or not np.allclose(np.diff(clock), 1/FS, rtol=0, atol=1e-9)):
        raise ValueError('Expected finite EEG on a nominal 512 Hz grid')
    device = raw.loc[raw['Event Id'].eq(33025), 'Event Date'].to_numpy(float) - clock[0]
    reference = pd.read_csv(annotations).timestamp_ms.to_numpy(float)/1000
    if not np.isfinite(reference).all() or (np.diff(reference) <= 0).any():
        raise ValueError('Video annotations must be finite and in strictly increasing order')
    frozen = joblib.load(model_file)
    duration = len(signal)/FS
    default_full = np.array([event.time_seconds for event in detect_blinks(signal, FS)])
    candidate_full = predict_shape(frozen['estimator'], peaks(signal, 12), 0.)

    # Approximate initial offset comes from the early device/video sequence.
    # Neither the late video nor the Ahn–Lee predictions participate in this fit.
    early_reference = between(reference, 0., CAL_END)
    maps, anchors = {}, {}
    for name, times in [('early_device', device), ('early_raw', default_full)]:
        pairs = match_event_pairs(between(times+1.4, 0., CAL_END), early_reference, .40)
        if len(pairs) < 4:
            raise ValueError(f'{name}: insufficient early anchors')
        a, b = np.array(pairs).T
        a -= 1.4
        scale, offset = np.polyfit(a, b, 1)
        maps[name] = np.poly1d((scale, offset))
        anchors[name] = [dict(eeg_s=x, video_s=y, residual_ms=(x*scale+offset-y)*1000)
                         for x, y in zip(a, b)]
    stop = min(VIDEO_END, min(f(duration) for f in maps.values())-.25)

    # Descriptive only: this fit reuses reference events from the scoring area.
    # A broad device-to-video association follows the observed event sequence.
    pairs = match_event_pairs(device+2.5, reference, 2.2)
    a, b = np.array(pairs).T
    a -= 2.5
    anchors['all_device_diagnostic'] = [dict(eeg_s=x, video_s=y, video_minus_eeg_s=y-x) for x, y in zip(a,b)]
    mapping = interp1d(a, b, kind='linear', fill_value='extrapolate', assume_sorted=True)
    # Compare all methods on the same video interval, inside the device anchors.
    stop = min(stop, float(b[-1])-TOLERANCE)
    evaluation = {mode: (fit, CAL_END, stop) for mode, fit in maps.items()}
    evaluation['all_device_diagnostic'] = (mapping, CAL_END, stop)

    rows, matched = [], []
    for mode, (mapping, start, stop) in evaluation.items():
        refs = between(reference, start, stop)
        for name, times in [('default', default_full), ('ahn_lee', candidate_full)]:
            pred = between(mapping(times), start, stop)
            rows.append(dict(alignment=mode, detector=name, start_video_s=start, stop_video_s=stop,
                             video_references=len(refs), **match_events(pred, refs, TOLERANCE)))
            pairs = match_event_pairs(pred, refs, TOLERANCE)
            used_pred = {p for p, _ in pairs}
            used_ref = {r for _, r in pairs}
            key = dict(alignment=mode, detector=name)
            matched.extend(dict(**key, status='matched', prediction_video_s=p, reference_video_s=r,
                                error_ms=(p-r)*1000) for p,r in pairs)
            matched.extend(dict(**key, status='prediction_only', prediction_video_s=p) for p in pred if p not in used_pred)
            matched.extend(dict(**key, status='reference_only', reference_video_s=r) for r in refs if r not in used_ref)

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    pd.DataFrame(matched).to_csv(output/'event_matches.csv', index=False)
    report = dict(scope='Exploratory. No independent synchronization markers. All-device diagnostic reuses test video events.',
                  samples=len(signal), nominal_duration_s=duration, video_duration_s=VIDEO_END,
                  confirmed_video_blinks=len(reference), device_events=len(device),
                  frozen_default_full=len(default_full), ahn_lee_full=len(candidate_full),
                  threshold_tuning=False, calibration_ends_video_s=CAL_END, tolerance_ms=200,
                  anchors=anchors, scores=rows,
                  model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest())
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(pd.DataFrame(rows)[['alignment','detector','video_references','true_positive','false_positive','false_negative','f1']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw_file', 'annotations', 'model_file', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.raw_file, args.annotations, args.model_file, args.output)
