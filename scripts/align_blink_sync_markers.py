"""Align video annotations from separately recorded EEG/video sync marks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_events


FS = 512


def align_markers(eeg, video, max_residual_ms):
    if (len(eeg) < 3 or eeg.event.duplicated().any() or video.event.duplicated().any()
            or set(eeg.event) != set(video.event)):
        raise ValueError('Need the same three or more unique sync IDs in EEG and video')
    markers = eeg.merge(video, on='event', validate='one_to_one').sort_values('video_timestamp_ms')
    x = markers.video_timestamp_ms.to_numpy(float)
    y = markers.signal_time_ms.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (np.diff(x) <= 0).any() or (np.diff(y) <= 0).any():
        raise ValueError('Sync marker times must be finite and strictly increasing in both files')
    scale = (y[-1]-y[0])/(x[-1]-x[0])
    offset = y[0]-scale*x[0]
    markers['residual_ms'] = y - (scale*x+offset)
    worst = float(markers.residual_ms.iloc[1:-1].abs().max())
    return markers, dict(scale=float(scale), offset_ms=float(offset), max_middle_residual_ms=worst,
                         middle_check_passed=bool(worst <= max_residual_ms))


def run(recording, video_markers, video_blinks, output, max_residual_ms, public_model=None):
    raw = pd.read_csv(recording)
    if (set(raw.fs_hz) != {FS} or not np.array_equal(raw.sample_index, np.arange(len(raw)))
            or not np.allclose(raw.signal_time_ms, raw.sample_index*1000/FS)
            or not np.isfinite(raw.raw_eeg.to_numpy(float)).all()):
        raise ValueError('Expected one continuous nominal 512 Hz acquisition CSV')
    eeg = raw.loc[raw.event.fillna('').str.fullmatch(r'SYNC_\d+'), ['event', 'signal_time_ms', 'sample_index']]
    markers, fit = align_markers(eeg, pd.read_csv(video_markers), max_residual_ms)
    summary_path = recording.with_suffix('.capture.json')
    capture = json.loads(summary_path.read_text(encoding='utf-8'))
    logged = pd.DataFrame(capture['sync_markers']).set_index('event')
    receipts = logged.reindex(markers.event)
    if not np.array_equal(receipts.next_sample_index, markers.sample_index):
        raise ValueError('Capture log does not match the samples marked in the EEG CSV')
    if {'requested_elapsed_ms', 'next_sample_elapsed_ms'} <= set(receipts.columns):
        delay = receipts.next_sample_elapsed_ms - receipts.requested_elapsed_ms
    else:
        delay = receipts.next_sample_receipt_ms - receipts.requested_timestamp_ms
    if not np.isfinite(delay).all():
        raise ValueError('Missing sync marker receipt times')
    receipt_ok = bool(delay.between(0, max_residual_ms).all())
    gaps = capture.get('raw_receipt_gaps')
    # A gap anywhere in the marker span can distort the global time mapping.
    gaps_in_span = None if gaps is None else [
        gap for gap in gaps
        if markers.sample_index.iloc[0] <= gap['sample_index_after_gap'] <= markers.sample_index.iloc[-1]
    ]
    blink = pd.read_csv(video_blinks).rename(columns={'timestamp_ms': 'video_timestamp_ms'})
    if not np.isfinite(blink.video_timestamp_ms.to_numpy(float)).all() or blink.video_timestamp_ms.duplicated().any():
        raise ValueError('Video blink annotations must contain unique, finite timestamps')
    blink['estimated_eeg_ms'] = blink.video_timestamp_ms*fit['scale'] + fit['offset_ms']
    left = int(markers.sample_index.iloc[0]+FS)
    right = int(markers.sample_index.iloc[-1]-FS)
    if right-left < FS:
        raise ValueError('Markers need at least three seconds of separation between first and last')
    mask = blink.estimated_eeg_ms.ge(left*1000/FS) & blink.estimated_eeg_ms.lt(right*1000/FS)
    blink['in_scored_interval'] = mask
    score = None
    candidate_score = None
    if fit['middle_check_passed'] and receipt_ok and not gaps_in_span:
        signal = raw.raw_eeg.iloc[left:right].to_numpy(float)
        reference = blink.loc[mask, 'estimated_eeg_ms'].to_numpy(float)/1000
        pred = [event.time_seconds + left/FS for event in detect_blinks(signal, FS)]
        score = match_events(pred, reference, .20)
        if public_model is not None:
            import joblib
            from evaluate_public_mindwave import FEATURES, peaks, predict_shape

            bundle = joblib.load(public_model)
            if bundle['features'] != FEATURES:
                raise ValueError('Unexpected public blink model feature order')
            candidate = predict_shape(bundle['estimator'], peaks(signal, 12), left/FS)
            candidate_score = match_events(candidate, reference, .20)
    output.mkdir(parents=True, exist_ok=False)
    markers['pc_marker_to_next_sample_ms'] = delay.to_numpy()
    markers.to_csv(output/'marker_check.csv', index=False)
    blink.to_csv(output/'video_blinks_aligned.csv', index=False)
    report = dict(fit=fit, receipt_check_passed=receipt_ok, raw_receipt_gaps_in_marker_span=gaps_in_span,
                  tolerance_ms=200, score=score,
                  public_model_score=candidate_score,
                  scope='Software alignment diagnostic. Constant audio and Bluetooth latency may remain.',
                  scored_interval_eeg_s=[left/FS, right/FS])
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'video_markers', 'video_blinks', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--max-residual-ms', type=float, default=50.)
    parser.add_argument('--public-model', type=Path, help='Locally generated public blink candidate joblib')
    args = parser.parse_args()
    if not np.isfinite(args.max_residual_ms) or args.max_residual_ms <= 0:
        parser.error('Expected a positive residual limit')
    run(args.recording, args.video_markers, args.video_blinks, args.output, args.max_residual_ms, args.public_model)
