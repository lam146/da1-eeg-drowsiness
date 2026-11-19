"""Audit 0–1000 ms video timestamp shifts without changing source recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_blink_detectors import FS, INPUTS, ROOT, TOLERANCE, in_range, read_session
from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_events


STEP_MS = 10


def peaks_in(samples, start_s, stop_s):
    start = max(0, int(np.floor(start_s * FS)))
    stop = min(len(samples), int(np.ceil(stop_s * FS)))
    return np.array([event.time_seconds + start / FS for event in detect_blinks(samples[start:stop], FS)])


def sweep(times, reference, video_limits, session, mode, max_delay_ms):
    rows = []
    for delay_ms in range(0, max_delay_ms + 1, STEP_MS):
        # Add the delay to each video annotation. Subtracting it from the
        # predicted times instead keeps both the scored interval and labels fixed.
        adjusted = in_range(times - delay_ms / 1000, video_limits)
        rows.append(dict(session=session, mode=mode, delay_ms=delay_ms,
                         **match_events(adjusted, reference, TOLERANCE)))
    return rows


def best(rows):
    return max(rows, key=lambda row: (row['f1'], row['true_positive'], -row['mean_absolute_timing_error_ms']))


def run(output, max_delay_by_session):
    all_rows, summary = [], {}
    for name, (filename, directory, _) in INPUTS.items():
        session = read_session(name)
        max_delay_ms = max_delay_by_session[name]
        margin = max_delay_ms / 1000
        x = pd.read_csv(ROOT / 'raw/Raw data/S02' / filename, usecols=['Electrode']).Electrode.to_numpy(float)
        ref = pd.read_csv(ROOT / 'output' / directory / 'video_blinks_confirmed.csv').timestamp_ms.to_numpy(float) / 1000
        cal_start, cal_stop = session['calibration_s']
        cal_limits = (cal_start, cal_stop - margin)
        limits = session['limits']
        if cal_limits[1] <= cal_start or limits[1]+margin > len(x)/FS:
            raise ValueError(f'{name}: delay range exceeds the common recording interval')
        # The guard keeps the calibration EEG crop disjoint from the holdout crop.
        early = peaks_in(x, cal_limits[0], cal_stop)
        test = peaks_in(x, limits[0], limits[1]+margin)
        early_rows = sweep(early, in_range(ref, cal_limits), cal_limits, name, 'absolute_calibration', max_delay_ms)
        absolute = sweep(test, session['reference'], limits, name, 'absolute_holdout', max_delay_ms)
        fit = session['alignments']['raw_anchors']
        anchor_times = peaks_in(x, (limits[0]-fit['offset_s'])/fit['scale'],
                                (limits[1]+1-fit['offset_s'])/fit['scale'])
        residual = sweep(anchor_times * fit['scale'] + fit['offset_s'], session['reference'], limits,
                         name, 'extra_after_raw_alignment', 1000)
        fitted = best(early_rows)
        # Invert the already-frozen raw-peak alignment to express the required
        # addition to a video timestamp in seconds on the raw EEG clock.
        video_to_eeg_scale = 1 / fit['scale']
        video_to_eeg_offset = -fit['offset_s'] / fit['scale']
        summary[name] = dict(
            references=len(session['reference']),
            absolute_range_ms=[0, max_delay_ms],
            video_holdout_s=limits, calibration_video_s=cal_limits,
            no_shift=absolute[0], best_absolute_on_holdout=best(absolute),
            best_calibration=fitted,
            holdout_at_calibrated_shift=absolute[fitted['delay_ms']//STEP_MS] if fitted['true_positive'] >= 4 else None,
            early_derived_video_to_eeg=dict(scale=video_to_eeg_scale, offset_s=video_to_eeg_offset,
                                           added_s_at_holdout_start=(video_to_eeg_scale-1)*limits[0]+video_to_eeg_offset,
                                           added_s_at_holdout_end=(video_to_eeg_scale-1)*limits[1]+video_to_eeg_offset),
            residual_no_shift=residual[0], best_residual_on_holdout=best(residual),
        )
        all_rows.extend(early_rows + absolute + residual)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(all_rows).to_csv(output / 'delay_sweep.csv', index=False)
    report = dict(shift_definition='Positive delay adds time to video blink timestamps',
                  residual_range_ms=[0, 1000], step_ms=STEP_MS, matching_tolerance_ms=200,
                  scope='Exploratory; best-on-holdout rows are an optimistic sensitivity audit, not validated accuracy.',
                  sessions=summary)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, modes in summary.items():
        for mode in ('no_shift', 'best_absolute_on_holdout', 'best_calibration', 'holdout_at_calibrated_shift',
                     'residual_no_shift', 'best_residual_on_holdout'):
            row = modes[mode]
            print(name, mode, 'insufficient anchors' if row is None else
                  (row['delay_ms'], round(row['f1']*100, 2), row['true_positive'], row['false_positive'], row['false_negative']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--blink01-max-ms', type=int, default=1000)
    parser.add_argument('--blink02-max-ms', type=int, default=1000)
    args = parser.parse_args()
    limits = dict(BLINK_01=args.blink01_max_ms, BLINK_02=args.blink02_max_ms)
    if any(limit < 0 or limit % STEP_MS for limit in limits.values()):
        parser.error('Delay limits must be nonnegative multiples of 10 ms')
    run(args.output, limits)
