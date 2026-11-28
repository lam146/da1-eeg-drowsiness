"""Audit local recordings and compare packaged runtime profiles reproducibly."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import time_ns

import pandas as pd

from n16_live_app import load_detector
from n16_eeg.csv_test import test_recording
from n16_eeg.simulate import generate_session
from n16_eeg.alarm_eval import alarm_onset_metrics


def run(output, timing_only=False):
    output.mkdir(parents=True, exist_ok=True)
    release = output / 'release'
    paths = sorted(Path('raw').rglob('*.csv'))
    recording_folder = output / 'recordings' / str(time_ns())

    def inspect(source):
        detector = load_detector(release)
        out = recording_folder / f'{source.parent.name}_{source.stem}.csv'
        identity = dict(file=str(source), person=source.parent.name,
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        try:
            result = test_recording(source, out, detector, Event())
            print(source.name, result['windows'], result['artifact_windows'], flush=True)
            return identity | dict(status='ok', **result)
        except (OSError, ValueError) as exc:
            return identity | dict(status='error', error=str(exc))

    if timing_only:
        inventory = pd.read_csv(output / 'recording_inventory.csv').to_dict('records')
        for row in inventory:
            if hashlib.sha256(Path(row['file']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Recording changed since the previous inventory')
    else:
        with ThreadPoolExecutor(max_workers=3) as workers:
            inventory = list(workers.map(inspect, paths))
        pd.DataFrame(inventory).to_csv(output / 'recording_inventory.csv', index=False, encoding='utf-8-sig')
    comparisons = []
    profiles = dict(Reviewed=release, Imported_RBF=Path('output/N16_CapNhat_20261127/release'))
    for seed in (401, 409, 419, 431, 443, 457):
        for phase in (.05, .29, .56, .87):
            raw = generate_session(alert_seconds=12 + phase, drowsy_seconds=10, seed=seed)
            onset = raw.loc[raw.event.eq('STATE_DROWSY'), 'timestamp_ms'].iloc[0]
            for name, profile in profiles.items():
                detector = load_detector(profile)
                alarms, times = [], []
                for row in raw.itertuples(index=False):
                    decision = detector.add_sample(row.raw_eeg, row.poor_signal)
                    if decision is not None:
                        times.append(row.timestamp_ms)
                        alarms.append(bool(decision['alarm']))
                measured = alarm_onset_metrics(times, alarms, onset)
                comparisons.append(dict(profile=name, seed=seed, phase=phase,
                    premature=measured['pre_onset_alarm'], delay_ms=measured['onset_to_alarm_ms'],
                    first_active_after_onset_ms=measured['first_active_after_onset_ms'],
                    valid_under_3s=measured['valid_under_3s']))
    timing = pd.DataFrame(comparisons)
    timing.to_csv(output / 'runtime_comparison.csv', index=False)
    timing_report = {name: dict(cases=len(group), early=int(group.premature.sum()),
        invalid_or_missing=int(group.delay_ms.isna().sum()), valid_under_3s=int(group.valid_under_3s.sum()),
        worst_valid_delay_ms=float(group.delay_ms.max()) if group.delay_ms.notna().any() else None)
        for name, group in timing.groupby('profile')}
    summary = dict(raw_files=len(inventory), local_people=sorted({row['person'] for row in inventory}),
                   valid_files=sum(row['status'] == 'ok' for row in inventory), timing=timing_report,
                   label_status='Original S01/S02 KSS windows remain provisional; local S03 note/CSV mismatch; imported five-person training data unavailable.',
                   scope='Diagnostic replay. Synthetic sample-axis timing, physical Bluetooth/audio timing pending.')
    (output / 'review.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.timing_only)
