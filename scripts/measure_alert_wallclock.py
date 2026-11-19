"""Measure Python alarm timing while a labeled synthetic EEG stream is paced at 512 Hz."""

import argparse
import json
import time
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd

from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel


FS = 512


def wait_until(deadline):
    time.sleep(max(0., deadline-time.perf_counter()))


def measure(frame, model, batch_samples, release_consecutive, beep=None):
    onsets, = np.where(frame.event.eq('STATE_DROWSY').to_numpy())
    if len(onsets) != 1 or not frame.is_synthetic.eq(True).all():
        raise ValueError('Expected a synthetic file with exactly one STATE_DROWSY marker')
    onset = int(onsets[0])
    if onset < FS*3 or batch_samples < 1:
        raise ValueError('Need three seconds of initial EEG and a positive batch size')
    samples = frame.raw_eeg.to_numpy(float)
    poor = frame.poor_signal.to_numpy(float)
    detector = LiveDetector(model, fs=FS, window_s=3., step_s=.25, consecutive=3,
                            release_consecutive=release_consecutive)
    active, premature = False, 0
    marker_at = alarm_at = alarm_index = None
    sound = None
    origin = time.perf_counter()
    expected_onset = origin + onset/FS
    for start in range(0, len(frame), batch_samples):
        stop = min(start + batch_samples, len(frame))
        if start <= onset < stop:
            wait_until(expected_onset)
            marker_at = time.perf_counter()
            if beep is not None:
                Thread(target=beep, args=(1850, 120), daemon=True).start()
        wait_until(origin + (stop-1)/FS)
        for index in range(start, stop):
            decision = detector.add_sample(samples[index], poor[index])
            if decision is None:
                continue
            triggered = bool(decision['alarm']) and not active
            active = bool(decision['alarm'])
            if triggered and index < onset:
                premature += 1
            if triggered and index >= onset:
                alarm_at, alarm_index = time.perf_counter(), index
                if beep is not None:
                    sound = Thread(target=beep, args=(1250, 250), daemon=True)
                    sound.start()
                break
        if alarm_at is not None:
            break
    if sound is not None:
        sound.join()
    return dict(
        input='synthetic EEG paced at 512 Hz', batch_samples=batch_samples, threshold=model.threshold,
        release_consecutive=release_consecutive, premature_alarm_episodes=premature, sound_requested=beep is not None,
        marker_schedule_error_ms=(marker_at-expected_onset)*1000 if marker_at is not None else None,
        sample_axis_delay_ms=(alarm_index-onset)*1000/FS if alarm_index is not None else None,
        wall_ms_from_scheduled_onset=(alarm_at-expected_onset)*1000 if alarm_at is not None else None,
        wall_ms_from_marker_request=(alarm_at-marker_at)*1000 if alarm_at is not None else None,
        scope='PC monotonic clock until software alarm request. No headset, Bluetooth or acoustic onset measurement.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-samples', type=int, default=16)
    parser.add_argument('--release-consecutive', type=int, default=1)
    parser.add_argument('--sound', action='store_true', help='Play a high marker tone and lower alarm tone on Windows')
    args = parser.parse_args()
    beep = None
    if args.sound:
        from winsound import Beep

        beep = Beep
    result = measure(pd.read_csv(args.input), LogisticModel.load(args.model), args.batch_samples,
                     args.release_consecutive, beep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
