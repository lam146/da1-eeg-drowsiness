"""Compare external EOG detectors with N16 using frozen video alignments."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from itertools import product

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

from compare_blink_detectors import read_session, transformed
from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_event_pairs, match_events


def predictions(signal):
    normal = (signal - np.median(signal)) / max(float(np.std(signal)), 1e-12)
    output = {'N16_default': np.array([event.time_seconds for event in detect_blinks(signal, 512)])}
    eog = mne.io.RawArray(normal[None, :], mne.create_info(['Frontal_artifact_proxy'], 512, ['eog']), verbose=False)
    events = mne.preprocessing.find_eog_events(eog, ch_name='Frontal_artifact_proxy', verbose=False)
    output['MNE_default'] = events[:, 0] / 512
    for polarity in (1, -1):
        cleaned = nk.eog_clean(normal * polarity, sampling_rate=512, method='neurokit')
        peaks = nk.eog_findpeaks(cleaned, sampling_rate=512, method='neurokit')
        output[f'NeuroKit_polarity_{polarity}'] = np.asarray(peaks) / 512
    return output


def tuned_mne(output):
    training = read_session('BLINK_01')
    signal = training['x']
    normal = (signal - np.median(signal)) / max(float(np.std(signal)), 1e-12)
    eog = mne.io.RawArray(normal[None, :], mne.create_info(['Frontal_artifact_proxy'], 512, ['eog']), verbose=False)
    trials = []
    for low, high, threshold in product((.5, 1.), (8., 10.), (.5, .75, 1., 1.5, 2., 3., 4., 6.)):
        events = mne.preprocessing.find_eog_events(eog, ch_name='Frontal_artifact_proxy',
            l_freq=low, h_freq=high, thresh=threshold, verbose=False)
        predicted = transformed(events[:, 0] / 512 + training['origin'], training, 'raw_anchors')
        trials.append(dict(low_hz=low, high_hz=high, threshold=threshold,
                           **match_events(predicted, training['reference'], .20)))
    pd.DataFrame(trials).to_csv(output / 'mne_training_trials.csv', index=False)
    selected = max(trials, key=lambda row: (row['f1'], row['precision']))
    results = []
    for name in ('BLINK_01', 'BLINK_02'):
        session = read_session(name)
        normal = (session['x'] - np.median(session['x'])) / max(float(np.std(session['x'])), 1e-12)
        eog = mne.io.RawArray(normal[None, :], mne.create_info(['Frontal_artifact_proxy'], 512, ['eog']), verbose=False)
        events = mne.preprocessing.find_eog_events(eog, ch_name='Frontal_artifact_proxy',
            l_freq=selected['low_hz'], h_freq=selected['high_hz'], thresh=selected['threshold'], verbose=False)
        for alignment in session['alignments']:
            predicted = transformed(events[:, 0] / 512 + session['origin'], session, alignment)
            results.append(dict(session=name, alignment=alignment,
                                **match_events(predicted, session['reference'], .20)))
    table = pd.DataFrame(results)
    table.to_csv(output / 'mne_tuned_comparison.csv', index=False)
    (output / 'mne_tuned_config.json').write_text(json.dumps(dict(selected=selected,
        selection='Highest BLINK_01 F1, then precision, under the previously fixed raw-anchor clock. BLINK_02 did not choose parameters. Both sessions were already inspected in earlier development; timing remains estimated.'), indent=2), encoding='utf-8')
    print('MNE selected', selected)
    print(table.to_string(index=False))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows, matches, provenance = [], [], {}
    for session_name in ('BLINK_01', 'BLINK_02'):
        session = read_session(session_name)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            detected = predictions(session['x'])
        provenance[session_name] = dict(alignments=session['alignments'], limits=session['limits'],
            warnings=[str(warning.message) for warning in captured],
            signal_sha256=hashlib.sha256(session['x'].tobytes()).hexdigest())
        for detector, relative_times in detected.items():
            eeg_times = relative_times + session['origin']
            pd.DataFrame({'eeg_time_s': eeg_times}).to_csv(output / f'{session_name}_{detector}.csv', index=False)
            for alignment in session['alignments']:
                predicted = transformed(eeg_times, session, alignment)
                reference = session['reference']
                result = match_events(predicted, reference, .20)
                rows.append(dict(session=session_name, detector=detector, alignment=alignment,
                                 references=len(reference), **result))
                for prediction, truth in match_event_pairs(predicted, reference, .20):
                    matches.append(dict(session=session_name, detector=detector, alignment=alignment,
                                        prediction_video_s=prediction, reference_video_s=truth))
        print(session_name, {name: len(times) for name, times in detected.items()}, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(matches).to_csv(output / 'matched_events.csv', index=False)
    (output / 'experiment.json').write_text(json.dumps(dict(
        versions=dict(mne=mne.__version__, neurokit2=nk.__version__, numpy=np.__version__),
        status='EXPLORATORY_ALREADY_REVIEWED_VIDEO',
        method='External library defaults; both NeuroKit polarities reported. Frontal EEG standardized to dimensionless proxy, no conversion from ADC counts to volts. Same cropped signals, references, fixed early alignments and 200 ms matching tolerance for all methods.',
        limits='Video synchronization is estimated, annotations single-reviewer and sessions previously used for development. Offline EOG algorithms do not establish live blink accuracy or warning latency. BLINK_03 drift prevents a reliable global timing comparison.',
        inputs=provenance), indent=2), encoding='utf-8')
    print(results[['session', 'detector', 'alignment', 'precision', 'recall', 'f1']].to_string(index=False))
    tuned_mne(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
