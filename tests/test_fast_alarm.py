import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd

from n16_eeg.alarm import AlarmPolicy
from n16_eeg import live
from n16_eeg.csv_test import test_recording as run_csv_recording
from n16_eeg.demo import replay
from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel


def test_fast_replay_and_stream_reset_confirmation_after_poor_signal(tmp_path):
    # A constant score isolates alarm timing and quality gating from the EEG model.
    model = LogisticModel(['theta_alpha'], [0.0], [1.0], [0.0], 4.0)
    fs = 128
    signal = pd.DataFrame({'timestamp_ms': range(4 * fs), 'raw_eeg': 0, 'poor_signal': 0})
    signal.loc[fs:fs + fs // 2 - 1, 'poor_signal'] = 200
    options = dict(fs=fs, window_s=1.0, step_s=0.25, consecutive=2)
    detector = LiveDetector(model, **options)
    expected = []
    for row in signal.itertuples(index=False):
        decision = detector.add_sample(row.raw_eeg, row.poor_signal)
        if decision is not None and decision['state'] != 'TIN_HIEU_KEM':
            expected.append((row.timestamp_ms, decision['alarm']))
    result = replay(signal, tmp_path / 'replay.csv', model=model, threshold=model.threshold,
                    verbose=False, **options)
    assert list(zip(result.window_end_ms, result.alarm)) == expected
    assert expected[:3] == [(127, 0), (319, 0), (351, 1)]


def test_release_waits_for_consecutive_low_scores_but_quality_resets_immediately():
    policy = AlarmPolicy(threshold=.54, consecutive=3, release_consecutive=3)
    risks = [.7, .7, .7, .4, .7, .4, .4, .4, .7, .7, .7, None, .7]
    expected = [False, False, True, True, True, True, True, False, False, False, True, False, False]
    assert [policy.update(risk) for risk in risks] == expected


def test_sustained_confirmation_rejects_brief_risk_spikes_and_resets_on_poor_contact():
    policy = AlarmPolicy(.57, consecutive=2, release_consecutive=3,
                         support_threshold=.545, support_consecutive=6)
    for score in [.60] * 5 + [.50] + [.60] * 5 + [None]:
        assert not policy.update(score)
    for score in [.55] * 4 + [.60]:
        assert not policy.update(score)
    assert policy.update(.60)


def test_csv_artifact_clears_alarm_and_requires_confirmation_again(tmp_path):
    fs = 512
    values = np.tile([0, 10], 3 * fs).astype(float)
    values[fs * 2] = -2048
    source = tmp_path / 'raw.csv'
    pd.DataFrame({'Time:512Hz': np.arange(len(values)) / fs, 'Electrode': values}).to_csv(source, index=False)
    model = LogisticModel(['theta_alpha'], [0.], [1.], [0.], 4.)
    detector = LiveDetector(model, fs=fs, window_s=1., step_s=.25, consecutive=2)
    output = tmp_path / 'output.csv'
    result = run_csv_recording(source, output, detector, Event(), screen_raw=True)
    rows = pd.read_csv(output)
    assert result['artifact_windows'] == 4
    flagged = rows.index[rows.raw_artifact.eq(1)]
    assert rows.loc[flagged, 'state'].eq('TIN_HIEU_KEM').all()
    assert rows.loc[flagged, 'alarm'].eq(0).all()
    assert rows.loc[flagged[-1] + 1:flagged[-1] + 2, 'alarm'].tolist() == [0, 1]


def test_live_alarm_waits_for_fresh_eeg_after_stall(monkeypatch, tmp_path):
    def packet(payload):
        return bytes([0xaa, 0xaa, len(payload), *payload, ~sum(payload) & 255])

    raw = packet([0x80, 2, 0, 31])
    good_contact = packet([2, 0])
    clock = SimpleNamespace(now=0.)
    # Build an alarm, drop delivery for .6 s, then send raw with unknown contact.
    batches = iter([(i / 4, good_contact + raw * 128) for i in range(1, 6)] +
                   [(1.8, b''), (1.85, b''), (1.9, raw * 64)] +
                   [(1.9 + i / 4, good_contact + raw * 128) for i in range(1, 7)])

    class FakeSerial:
        in_waiting = 4096

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            clock.now, chunk = next(batches, (6., b''))
            return chunk

    model_path = tmp_path / 'model.json'
    output = tmp_path / 'live.csv'
    model = LogisticModel(['theta_alpha'], [0.0], [1.0], [0.0], 4.0)
    model.save(model_path)
    monkeypatch.setattr(live, 'time', SimpleNamespace(monotonic=lambda: clock.now,
        time_ns=lambda: int(clock.now * 1e9), perf_counter_ns=lambda: 0))
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=lambda *args, **kwargs: FakeSerial()))
    monkeypatch.setattr(sys, 'argv', ['live', '--port', 'TEST', '--model', str(model_path),
        '--duration-min', '0.055', '--window-s', '1', '--step-s', '0.25', '--log-output', str(output), '--no-sound'])
    live.main()
    decisions = pd.read_csv(output)
    assert decisions.state.tolist()[:6] == [
        'NGUY_CO', 'CANH_BAO', 'MAT_TIN_HIEU', 'NGUY_CO', 'CANH_BAO', 'CANH_BAO',
    ]
    gap_index = decisions.index[decisions.state.eq('MAT_TIN_HIEU')][0]
    assert decisions.loc[gap_index, 'alarm'] == 0
    assert decisions.loc[gap_index + 1, 'timestamp_ms'] == 2900
    assert decisions.loc[gap_index + 1, 'poor_signal_fraction'] == 0.125
    assert decisions.loc[decisions.alarm.eq(1), 'episode_id'].drop_duplicates().tolist() == [1, 2]
    assert decisions.loc[decisions.alarm.eq(0), 'episode_id'].isna().all()
    assert decisions.loc[gap_index, 'raw_segment'] == 1
    assert pd.isna(decisions.loc[gap_index, 'sample_index'])
    assert decisions.sample_index.dropna().iloc[:3].tolist() == [511, 639, 1151]
