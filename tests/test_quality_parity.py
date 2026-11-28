import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd

from n16_eeg import live
from n16_eeg.csv_test import test_recording as run_csv_recording
from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel


def constant_model():
    return LogisticModel(['theta_alpha'], [0.], [1.], [0.], 4.)


def test_csv_and_stream_have_identical_causal_quality_and_decisions(tmp_path):
    fs = 128
    signal = np.tile([0., 10.], 5 * fs // 2)
    signal[fs:fs + fs // 2] = 31
    signal[3 * fs] = -2048
    # CSV format is fixed to 512 Hz, so use fourfold repetition with a real 512 Hz detector.
    signal = np.repeat(signal, 4)
    frame = pd.DataFrame({'Time:512Hz': np.arange(len(signal)) / 512, 'Electrode': signal})
    source, output = tmp_path / 'input.csv', tmp_path / 'output.csv'
    frame.to_csv(source, index=False)
    options = dict(window_s=1, step_s=.25, consecutive=2, screen_raw=True)
    stream = LiveDetector(constant_model(), **options)
    expected = []
    for sample in signal:
        decision = stream.add_sample(sample, 0)
        if decision is not None:
            expected.append((decision['state'], decision['risk'], decision['alarm'], decision['raw_artifact']))
    run_csv_recording(source, output, LiveDetector(constant_model(), **options), Event())
    actual = pd.read_csv(output, keep_default_na=False)
    assert list(zip(actual.state, actual.risk.astype(str), actual.alarm, actual.raw_artifact)) == [
        (state, str(risk), alarm, artifact) for state, risk, alarm, artifact in expected]
    assert actual.loc[actual.raw_artifact.eq(1), 'alarm'].eq(0).all()
    assert actual.state.iloc[0] == 'NGUY_CO'


def test_nonfinite_eeg_and_quality_clear_confirmation():
    detector = LiveDetector(constant_model(), fs=128, window_s=1, step_s=.25, consecutive=1)
    for _ in range(128):
        decision = detector.add_sample(0, 0)
    assert decision['alarm'] == 1
    for _ in range(32):
        decision = detector.add_sample(np.nan, 0)
    assert decision['alarm'] == 0 and decision['raw_artifact'] == 1
    detector.reset()
    for _ in range(128):
        decision = detector.add_sample(0, np.nan)
    assert decision['state'] == 'TIN_HIEU_KEM'


def test_live_requires_fresh_contact_quality_even_when_raw_keeps_arriving(monkeypatch, tmp_path):
    def packet(payload):
        return bytes([0xaa, 0xaa, len(payload), *payload, ~sum(payload) & 255])

    clock = SimpleNamespace(now=0.)
    raw = packet([0x80, 2, 0, 10]) * 128

    class SerialFixture:
        in_waiting = len(raw)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            clock.now += .25
            return (packet([2, 0]) if clock.now == .25 else b'') + raw

    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=lambda *args, **kwargs: SerialFixture()))
    monkeypatch.setattr(live, 'time', SimpleNamespace(monotonic=lambda: clock.now,
        time_ns=lambda: int(clock.now * 1e9), perf_counter_ns=lambda: 0))
    output = tmp_path / 'live.csv'
    detector = LiveDetector(constant_model(), window_s=1, step_s=.25, consecutive=1)
    live.run_live('COM_TEST', detector, output, sound=False, duration_min=.075)
    rows = pd.read_csv(output)
    assert rows.loc[rows.timestamp_ms.lt(3250), 'alarm'].eq(1).all()
    assert rows.loc[rows.timestamp_ms.ge(3500), 'state'].eq('TIN_HIEU_KEM').all()
