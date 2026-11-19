import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from n16_eeg import acquire


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows marker console')
def test_recording_links_each_sync_marker_to_next_raw_sample(monkeypatch, tmp_path):
    payload = bytes([0x80, 2, 0, 31])
    packet = bytes([0xaa, 0xaa, len(payload)]) + payload + bytes([~sum(payload) & 255])
    receipt_times = iter((.05, .10, .15, .20, .85, .90, .95))
    clock = SimpleNamespace(now=0.)

    class SerialFixture:
        in_waiting = len(packet)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            clock.now = next(receipt_times)
            return packet

    class SilentThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

    keys = iter(('', 'm', '', 'm', '', 'm', '', 'q'))
    keyboard = SimpleNamespace(key='')

    def kbhit():
        keyboard.key = next(keys)
        return bool(keyboard.key)

    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=lambda *args, **kwargs: SerialFixture()))
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(kbhit=kbhit, getwch=lambda: keyboard.key))
    monkeypatch.setitem(sys.modules, 'winsound', SimpleNamespace(Beep=lambda *args: None))
    monkeypatch.setattr(acquire, 'Thread', SilentThread)
    monkeypatch.setattr(acquire, 'time', SimpleNamespace(monotonic=lambda: clock.now,
                                                       time_ns=lambda: int(clock.now * 1e9)))
    output = tmp_path/'blink.csv'
    monkeypatch.setattr(sys, 'argv', ['acquire', '--port', 'COM_TEST', '--subject', 'S03', '--session', 'BLINK_03',
                                     '--output', str(output), '--sync-markers'])
    acquire.main()
    frame = pd.read_csv(output)
    marked = frame.loc[frame.event.notna()]
    assert marked.event.tolist() == ['SYNC_01', 'SYNC_02', 'SYNC_03']
    assert marked.sample_index.tolist() == [1, 3, 5]
    log = json.loads(output.with_suffix('.capture.json').read_text())
    assert [row['next_sample_index'] for row in log['sync_markers']] == marked.sample_index.tolist()
    assert frame.raw_segment.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert log['raw_receipt_gaps'] == [dict(sample_index_after_gap=4, last_raw_elapsed_s=.2,
                                            recovered_elapsed_s=.85)]


def test_middle_marker_checks_drift_without_fitting_it():
    path = Path(__file__).parents[1]/'scripts'/'align_blink_sync_markers.py'
    spec = importlib.util.spec_from_file_location('sync_markers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = ['SYNC_01', 'SYNC_02', 'SYNC_03']
    video = pd.DataFrame({'event': ids, 'video_timestamp_ms': [1000., 6000., 11000.]})
    eeg = pd.DataFrame({'event': ids, 'signal_time_ms': [1510., 6560., 11610.]})
    _, fit = module.align_markers(eeg, video, 50.)
    assert fit['scale'] == pytest.approx(1.01)
    assert fit['offset_ms'] == pytest.approx(500.)
    assert fit['middle_check_passed']
    eeg.loc[1, 'signal_time_ms'] += 75
    _, fit = module.align_markers(eeg, video, 50.)
    assert fit['scale'] == pytest.approx(1.01)
    assert fit['max_middle_residual_ms'] == pytest.approx(75.)
    assert not fit['middle_check_passed']


def test_blink_score_requires_continuity_between_outer_markers(tmp_path):
    path = Path(__file__).parents[1]/'scripts'/'align_blink_sync_markers.py'
    spec = importlib.util.spec_from_file_location('sync_markers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recording = tmp_path / 'raw.csv'
    frame = pd.DataFrame({'sample_index': range(4096), 'fs_hz': 512, 'raw_eeg': 0, 'event': ''})
    frame['signal_time_ms'] = frame.sample_index * 1000 / 512
    log = []
    for number, index in enumerate((512, 2048, 3584), start=1):
        event = f'SYNC_{number:02d}'
        frame.loc[index, 'event'] = event
        log.append(dict(event=event, next_sample_index=index, requested_elapsed_ms=index,
                        next_sample_elapsed_ms=index + 5))
    frame.to_csv(recording, index=False)
    markers, blinks = tmp_path / 'markers.csv', tmp_path / 'blinks.csv'
    pd.DataFrame({'event': [row['event'] for row in log],
                  'video_timestamp_ms': [1000, 4000, 7000]}).to_csv(markers, index=False)
    pd.DataFrame({'timestamp_ms': [3000]}).to_csv(blinks, index=False)
    gap = dict(sample_index_after_gap=2000, last_raw_elapsed_s=3.9, recovered_elapsed_s=4.5)
    for name, gaps in [('continuous', []), ('interrupted', [gap])]:
        recording.with_suffix('.capture.json').write_text(json.dumps({'sync_markers': log,
                                                                      'raw_receipt_gaps': gaps}))
        output = tmp_path / name
        module.run(recording, markers, blinks, output, 50.)
        result = json.loads((output / 'metrics.json').read_text())
        assert result['fit']['middle_check_passed'] and result['receipt_check_passed']
        assert result['raw_receipt_gaps_in_marker_span'] == gaps
        assert (result['score'] is None) == bool(gaps)
