import json
import sys
from queue import Queue
from types import SimpleNamespace

import pandas as pd
import pytest

from n16_eeg import live
from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel


def test_live_saves_raw_markers_and_metadata_when_port_fails(monkeypatch, tmp_path):
    payload = bytes([2, 0, 0x80, 2, 0, 10])
    packet = bytes([0xaa, 0xaa, len(payload)]) + payload + bytes([~sum(payload) & 255])

    class SerialFixture:
        in_waiting = len(packet)

        def __init__(self):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            if self.sent:
                raise OSError('Test disconnect')
            self.sent = True
            return packet

    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=lambda *args, **kwargs: SerialFixture()))
    observations = Queue()
    observations.put(dict(event='KSS', value=4, requested_timestamp_ms=1))
    observations.put(dict(event='SELF_REPORT_AWAKE', value=None, requested_timestamp_ms=2))
    detector = LiveDetector(LogisticModel(['theta_alpha'], [0.], [1.], [0.], 4.))
    raw_path = tmp_path / 'raw.csv'
    saved = []
    with pytest.raises(OSError, match='Test disconnect'):
        live.run_live('COM_TEST', detector, tmp_path / 'predictions.csv', sound=False,
            raw_output=raw_path, subject_id='S03', session_id='LIVE_TEST', observations=observations,
            on_observation=saved.append)
    raw = pd.read_csv(raw_path)
    assert raw.raw_eeg.tolist() == [10]
    assert raw.kss.tolist() == [4]
    assert raw.event.tolist() == ['KSS|SELF_REPORT_AWAKE']
    assert raw.subject_id.tolist() == ['S03']
    assert raw.signal_time_ms.tolist() == [0.]
    metadata = json.loads(raw_path.with_suffix('.session.json').read_text())
    assert metadata['samples'] == 1 and metadata['stop_reason'] == 'error'
    assert [marker['next_sample_index'] for marker in metadata['observations']] == [0, 0]
    assert saved == metadata['observations']
    assert metadata['configuration']['threshold'] == detector.threshold
