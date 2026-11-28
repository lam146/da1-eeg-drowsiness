"""Save raw EEG and explicit user observations from the live app together."""

import csv
import json
import time
from queue import Empty


class SessionRecording:
    def __init__(self, path, subject, session, fs, observations, configuration, on_observation=None):
        self.path = path
        self.subject = subject
        self.session = session
        self.fs = fs
        self.observations = observations
        self.configuration = configuration
        self.on_observation = on_observation
        self.markers = []
        self.pending = []
        self.samples = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('x', newline='', encoding='utf-8')
        self.writer = csv.writer(self.handle)
        self.writer.writerow(['timestamp_ms', 'subject_id', 'session_id', 'raw_eeg', 'poor_signal',
                              'kss', 'event', 'sample_index', 'signal_time_ms', 'signal_unit',
                              'dataset_id', 'fs_hz', 'raw_segment'])
        self.handle.flush()
        return self

    def collect_observations(self):
        if self.observations is None:
            return
        while True:
            try:
                marker = dict(self.observations.get_nowait())
            except Empty:
                return
            marker['next_sample_index'] = None
            self.markers.append(marker)
            self.pending.append(marker)

    def write(self, sample, quality, receipt_ms, segment):
        events, kss = [], ''
        for marker in self.pending:
            events.append(marker['event'])
            if marker['event'] == 'KSS':
                kss = marker['value']
        self.writer.writerow([receipt_ms, self.subject, self.session, sample, quality, kss,
                              '|'.join(events), self.samples, self.samples * 1000 / self.fs,
                              'count', 'neurosky', self.fs, segment])
        if self.pending:
            self.flush()
            for marker in self.pending:
                marker.update(next_sample_index=self.samples, sample_receipt_ms=receipt_ms,
                              signal_time_ms=self.samples * 1000 / self.fs, raw_segment=segment)
                if self.on_observation is not None:
                    self.on_observation(dict(marker))
            self.pending.clear()
        self.samples += 1

    def flush(self):
        self.handle.flush()

    def __exit__(self, error_type, error, traceback):
        self.collect_observations()
        self.handle.close()
        payload = dict(subject_id=self.subject, session_id=self.session, samples=self.samples,
                       observations=self.markers, configuration=self.configuration,
                       stopped_timestamp_ms=time.time_ns() // 1_000_000,
                       stop_reason='error' if error_type else 'stopped', error=str(error) if error else None,
                       time_note='PC receipt timestamps; signal_time_ms is sample index/fs. Observations attach to the next received raw sample. SELF_REPORT_AWAKE and SELF_REPORT_DROWSY are user observations, pending label review.')
        self.path.with_suffix('.session.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
