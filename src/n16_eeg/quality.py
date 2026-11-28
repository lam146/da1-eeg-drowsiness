"""Causal raw-count checks shared by CSV replay and live inference."""

from collections import deque
from math import isfinite


class RawArtifactGuard:
    def __init__(self, window_samples, fs, screen_raw):
        self.window_samples = window_samples
        self.flat_samples = max(1, round(fs * .25))
        self.screen_raw = screen_raw
        self.reset()

    def reset(self):
        self.flags = deque()
        self.bad_count = 0
        self.previous = None
        self.identical_samples = 0

    def update(self, sample):
        self.identical_samples = self.identical_samples + 1 if sample == self.previous else 1
        self.previous = sample
        bad = not isfinite(sample)
        if self.screen_raw:
            bad |= sample <= -2048 or sample >= 2047 or self.identical_samples >= self.flat_samples
        if len(self.flags) == self.window_samples:
            self.bad_count -= self.flags.popleft()
        self.flags.append(bad)
        self.bad_count += bad
        return self.bad_count > 0
