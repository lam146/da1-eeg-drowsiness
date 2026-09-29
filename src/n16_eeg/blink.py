"""Baseline blink detector for frontal single-channel EEG.

This intentionally simple adaptive detector is a starting point. Its polarity,
threshold and timing must be calibrated against manually annotated N16 data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlinkEvent:
    sample_index: int
    time_seconds: float
    amplitude: float
    width_seconds: float


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return x.astype(float, copy=True)
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(x, kernel, mode="same")


def detect_blinks(
    samples: np.ndarray,
    fs: float = 512.0,
    threshold_mad: float = 6.0,
    refractory_s: float = 0.50,
    min_width_s: float = 0.08,
    max_width_s: float = 0.60,
) -> list[BlinkEvent]:
    """Detect large low-frequency transients using robust adaptive thresholds."""

    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size < int(fs):
        return []

    baseline = _moving_average(x, int(0.8 * fs))
    smooth = _moving_average(x - baseline, int(0.04 * fs))
    center = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - center)))
    sigma = max(1.4826 * mad, np.finfo(float).eps)
    score = np.abs(smooth - center) / sigma
    candidates = np.flatnonzero((score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:]) & (score[1:-1] >= threshold_mad)) + 1

    events: list[BlinkEvent] = []
    refractory = max(1, int(refractory_s * fs))
    low_threshold = threshold_mad * 0.35
    # Strongest-first non-maximum suppression prevents slow-baseline side lobes
    # from being counted before the actual blink peak.
    selected: list[int] = []
    for index in candidates[np.argsort(score[candidates])[::-1]]:
        if any(abs(int(index) - previous) < refractory for previous in selected):
            continue
        left = index
        while left > 0 and score[left] > low_threshold:
            left -= 1
        right = index
        while right + 1 < score.size and score[right] > low_threshold:
            right += 1
        width = (right - left) / fs
        if min_width_s <= width <= max_width_s:
            events.append(BlinkEvent(int(index), float(index / fs), float(smooth[index]), float(width)))
            selected.append(int(index))
    return sorted(events, key=lambda event: event.sample_index)


def blink_window_features(samples: np.ndarray, fs: float = 512.0) -> dict[str, float]:
    events = detect_blinks(samples, fs=fs)
    duration_minutes = len(samples) / fs / 60.0
    amplitudes = np.abs([event.amplitude for event in events])
    widths = [event.width_seconds for event in events]
    return {
        "blink_count": float(len(events)),
        "blink_rate_per_min": float(len(events) / max(duration_minutes, 1e-9)),
        "blink_amp_mean": float(np.mean(amplitudes)) if len(amplitudes) else 0.0,
        "blink_width_mean": float(np.mean(widths)) if widths else 0.0,
        "long_blink_count": float(sum(width >= 0.35 for width in widths)),
    }
