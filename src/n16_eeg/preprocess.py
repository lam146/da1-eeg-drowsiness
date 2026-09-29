"""Deterministic preprocessing for short single-channel EEG windows."""

from __future__ import annotations

import numpy as np


def preprocess_window(
    samples: np.ndarray,
    fs: float = 512.0,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
    notch_hz: float = 50.0,
    notch_width_hz: float = 1.0,
) -> np.ndarray:
    """Band-limit and notch a window using an FFT mask.

    The FFT implementation keeps the core package usable without SciPy. For the
    final experiment, the same parameters must be frozen and reported.
    """

    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("A 1-D window with at least 8 samples is required")
    if not np.all(np.isfinite(x)):
        raise ValueError("EEG window contains non-finite samples")
    if not (0 <= low_hz < high_hz < fs / 2):
        raise ValueError("Band limits must satisfy 0 <= low < high < Nyquist")

    centered = x - np.median(x)
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / fs)
    keep = (frequencies >= low_hz) & (frequencies <= high_hz)
    if 0 < notch_hz < fs / 2:
        keep &= np.abs(frequencies - notch_hz) > notch_width_hz / 2
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=centered.size)
