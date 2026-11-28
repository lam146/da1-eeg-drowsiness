"""Signal preprocessing and spectral features for one-channel EEG."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .preprocess import preprocess_window


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@lru_cache(maxsize=32)
def _periodogram_grid(size, fs):
    window = np.hanning(size)
    scale = fs * np.sum(window**2)
    return window, scale, np.fft.rfftfreq(size, 1.0 / fs)


def _periodogram(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a Hann-windowed one-sided periodogram without scipy."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("A 1-D window with at least 8 samples is required")
    x = x - np.median(x)
    window, scale, frequencies = _periodogram_grid(x.size, fs)
    spectrum = np.fft.rfft(x * window)
    psd = (np.abs(spectrum) ** 2) / max(scale, np.finfo(float).eps)
    if psd.size > 2:
        psd[1:-1] *= 2.0
    return frequencies, psd


def _welch_3s(filtered, fs, estimator):
    if fs != 512 or len(filtered) != 1536:
        raise ValueError("The N16 Welch model requires a 3-second window at 512 Hz")
    frames = np.lib.stride_tricks.sliding_window_view(filtered, 512)[::256]
    taper = np.hanning(513)[:-1]  # periodic Hann
    transformed = np.fft.rfft((frames - frames.mean(axis=1, keepdims=True)) * taper, n=1536)
    spectra = np.abs(transformed) ** 2 / (fs * np.sum(taper**2))
    spectra[:, 1:-1] *= 2.
    # Five half-overlapping periodograms. SciPy's median bias factor for five is 47/60.
    median = np.median(spectra, axis=0) / (47 / 60)
    psd = median if estimator == "welch_median" else (spectra.mean(axis=0) + median) / 2
    return np.fft.rfftfreq(1536, 1 / fs), psd


def bandpower_features(samples: np.ndarray, fs: float = 512.0, estimator: str = "periodogram") -> dict[str, float]:
    """Compute absolute/relative band powers and robust drowsiness ratios."""

    filtered = preprocess_window(np.asarray(samples), fs=fs)
    if estimator == "periodogram":
        freqs, psd = _periodogram(filtered, fs)
    elif estimator in ("welch_median", "welch_blend"):
        freqs, psd = _welch_3s(filtered, fs, estimator)
    else:
        raise ValueError(f"Unsupported spectral estimator: {estimator}")
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask]))
    total = max(total, np.finfo(float).eps)

    features: dict[str, float] = {}
    powers: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        power = float(np.trapezoid(psd[mask], freqs[mask])) if np.count_nonzero(mask) > 1 else 0.0
        powers[name] = power
        features[f"abs_{name}"] = power
        features[f"rel_{name}"] = power / total

    eps = np.finfo(float).eps
    features["theta_alpha"] = powers["theta"] / max(powers["alpha"], eps)
    features["theta_beta"] = powers["theta"] / max(powers["beta"], eps)
    features["alpha_beta"] = powers["alpha"] / max(powers["beta"], eps)
    normalized_psd = psd[total_mask] / max(float(np.sum(psd[total_mask])), eps)
    features["spectral_entropy"] = float(
        -np.sum(normalized_psd * np.log2(normalized_psd + eps))
        / max(np.log2(normalized_psd.size), eps)
    )
    features["rms"] = float(np.sqrt(np.mean(np.square(filtered))))
    features["peak_to_peak"] = float(np.ptp(filtered))
    return features
