"""Offline AR and multiscale sample entropy candidates for the N16 pilot."""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.spatial import cKDTree

from .preprocess import preprocess_window


def sample_entropy(signal, tolerance):
    # Both embeddings use the same starting indices and exclude self matches.
    triples = np.lib.stride_tricks.sliding_window_view(signal, 3)
    counts = []
    for dimension in (2, 3):
        tree = cKDTree(triples[:, :dimension])
        counts.append(int(tree.count_neighbors(tree, tolerance, p=np.inf)) - len(triples))
    return float(-np.log((counts[1] + 1) / (counts[0] + 1)))


def research_features(samples, fs=512):
    if fs not in (128, 512):
        raise ValueError("Research feature configuration requires 128 or 512 Hz")
    filtered = preprocess_window(samples, fs=fs)
    # Band limit is 45 Hz, below the 64 Hz Nyquist frequency after decimation.
    signal = filtered[::fs // 128]
    centered = signal - signal.mean()
    variance = float(np.mean(centered**2))
    normal = centered / max(np.sqrt(variance), 1e-12)
    correlation = np.correlate(normal, normal, mode="full")[len(normal) - 1:len(normal) + 8] / len(normal)
    correlation[0] += 1e-6
    coefficients = solve_toeplitz(correlation[:8], correlation[1:9])
    features = {f"ar_{index}": float(value) for index, value in enumerate(coefficients, 1)}
    features["ar_log_residual"] = float(np.log(max(correlation[0] - coefficients @ correlation[1:9], 1e-12)))
    for scale in (1, 2, 4):
        coarse = normal[:len(normal) // scale * scale].reshape(-1, scale).mean(axis=1)
        features[f"sample_entropy_scale_{scale}"] = sample_entropy(coarse, .2)
    return features
