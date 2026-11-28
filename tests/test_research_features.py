import numpy as np
import pytest

from n16_eeg.research_features import research_features, sample_entropy


def test_entropy_counts_match_brute_force_without_self_matches():
    signal = np.random.default_rng(16).normal(size=90)
    embedded = np.lib.stride_tricks.sliding_window_view(signal, 3)
    counts = []
    for width in (2, 3):
        distances = np.max(np.abs(embedded[:, None, :width] - embedded[None, :, :width]), axis=-1)
        counts.append(int(np.sum(distances <= .3)) - len(embedded))
    expected = -np.log((counts[1] + 1) / (counts[0] + 1))
    assert sample_entropy(signal, .3) == pytest.approx(expected)


@pytest.mark.parametrize("fs", [128, 512])
def test_research_features_are_finite_on_constant_and_regular_windows(fs):
    time = np.arange(3 * fs) / fs
    for signal in (np.zeros(3 * fs), np.sin(2 * np.pi * 10 * time)):
        features = research_features(signal, fs=fs)
        assert len(features) == 12
        assert np.isfinite(list(features.values())).all()


def test_entropy_is_invariant_to_positive_signal_scaling():
    signal = np.random.default_rng(16).normal(size=1536)
    first, second = research_features(signal), research_features(signal * 20 + 5)
    np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-10)
