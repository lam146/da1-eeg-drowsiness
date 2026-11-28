from pathlib import Path

import numpy as np
import pytest

from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel, load_model
from n16_eeg.preprocess import preprocess_window
from n16_eeg.signal_features import _welch_3s
from n16_eeg.alarm import AlarmPolicy


def test_smoothed_alarm_requires_new_observations_after_gap():
    policy = AlarmPolicy(.54, consecutive=2, release_consecutive=2, smoothing_window=3)
    assert [policy.update(x) for x in [.8, .8, .8, .8, None, .8, .8, .8, .8]] == [
        False, False, False, True, False, False, False, False, True]


def test_local_alpha_detects_intermittent_power_independent_of_gain(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_welch_alarms import local_spectral_features

    t = np.arange(1536) / 512
    signal = np.sin(2 * np.pi * 20 * t) + 3 * (t < 1.) * np.sin(2 * np.pi * 10 * t)
    # Broadband component provides a measurable slope.
    signal += np.random.default_rng(28).normal(scale=.05, size=len(t))
    features = local_spectral_features([signal, 4 * signal + 3, np.zeros_like(signal)])
    assert features[0, 0] > .8 and features[0, 1] > .8
    np.testing.assert_allclose(features[0], features[1], atol=1e-10)
    np.testing.assert_allclose(features[2], 0., atol=1e-10)


@pytest.mark.parametrize("estimator", ["welch_median", "welch_blend"])
def test_welch_executable_matches_training_reference(tmp_path, estimator):
    scipy = pytest.importorskip("scipy.signal")
    signal = np.random.default_rng(28).normal(size=1536)
    signal[512:524] += 80
    filtered = preprocess_window(signal)
    options = dict(fs=512, window="hann", nperseg=512, noverlap=256, nfft=1536)
    frequency, median = scipy.welch(filtered, average="median", **options)
    _, mean = scipy.welch(filtered, average="mean", **options)
    grid, actual = _welch_3s(filtered, 512, estimator)
    np.testing.assert_allclose(grid, frequency)
    np.testing.assert_allclose(actual, median if estimator == "welch_median" else (mean + median)/2,
                               rtol=1e-11, atol=1e-15)
    path = tmp_path / "model.json"
    LogisticModel(["theta_alpha"], [0], [1], [1], 0, format_version=2, spectral_estimator=estimator).save(path)
    restored = load_model(path)
    assert LiveDetector(restored, window_s=3).spectral_estimator == estimator
    with pytest.raises(ValueError, match="3-second"):
        LiveDetector(restored, window_s=4)


def test_missing_label_and_conflicting_note_never_generate_training_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from update_openvibe_model import recording_labels

    folder = Path("raw") / "Raw data"
    path = folder / "S03" / "example.csv"
    bounds = np.array([1536])
    args = (path, "hash", 4096, bounds - 1536, bounds)
    with pytest.raises(ValueError, match="Missing binary"):
        recording_labels(folder, {}, *args)
    with pytest.raises(ValueError, match="belongs to S02"):
        recording_labels(folder, {"note": "note/S02/example.docx"}, *args)
