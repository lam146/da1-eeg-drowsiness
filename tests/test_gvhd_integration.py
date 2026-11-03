"""Fixtures here are synthetic; none are SEED-VIG or human EEG results."""
import json

import numpy as np
import pandas as pd
import pytest

from n16_eeg.openvibe_import import convert_frame
from n16_eeg.thinkgear import ThinkGearParser
from n16_eeg.train import select_feature_columns


def packet(payload):
    return b"\xaa\xaa" + bytes([len(payload)]) + payload + bytes([(~sum(payload)) & 255])


def test_parser_fragmentation_stats_and_signed_limits():
    parser = ThinkGearParser()
    data = packet(bytes.fromhex("80 02 80 00 80 02 7F FF"))
    result = []
    for byte in data:
        result.extend(parser.feed(bytes([byte])))
    assert result[0].raw_eeg == (-32768, 32767)
    assert parser.valid_packets == 1
    broken = bytearray(data)
    broken[-1] ^= 1
    assert parser.feed(broken) == []
    assert parser.checksum_errors == 1
    assert parser.feed(data)[0].raw_eeg == (-32768, 32767)


@pytest.mark.parametrize("payload", ["80 03 00", "80 01 00", "02", "55"])
def test_parser_does_not_return_partial_malformed_payload(payload):
    parser = ThinkGearParser()
    assert parser.feed(packet(bytes.fromhex(payload))) == []
    assert parser.payload_errors == 1


def test_extended_code_not_misread_as_standard_raw():
    parser = ThinkGearParser()
    result = parser.feed(packet(bytes.fromhex("55 80 02 FF FF 80 02 00 64")))
    assert result[0].raw_eeg == (100,)


def openvibe_frame():
    return pd.DataFrame({"Time:512Hz": np.arange(512) / 512, "Electrode": np.arange(512),
                         "Delta": 999, "Quality": 0})


def test_openvibe_preserves_unit_and_does_not_invent_quality_or_labels():
    frame = convert_frame(openvibe_frame(), "Time:512Hz", "Electrode", "S01", "run", "uV")
    assert len(frame) == 512
    assert frame["poor_signal"].isna().all()
    assert frame["kss"].isna().all()
    assert frame["raw_eeg"].iloc[10] == 10
    assert set(frame["signal_unit"]) == {"uV"}
    assert "Delta" not in frame


@pytest.mark.parametrize("bad", ["time", "fs", "unit", "same_column"])
def test_openvibe_rejects_ambiguous_input(bad):
    data = openvibe_frame()
    kwargs = {}
    electrode = "Electrode"
    if bad == "time":
        data.loc[3, "Time:512Hz"] = 0
    if bad == "fs":
        kwargs["fs"] = 128
    if bad == "unit":
        kwargs["unit"] = "V"
    if bad == "same_column":
        electrode = "Time:512Hz"
    kwargs.setdefault("unit", "count")
    with pytest.raises(ValueError):
        convert_frame(data, "Time:512Hz", electrode, "S", "r", **kwargs)


def test_legacy_selection_excludes_labels_and_new_metadata():
    frame = pd.DataFrame({"subject_id": ["S1"], "label": [1], "perclos": [.8],
                          "kss": [8], "fs_hz": [512], "sample_index": [0],
                          "feature_row": [3], "theta_alpha": [1.5]})
    assert select_feature_columns(frame) == ["theta_alpha"]


def fixture_features():
    # Person S3 has only class 0. Its test fold must remain visible.
    return pd.DataFrame({"subject_id": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 2,
        "label": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        "f": [-2, -1, 1, 2, -1.5, -1, 1, 1.5, 100, 102],
        "source_is_synthetic": True, "dataset_id": "seed-vig"})


def test_sklearn_loso_retains_single_class_test_and_scaler_is_train_only():
    pytest.importorskip("sklearn")
    from n16_eeg.sklearn_train import evaluate_frame
    final, report, predictions = evaluate_frame(fixture_features(), ["f"], "seed-vig")
    assert len(report["folds"]) == 3
    fold = next(f for f in report["folds"] if f["subject"] == "S3")
    assert fold["status"] == "evaluated"
    assert fold["roc_auc"] is None
    assert fold["balanced_accuracy"] is None
    assert abs(fold["scaler_mean"][0]) < 1e-12
    assert "S3" not in fold["train_subjects"]
    assert len(predictions) == 10
    assert "SYNTHETIC" in report["warning"]
    assert final.predict_proba([[0]]).shape == (1, 2)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("classifier", ["svm", "random-forest"])
def test_sklearn_other_baselines(classifier):
    pytest.importorskip("sklearn")
    from n16_eeg.sklearn_train import evaluate_frame
    _, report, predictions = evaluate_frame(fixture_features(), ["f"], "seed-vig", classifier)
    assert len(predictions) == 10
    assert report["classifier"] == classifier


@pytest.mark.parametrize("bad", ["target", "fractional_label", "dataset", "nan"])
def test_sklearn_rejects_leakage_and_invalid_data(bad):
    from n16_eeg.sklearn_train import validate_frame
    frame = fixture_features()
    features = ["f"]
    if bad == "target":
        features = ["perclos"]
        frame["perclos"] = .5
    if bad == "fractional_label":
        frame["label"] = frame["label"].astype(float)
        frame.loc[0, "label"] = .5
    if bad == "dataset":
        frame.loc[0, "dataset_id"] = "neurosky"
    if bad == "nan":
        frame.loc[0, "f"] = np.nan
    with pytest.raises(ValueError):
        validate_frame(frame, features, "seed-vig")


def test_sklearn_keeps_skipped_training_fold_visible():
    pytest.importorskip("sklearn")
    from n16_eeg.sklearn_train import evaluate_frame
    frame = fixture_features()
    frame["label"] = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    _, report, _ = evaluate_frame(frame, ["f"], "seed-vig")
    fold = report["folds"][0]
    assert fold["subject"] == "S1" and fold["status"] == "skipped"


def write_mat_manifest(tmp_path):
    scipy = pytest.importorskip("scipy.io")
    scipy.savemat(tmp_path / "features.mat", {"psd_movingAve": np.arange(30).reshape(2, 3, 5)})
    scipy.savemat(tmp_path / "labels.mat", {"perclos": [[.1, .5, .9]]})
    item = {"subject_id": "person_01", "session_id": "run_01", "feature_file": "features.mat",
        "feature_key": "psd_movingAve", "sample_axis": 1,
        "label_file": "labels.mat", "label_key": "perclos", "alignment_verified": True}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"dataset": "seed-vig", "sessions": [item]}), encoding="utf-8")
    return path


def test_seed_manifest_preserves_person_alignment_and_thresholds(tmp_path):
    from n16_eeg.seed_vig import convert_manifest, inspect_mat
    manifest = write_mat_manifest(tmp_path)
    result, provenance = convert_manifest(manifest, .2, .8)
    assert len(result) == 2
    assert result["label"].tolist() == [0, 1]
    assert result["feature_row"].tolist() == [0, 2]
    assert result["subject_id"].tolist() == ["person_01"] * 2
    assert result["seed_psd_movingAve_0000"].tolist() == [0, 10]
    assert len(provenance["feature_names"]) == 10
    assert inspect_mat(tmp_path / "features.mat")[0]["shape"] == [2, 3, 5]


@pytest.mark.parametrize("bad", ["axis", "alignment", "label_length", "threshold"])
def test_seed_rejects_unverified_or_misaligned_data(tmp_path, bad):
    from n16_eeg.seed_vig import convert_manifest
    manifest = write_mat_manifest(tmp_path)
    config = json.loads(manifest.read_text())
    if bad == "axis":
        config["sessions"][0]["sample_axis"] = 0
    if bad == "alignment":
        config["sessions"][0]["alignment_verified"] = False
    if bad == "label_length":
        from scipy.io import savemat
        savemat(tmp_path / "labels.mat", {"perclos": [[.1, .2]]})
    manifest.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        convert_manifest(manifest, .8 if bad == "threshold" else .2, .3 if bad == "threshold" else .8)


def test_featurizer_retains_provenance_and_rejects_mixed_sessions():
    from n16_eeg.featurize import featurize_session
    from n16_eeg.simulate import generate_session
    frame = generate_session(alert_seconds=5, drowsy_seconds=5)
    frame["sample_index"] = np.arange(len(frame))
    frame["signal_unit"] = "uV"
    frame["dataset_id"] = "openvibe"
    frame["fs_hz"] = 512
    result = featurize_session(frame)
    assert set(result["signal_unit"]) == {"uV"}
    assert set(result["dataset_id"]) == {"openvibe"}
    frame.loc[0, "session_id"] = "other"
    with pytest.raises(ValueError):
        featurize_session(frame)


@pytest.mark.parametrize("kind", ["dataset", "unit"])
def test_legacy_export_cannot_silently_use_external_features(tmp_path, kind):
    from n16_eeg.train import train_from_frame
    frame = fixture_features()
    if kind == "unit":
        frame["dataset_id"] = "neurosky"
        frame["signal_unit"] = "uV"
    with pytest.raises(ValueError):
        train_from_frame(frame, tmp_path / "model.json", tmp_path / "metrics.json")


def test_sklearn_cli_writes_oof_and_joblib_without_runtime_export(tmp_path):
    pytest.importorskip("sklearn")
    import os
    import subprocess
    import sys
    import joblib
    frame = fixture_features()
    source = tmp_path / "features.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "run"
    environment = dict(os.environ, PYTHONPATH=str(__import__("pathlib").Path("src").resolve()))
    command = [sys.executable, "-m", "n16_eeg.sklearn_train", str(source), "--features", "f",
               "--dataset", "seed-vig", "--output-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True, env=environment)
    model = joblib.load(output / "pipeline.joblib")
    assert model.predict([[0]]).shape == (1,)
    assert len(pd.read_csv(output / "oof_predictions.csv")) == len(frame)
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert report["dataset"] == "seed-vig"
    assert not (output / "model.json").exists()
    again = subprocess.run(command, capture_output=True, env=environment)
    assert again.returncode != 0  # Preserve old outputs.
