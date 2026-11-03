"""Synthetic MAT fixtures test imports; no SEED-VIG participant results here."""
import json

import numpy as np
import pytest

from n16_eeg import ml_workbench as work


def fixture_manifest(tmp_path):
    scipy = pytest.importorskip("scipy.io")
    sessions = []
    for i in range(3):
        feature = f"features_{i}.mat"
        label = f"labels_{i}.mat"
        scipy.savemat(tmp_path / feature, {"psd_movingAve": np.arange(60).reshape(2, 6, 5) + i})
        scipy.savemat(tmp_path / label, {"perclos": [[.1, .2, .5, .8, .9, .4]]})
        sessions.append(dict(subject_id=f"person_{i}", session_id="run_1", feature_file=feature,
            label_file=label, feature_key="psd_movingAve", label_key="perclos", sample_axis=1,
            alignment_verified=True))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(dataset="seed-vig", source_is_synthetic=True,
                                   sessions=sessions)), encoding="utf-8")
    return path


def test_seed_import_audited_training_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    path = fixture_manifest(tmp_path)
    target = tmp_path / "runs"
    directory, audit = work.import_seed_manifest(path, .2, .8, "fixture policy", True, target)
    assert audit["subjects"] == 3 and audit["rows"] == 12
    assert len(audit["inputs"]) == 7
    assert all(len(item["sha256"]) == 64 for item in audit["inputs"])
    stored = json.loads((directory / "seed_import.json").read_text(encoding="utf-8"))
    assert stored["manifest_snapshot"]["source_is_synthetic"] is True
    assert stored["label_policy"]["alert_max"] == .2
    assert stored["policy"] == "fixture policy"
    frame = work.read_csv_file(directory / "features.csv")
    assert frame["source_is_synthetic"].all()
    assert set(frame["dataset_id"]) == {"seed-vig"}
    trained, report = work.train_files([directory / "features.csv"], audit["feature_names"],
                                     "seed-vig", "logistic", target)
    assert len(report["folds"]) == 3
    assert "SYNTHETIC" in report["warning"]
    assert (trained / "pipeline.joblib").exists()
    again, _ = work.import_seed_manifest(path, .2, .8, "fixture policy", True, target)
    assert again != directory
    with pytest.raises(ValueError, match="Dataset provenance"):
        work.train_files([directory / "features.csv"], audit["feature_names"],
                         "neurosky", "logistic", target)


@pytest.mark.parametrize("problem", ["policy", "review", "threshold", "alignment", "source_flag"])
def test_seed_invalid_import_does_not_create_outputs(tmp_path, problem):
    path = fixture_manifest(tmp_path)
    config = json.loads(path.read_text())
    if problem == "alignment": config["sessions"][0]["alignment_verified"] = False
    if problem == "source_flag": config["source_is_synthetic"] = "false"
    path.write_text(json.dumps(config), encoding="utf-8")
    target = tmp_path / "must_not_be_created"
    with pytest.raises(ValueError):
        work.import_seed_manifest(path, .9 if problem == "threshold" else .2, .8,
                                  "" if problem == "policy" else "fixture",
                                  problem != "review", target)
    assert not target.exists()
