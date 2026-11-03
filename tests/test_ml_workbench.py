import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from n16_eeg import ml_workbench as work


def raw_session(seconds=16):
    time = np.arange(seconds * 512) / 512
    return pd.DataFrame({"Time:512Hz": time, "Electrode": np.sin(2 * np.pi * 10 * time)})


def intervals():
    return pd.DataFrame({"start_s": [0, 8], "end_s": [8, 16], "kss": [3, 8],
                         "quality_status": ["usable", "usable"]})


def test_csv_sniff_semicolon_preserves_leading_zero_ids(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame({"subject_id": ["001", "002"], "label": [0, 1], "theta_alpha": [1, 2]}).to_csv(path, sep=";", index=False)
    frame = work.read_csv_file(path)
    assert frame["subject_id"].tolist() == ["001", "002"]
    assert work.file_kind(frame) == "features"
    assert work.feature_candidates(frame) == ["theta_alpha"]


def test_duplicate_headers_and_empty_csv_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        work.read_csv_file(path)
    path.write_text("start_s,end_s,kss,quality_status\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        work.read_csv_file(path)


def test_inspect_raw_and_duration():
    result = work.inspect_frame(raw_session())
    assert result["kind"] == "openvibe_raw"
    assert result["duration_s"] == 16
    assert result["rows"] == 8192
    assert work.feature_candidates(raw_session()) == []


def test_prepare_only_full_windows_inside_reviewed_intervals():
    result, report = work.prepare_features(raw_session(), intervals(), "S01", "P01", "uV", True, "review_v1")
    assert len(result) == 10  # 5 windows in each 8-second region; 3 crossing windows excluded.
    assert result["label"].tolist() == [0] * 5 + [1] * 5
    assert set(result["dataset_id"]) == {"openvibe"}
    assert "poor_signal" not in result  # Do not invent a manufacturer measurement.
    assert report["counts"]["candidate_windows"] == 13
    assert report["counts"]["no_usable_interval"] == 3
    assert report["counts"]["kept"] == 10
    assert "no forward-fill" in report["label_policy"]


@pytest.mark.parametrize("problem", ["review", "protocol", "unit", "person", "time_gap", "nan", "overlap", "fractional_kss"])
def test_invalid_raw_or_labels_rejected(problem):
    raw, ann = raw_session(), intervals()
    args = ["S01", "P01", "uV", True, "v1"]
    if problem == "review": args[3] = False
    if problem == "protocol": args[4] = ""
    if problem == "unit": args[2] = "unknown"
    if problem == "person": args[0] = ""
    if problem == "time_gap": raw.loc[100, "Time:512Hz"] += 1
    if problem == "nan": raw.loc[100, "Electrode"] = np.nan
    if problem == "overlap": ann.loc[1, "start_s"] = 7
    if problem == "fractional_kss":
        ann["kss"] = ann["kss"].astype(float)
        ann.loc[0, "kss"] = 3.5
    with pytest.raises(ValueError):
        work.prepare_features(raw, ann, *args)


def test_kss6_excluded_unknown_quality_not_used():
    ann = intervals()
    ann.loc[0, "kss"] = 6
    result, report = work.prepare_features(raw_session(), ann, "S01", "P01", "count", True, "v1")
    assert len(result) == 5 and set(result.label) == {1}
    assert report["counts"]["kss_transition"] == 5
    ann.loc[1, "quality_status"] = "unknown"
    with pytest.raises(ValueError, match="Không có"):
        work.prepare_features(raw_session(), ann, "S01", "P01", "count", True, "v1")


def test_explicit_gap_in_labels_not_bridged():
    ann = pd.DataFrame({"start_s": [0, 7, 9], "end_s": [7, 9, 16],
                        "kss": [3, np.nan, 8], "quality_status": ["usable", "exclude", "usable"]})
    result, _ = work.prepare_features(raw_session(), ann, "S01", "P01", "uV", True, "v1")
    assert all((result.window_end_ms <= 7000) | (result.window_start_ms >= 9000))


def test_existing_device_quality_overrides_manual_usable():
    raw = raw_session()
    raw["poor_signal"] = 200
    with pytest.raises(ValueError, match="Không có"):
        work.prepare_features(raw, intervals(), "S01", "P01", "count", True, "v1")


def test_canonical_origin_units_and_ids_not_silently_overwritten():
    source = raw_session().rename(columns={"Electrode": "raw_eeg", "Time:512Hz": "timestamp_ms"})
    source["timestamp_ms"] = source["timestamp_ms"] * 1000 + 100000
    source["subject_id"] = "S01"; source["session_id"] = "P01"
    source["signal_unit"] = "count"
    time, _, _ = work.raw_arrays(source)
    assert time[0] == 0
    with pytest.raises(ValueError, match="signal_unit"):
        work.prepare_features(source, intervals(), "S01", "P01", "uV", True, "v1")
    with pytest.raises(ValueError, match="subject_id"):
        work.prepare_features(source, intervals(), "S02", "P01", "count", True, "v1")


def test_prepare_synthetic_provenance_preserved():
    source = raw_session()
    source["is_synthetic"] = True
    result, _ = work.prepare_features(source, intervals(), "S01", "P01", "uV", True, "v1")
    assert result["source_is_synthetic"].all()


def test_merge_duplicate_file_and_duplicate_window_rejected(tmp_path):
    result, _ = work.prepare_features(raw_session(), intervals(), "S01", "P01", "uV", True, "v1")
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    result.to_csv(one, index=False); result.to_csv(two, index=False)
    with pytest.raises(ValueError): work.merge_features([one, one])
    with pytest.raises(ValueError): work.merge_features([one, two])


def test_prepare_save_and_training_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    sources = []
    for i in range(3):
        raw, ann = raw_session(), intervals()
        raw_path, ann_path = tmp_path / f"raw{i}.csv", tmp_path / f"ann{i}.csv"
        # This is a software fixture, not human EEG.
        raw["is_synthetic"] = True
        raw.to_csv(raw_path, index=False); ann.to_csv(ann_path, index=False)
        result, report = work.prepare_features(raw, ann, f"S{i}", "P1", "uV", True, "fixture")
        out = work.save_prepared(result, report, tmp_path, raw_path, ann_path)
        sources.append(out / "features.csv")
        assert len(json.loads((out / "preparation.json").read_text(encoding="utf-8"))["inputs"][0]["sha256"]) == 64
    directory, report = work.train_files(sources, ["theta_alpha", "blink_count"], "openvibe", "logistic", tmp_path)
    assert "SYNTHETIC" in report["warning"]
    assert len(report["folds"]) == 3
    assert {p.name for p in directory.iterdir()} == {"pipeline.joblib", "metrics.json", "oof_predictions.csv", "inputs.json"}
    again, _ = work.train_files(sources, ["theta_alpha"], "openvibe", "logistic", tmp_path)
    assert again != directory  # Unique outputs even within the same second.


def test_real_user_pilot_can_be_inspected_without_training():
    path = Path("raw/test1.csv")
    if not path.exists():
        pytest.skip("Local pilot not distributed")
    frame = work.read_csv_file(path)
    report = work.inspect_frame(frame)
    assert report["kind"] == "openvibe_raw"
    assert report["rows"] > 0
    assert work.feature_candidates(frame) == []
