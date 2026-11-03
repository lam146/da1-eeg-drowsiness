"""GUI-independent CSV preparation and audited offline ML runs."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .blink import blink_window_features
from .featurize import kss_to_label
from .signal_features import bandpower_features
from .sklearn_train import FORBIDDEN_FEATURES, evaluate_frame, validate_frame


def read_csv_file(path: Path) -> pd.DataFrame:
    """Preserve subject IDs (including leading zeroes); never execute CSV content."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
    if not sample.strip():
        raise ValueError("File CSV trống.")
    try:
        separator = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error as exc:
        raise ValueError("Không xác định được dấu phân cách CSV (phẩy, chấm phẩy hoặc tab).") from exc
    header = next(csv.reader(sample.splitlines(), delimiter=separator))
    if len(header) != len(set(header)):
        raise ValueError("CSV có tên cột trùng nhau; cần sửa bản sao và giữ nguyên bản gốc.")
    frame = pd.read_csv(path, sep=separator, encoding="utf-8-sig",
                        dtype={"subject_id": "string", "session_id": "string"})
    if frame.empty:
        raise ValueError("CSV chỉ có header, chưa có dòng dữ liệu.")
    return frame


def file_kind(frame: pd.DataFrame) -> str:
    if {"Time:512Hz", "Electrode"} <= set(frame):
        return "openvibe_raw"
    if "raw_eeg" in frame:
        return "canonical_raw"
    if {"subject_id", "label"} <= set(frame):
        return "features"
    return "unknown"


def raw_arrays(frame: pd.DataFrame):
    kind = file_kind(frame)
    if kind == "openvibe_raw":
        time = frame["Time:512Hz"].to_numpy(float)
        signal = frame["Electrode"].to_numpy(float)
        dataset = "openvibe"
    elif kind == "canonical_raw":
        time_key = "signal_time_ms" if "signal_time_ms" in frame else "timestamp_ms"
        if time_key not in frame:
            raise ValueError("CSV raw cần timestamp_ms hoặc signal_time_ms.")
        time = frame[time_key].to_numpy(float) / 1000
        signal = frame["raw_eeg"].to_numpy(float)
        dataset = "neurosky"
        if "dataset_id" in frame:
            values = frame["dataset_id"].dropna().astype(str).unique()
            if len(values) != 1 or values[0] not in {"neurosky", "openvibe"}:
                raise ValueError("Nguồn raw thiếu/không đồng nhất; không trộn SEED-VIG vào raw.")
            dataset = values[0]
        for name in ("subject_id", "session_id", "signal_unit", "fs_hz"):
            if name in frame and (frame[name].isna().any() or frame[name].nunique() != 1):
                raise ValueError(f"Cột {name} thiếu hoặc không đồng nhất trong một phiên.")
        if "fs_hz" in frame and float(frame["fs_hz"].iloc[0]) != 512:
            raise ValueError("Bộ tạo feature này chỉ nhận dữ liệu 512 Hz.")
        if "sample_index" in frame and not np.all(np.diff(frame["sample_index"].to_numpy(float)) == 1):
            raise ValueError("sample_index bị đứt/trùng. Hãy kiểm tra và tách đoạn liên tục.")
    else:
        raise ValueError("Không phải CSV raw OpenViBE/N16 được hỗ trợ.")
    if len(signal) < 2 or not np.isfinite(time).all() or not np.isfinite(signal).all():
        raise ValueError("Raw hoặc thời gian chứa ô trống/NaN/Inf hoặc có quá ít mẫu.")
    if not np.allclose(np.diff(time), 1 / 512, rtol=.05, atol=1e-6):
        raise ValueError("Thời gian không liên tục gần 512 Hz. Không tự vá khoảng mất mẫu.")
    return time - time[0], signal, dataset


def inspect_frame(frame: pd.DataFrame) -> dict:
    kind = file_kind(frame)
    result = {"kind": kind, "rows": len(frame), "columns": len(frame.columns),
              "missing_cells": int(frame.isna().sum().sum())}
    if kind in {"openvibe_raw", "canonical_raw"}:
        time, signal, dataset = raw_arrays(frame)
        result.update(dataset=dataset, nominal_fs_hz=512,
                      duration_s=float(time[-1] + 1 / 512),
                      minimum=float(signal.min()), maximum=float(signal.max()),
                      note="Đủ cấu trúc để kiểm tra; chưa xác nhận tín hiệu sạch hoặc nhãn.")
    elif kind == "features":
        result.update(subjects=int(frame["subject_id"].nunique()),
                      label_counts={str(k): int(v) for k, v in frame["label"].value_counts(dropna=False).items()},
                      note="Chỉ huấn luyện khi chọn đúng đặc trưng, nguồn và nhãn hợp lệ.")
    else:
        result["note"] = "Chưa nhận dạng. Xem lại header; không tự đoán nhãn hoặc cột tín hiệu."
    return result


def validate_intervals(frame: pd.DataFrame, duration: float) -> pd.DataFrame:
    required = {"start_s", "end_s", "kss", "quality_status"}
    if not required <= set(frame):
        raise ValueError("File nhãn cần start_s,end_s,kss,quality_status.")
    rows = frame.copy()
    for key in ("start_s", "end_s"):
        rows[key] = pd.to_numeric(rows[key], errors="raise")
    rows["kss"] = pd.to_numeric(rows["kss"], errors="coerce")
    rows["quality_status"] = rows["quality_status"].astype(str).str.strip().str.lower()
    bounds = rows[["start_s", "end_s"]].to_numpy(float)
    if (not np.isfinite(bounds).all() or np.any(bounds[:, 0] < 0)
            or np.any(bounds[:, 1] > duration + 1e-6) or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("Khoảng nhãn sai thời gian: cần 0 <= start_s < end_s <= thời lượng raw.")
    if not rows["quality_status"].isin(["usable", "exclude", "unknown"]).all():
        raise ValueError("quality_status chỉ nhận usable, exclude hoặc unknown.")
    usable = rows["quality_status"] == "usable"
    values = rows.loc[usable, "kss"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any((values < 1) | (values > 9)):
        raise ValueError("Khoảng usable phải có KSS nguyên 1-9 được ghi thực tế.")
    rows = rows.sort_values("start_s", kind="stable").reset_index(drop=True)
    if len(rows) > 1 and np.any(rows["start_s"].to_numpy()[1:] < rows["end_s"].to_numpy()[:-1] - 1e-9):
        raise ValueError("Các khoảng nhãn đang chồng nhau. Tách vùng rõ ràng trước tạo đặc trưng.")
    return rows


def prepare_features(frame: pd.DataFrame, intervals: pd.DataFrame, subject: str, session: str,
                     signal_unit: str, reviewed: bool, protocol: str,
                     window_s: float = 4, step_s: float = 1):
    """Only windows completely inside explicit, manually verified intervals; no KSS ffill."""
    if not reviewed or not protocol.strip():
        raise ValueError("Cần xác nhận đã đối chiếu nhãn/chất lượng và ghi tên quy tắc áp dụng.")
    subject, session = subject.strip(), session.strip()
    if not subject or not session or signal_unit not in {"count", "uV"}:
        raise ValueError("Cần mã người, mã phiên và đơn vị count/uV đã xác nhận.")
    for name, expected in (("subject_id", subject), ("session_id", session), ("signal_unit", signal_unit)):
        if name in frame and (frame[name].isna().any() or set(frame[name].astype(str)) != {expected}):
            raise ValueError(f"{name} đã có trong raw nhưng khác giá trị nhập; không tự đổi nguồn gốc.")
    if not (np.isfinite([window_s, step_s]).all() and window_s >= 1 and step_s > 0):
        raise ValueError("Cửa sổ phải >=1 giây, bước phải >0.")
    n, hop = int(round(window_s * 512)), int(round(step_s * 512))
    if hop < 1:
        raise ValueError("Bước quá nhỏ.")
    time, signal, dataset = raw_arrays(frame)
    duration = float(time[-1] + 1 / 512)
    intervals = validate_intervals(intervals, duration)
    poor = (pd.to_numeric(frame["poor_signal"], errors="coerce").to_numpy(float)
            if "poor_signal" in frame else np.full(len(frame), np.nan))
    synthetic = False
    for name in ("is_synthetic", "source_is_synthetic"):
        if name in frame:
            flags = frame[name].astype(str).str.lower()
            if not flags.isin(["true", "false", "0", "1"]).all():
                raise ValueError("Cờ dữ liệu mô phỏng không hợp lệ.")
            synthetic |= bool(flags.isin(["true", "1"]).any())
    records = []
    counts = dict(candidate_windows=0, kept=0, no_usable_interval=0, kss_transition=0,
                  device_quality_rejected=0)
    for start in range(0, len(signal) - n + 1, hop):
        counts["candidate_windows"] += 1
        left, right = float(time[start]), float(time[start + n - 1] + 1 / 512)
        fits = intervals[(intervals["start_s"] <= left + 1e-9)
                         & (intervals["end_s"] >= right - 1e-9)
                         & (intervals["quality_status"] == "usable")]
        if fits.empty:
            counts["no_usable_interval"] += 1
            continue
        label = kss_to_label(float(fits.iloc[0]["kss"]))
        if label is None:
            counts["kss_transition"] += 1
            continue
        device_values = poor[start:start + n]
        if np.any(np.isfinite(device_values)):
            if (not np.isfinite(device_values).all() or np.any((device_values < 0) | (device_values > 255))
                    or np.mean(device_values > 50) > .2):
                counts["device_quality_rejected"] += 1
                continue
        samples = signal[start:start + n]
        record = dict(subject_id=subject, session_id=session, window_start_ms=left * 1000,
                      window_end_ms=right * 1000, label=label, dataset_id=dataset,
                      signal_unit=signal_unit, fs_hz=512, source_is_synthetic=synthetic)
        record.update(bandpower_features(samples, 512))
        record.update(blink_window_features(samples, 512))
        records.append(record)
    counts["kept"] = len(records)
    if not records:
        raise ValueError("Không có cửa sổ đủ nhãn/chất lượng. Kiểm tra khoảng nhãn, KSS=6 và độ dài >=cửa sổ.")
    result = pd.DataFrame(records)
    provenance = dict(dataset=dataset, signal_unit=signal_unit, fs_hz=512,
        window_s=n / 512, step_s=hop / 512, subject_id=subject, session_id=session,
        protocol=protocol.strip(), quality_source="Explicit reviewed intervals; device poorSignal checked when present",
        label_policy="KSS 1-5=0, 7-9=1, 6 excluded; full window in one usable interval; no forward-fill",
        time_origin="seconds since first raw sample", counts=counts,
        warning="Manual quality is NOT a generated device poorSignal; verify protocol independently",
        feature_names=[c for c in result if c not in FORBIDDEN_FEATURES])
    return result, provenance


def feature_candidates(frame: pd.DataFrame) -> list[str]:
    # Never suggest raw samples or manufacturer power values as already-computed N16 features.
    if file_kind(frame) != "features":
        return []
    known = set(bandpower_features(np.zeros(2048), 512)) | set(blink_window_features(np.zeros(2048), 512))
    return [c for c in frame.columns if c in known or c.startswith("seed_")]


def merge_features(paths: list[Path]):
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Chưa chọn file hoặc có file bị chọn trùng.")
    frames = []
    for path in paths:
        frame = read_csv_file(path)
        if file_kind(frame) != "features":
            raise ValueError(f"{path.name}: cần bảng đặc trưng có subject_id,label, không phải raw.")
        if frames and set(frame.columns) != set(frames[0].columns):
            raise ValueError("Các file khác bộ cột. Không tự ghép thiếu cột thành NaN.")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated().any():
        raise ValueError("Có hàng dữ liệu trùng hoàn toàn; kiểm tra file bị nạp lặp.")
    if {"subject_id", "session_id", "window_start_ms"} <= set(merged):
        if merged.duplicated(["subject_id", "session_id", "window_start_ms"]).any():
            raise ValueError("Trùng người/phiên/mốc cửa sổ giữa các file; kiểm tra trước huấn luyện.")
    return merged


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    return path


def save_prepared(frame, provenance, output_parent, raw_path, annotations_path):
    directory = new_run_dir(Path(output_parent), "features")
    frame.to_csv(directory / "features.csv", index=False)
    provenance = {**provenance, "inputs": [dict(path=str(p.resolve()), sha256=sha256(p))
                                         for p in (raw_path, annotations_path)]}
    (directory / "preparation.json").write_text(json.dumps(provenance, ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding="utf-8")
    return directory


def train_files(paths, features, dataset, classifier, parent):
    import joblib
    frame = merge_features(paths)
    validate_frame(frame, features, dataset)
    final, report, predictions = evaluate_frame(frame, features, dataset, classifier)
    directory = new_run_dir(Path(parent), "training")
    joblib.dump(final, directory / "pipeline.joblib")
    (directory / "metrics.json").write_text(json.dumps(report, ensure_ascii=False,
                               indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(directory / "oof_predictions.csv", index=False)
    (directory / "inputs.json").write_text(json.dumps({"inputs": [
        dict(path=str(path.resolve()), sha256=sha256(path)) for path in paths],
        "warning": "Offline model. Do not load untrusted joblib. Not C++ runtime JSON."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return directory, report


def import_seed_manifest(manifest_path: Path, alert_max: float, drowsy_min: float,
                         policy: str, reviewed: bool, parent: Path):
    """Import licensed local files; record mapping and inputs without downloading data."""
    from .seed_vig import convert_manifest

    if not reviewed or not policy.strip():
        raise ValueError("Cần xác nhận quyền sử dụng, căn nhãn và ghi quy tắc ngưỡng PERCLOS trước đánh giá.")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Validate first; malformed configurations must not produce an output directory.
    frame, report = convert_manifest(manifest_path, alert_max, drowsy_min)
    inputs = [manifest_path]
    for item in manifest["sessions"]:
        inputs.extend((manifest_path.parent / item[key]).resolve()
                      for key in ("feature_file", "label_file"))
    inputs = list(dict.fromkeys(inputs))
    audit = {**report, "policy": policy.strip(), "rows": len(frame),
             "subjects": int(frame["subject_id"].nunique()),
             "manifest_path": str(manifest_path), "manifest_snapshot": manifest,
             "inputs": [dict(path=str(path), sha256=sha256(path)) for path in inputs],
             "scope": "SEED-VIG only; do not merge with NeuroSky/KSS or export as a live device model."}
    serialized = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)
    directory = new_run_dir(Path(parent), "seed_vig")
    frame.to_csv(directory / "features.csv", index=False)
    (directory / "seed_import.json").write_text(serialized, encoding="utf-8")
    return directory, audit
