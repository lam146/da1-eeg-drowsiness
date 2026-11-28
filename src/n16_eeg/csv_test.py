"""Run the live model on a single raw EEG CSV without opening a COM port."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Event

import numpy as np

from .live import LiveDetector


def raw_artifacts(values, segments, fs):
    """Flag expected ADC rails and constant runs within a continuous segment."""
    signal = np.asarray(values)
    flags = (signal <= -2048) | (signal >= 2047)
    changes = (np.diff(signal) != 0) | (np.diff(segments) != 0)
    edges = np.r_[0, np.flatnonzero(changes) + 1, len(signal)]
    for index in np.flatnonzero(np.diff(edges) >= fs // 4):
        flags[edges[index]:edges[index + 1]] = True
    return np.r_[0, np.cumsum(flags)]


def test_recording(source: Path, output: Path, detector: LiveDetector, stop: Event, *,
                   screen_raw: bool | None = None) -> dict | None:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.readline()
        if not sample:
            raise ValueError("CSV trống.")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error as exc:
            raise ValueError("Không xác định được dấu phân cách CSV.") from exc
        handle.seek(0)
        reader = csv.DictReader(handle, dialect=dialect)
        fields = set(reader.fieldnames or ())
        if len(fields) != len(reader.fieldnames or ()):
            raise ValueError("CSV có tên cột trùng.")
        openvibe = {"Time:512Hz", "Electrode"} <= fields
        signal = "Electrode" if openvibe else "raw_eeg"
        clock = "Time:512Hz" if openvibe else (
            "signal_time_ms" if "signal_time_ms" in fields else "timestamp_ms"
        )
        if not {signal, clock} <= fields:
            raise ValueError("Cần CSV raw OpenViBE hoặc N16 có cột thời gian và EEG.")
        use_quality = "poor_signal" in fields
        identities = [name for name in ("subject_id", "session_id", "signal_unit") if name in fields]
        metadata = None
        values, times, qualities, segments = [], [], [], []
        for line, row in enumerate(reader, 2):
            if stop.is_set():
                return None
            try:
                identity = tuple(row[name].strip() for name in identities)
                if metadata is None:
                    metadata = identity
                if identity != metadata or "" in identity:
                    raise ValueError("CSV phải chứa một phiên với thông tin thống nhất")
                if "signal_unit" in fields and row["signal_unit"].strip().lower() != "count":
                    raise ValueError("mô hình Main nhận raw theo count")
                if "fs_hz" in fields and float(row["fs_hz"]) != detector.fs:
                    raise ValueError("tần số phải là 512 Hz")
                values.append(float(row[signal]))
                times.append(float(row[clock]) * (1000 if openvibe else 1))
                qualities.append(float(row["poor_signal"] or 255) if use_quality else 0.)
                segments.append(int(row["raw_segment"]) if "raw_segment" in fields else 0)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"Dòng {line} có dữ liệu không hợp lệ. {exc}") from exc

    if len(values) < detector.window_samples:
        raise ValueError("Cần ít nhất 3 giây EEG ở 512 Hz.")
    if not all(np.isfinite(series).all() for series in (values, times, qualities)):
        raise ValueError("EEG, thời gian và chất lượng phải hữu hạn.")
    if min(qualities) < 0 or max(qualities) > 255 or min(segments) < 0:
        raise ValueError("Giá trị chất lượng hoặc mã đoạn nằm ngoài phạm vi.")
    offsets = np.asarray(times) - times[0]
    nominal = np.arange(len(values)) * (1000 / detector.fs)
    if np.any(np.diff(offsets) <= 0) or not np.allclose(offsets, nominal, atol=1.01, rtol=0):
        raise ValueError("Trục thời gian không khớp lưới 512 Hz. CSV bộ ghi N16 cần cột signal_time_ms.")
    if np.any(np.diff(segments) < 0):
        raise ValueError("Mã đoạn raw_segment phải tăng dần theo thời gian.")

    detector.raw_guard.screen_raw = detector.screen_raw if screen_raw is None else screen_raw
    detector.reset()
    previous_segment = segments[0]
    was_alarming = False
    rows = []
    scores = []
    episodes = 0
    for index, (value, quality, segment) in enumerate(zip(values, qualities, segments)):
        if stop.is_set():
            return None
        if segment != previous_segment:
            detector.reset()
            was_alarming = False
            previous_segment = segment
        decision = detector.add_sample(value, quality)
        if decision is None:
            continue
        alarming = bool(decision["alarm"])
        episodes += int(alarming and not was_alarming)
        was_alarming = alarming
        if decision["risk"] != "":
            scores.append(decision["risk"])
        decision.pop("timestamp_ms")
        decision.setdefault("raw_artifact", 0)
        if not use_quality:
            decision["poor_signal_fraction"] = ""
        rows.append({"window_end_s": round(float(nominal[index]) / 1000, 6),
                     "sample_index": index, "raw_segment": segment, **decision})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        columns = ("window_end_s", "sample_index", "raw_segment", "state", "risk", "alarm",
                   "poor_signal_fraction", "theta_alpha", "blink_count", "long_blink_count", "latency_ms",
                   "raw_artifact")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return dict(windows=len(scores), poor_windows=len(rows) - len(scores),
                alarm_windows=sum(row["alarm"] for row in rows), episodes=episodes,
                max_risk=max(scores, default=None), samples=len(values), device_quality=use_quality,
                artifact_windows=sum(row["raw_artifact"] for row in rows))
