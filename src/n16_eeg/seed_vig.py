"""Import authorized SEED-VIG MAT features using an explicit session manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_mat(path: Path):
    from scipy.io import whosmat
    return [{"key": name, "shape": list(shape), "type": kind}
            for name, shape, kind in whosmat(path)]


def convert_manifest(path: Path, alert_max: float, drowsy_min: float):
    from scipy.io import loadmat
    if not 0 <= alert_max < drowsy_min <= 1:
        raise ValueError("Require 0 <= alert_max < drowsy_min <= 1 (not KSS thresholds)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "seed-vig" or not manifest.get("sessions"):
        raise ValueError("Manifest needs dataset=seed-vig and nonempty sessions")
    synthetic = manifest.get("source_is_synthetic", False)
    if type(synthetic) is not bool:
        raise ValueError("source_is_synthetic must be an explicit boolean when supplied")
    frames, names, schema, seen = [], None, None, set()
    provenance = []
    for item in manifest["sessions"]:
        subject = str(item["subject_id"]).strip()
        session = str(item["session_id"]).strip()
        if (not subject or not session or "REPLACE" in subject or "REPLACE" in session
                or (subject, session) in seen):
            raise ValueError("Require identified, unique person/session pairs; never infer ID from filename")
        seen.add((subject, session))
        if item.get("alignment_verified") is not True:
            raise ValueError("Verify feature/label temporal alignment and set alignment_verified=true")
        source = (path.parent / item["feature_file"]).resolve()
        label_path = (path.parent / item["label_file"]).resolve()
        key, label_key = item["feature_key"], item["label_key"]
        tensor = np.asarray(loadmat(source)[key], dtype=float)
        axis = item["sample_axis"]
        if type(axis) is not int or axis < 0 or axis >= tensor.ndim:
            raise ValueError("sample_axis must be an explicit zero-based tensor axis")
        data = np.moveaxis(tensor, axis, 0)
        # Require identical field and shape across sessions, not just equal flattened length.
        current_schema = (key, axis, data.shape[1:])
        if schema is not None and current_schema != schema:
            raise ValueError("Feature fields/layout differ across sessions")
        schema = current_schema
        labels = np.asarray(loadmat(label_path)[label_key], dtype=float)
        if labels.ndim > 2 or (labels.ndim == 2 and min(labels.shape) != 1):
            raise ValueError("PERCLOS must be a vector, not an arbitrary matrix")
        labels = labels.reshape(-1)
        if len(labels) != len(data) or not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
            raise ValueError("Labels must align one-to-one with feature rows and lie in [0,1]")
        flat = data.reshape(len(data), -1)
        if not np.isfinite(flat).all():
            raise ValueError("Features contain NaN/Inf")
        names = [f"seed_{key}_{j:04d}" for j in range(flat.shape[1])]
        frame = pd.DataFrame(flat, columns=names)
        frame.insert(0, "feature_row", np.arange(len(frame)))
        frame.insert(0, "session_id", session)
        frame.insert(0, "subject_id", subject)
        frame["perclos"] = labels
        keep = (labels <= alert_max) | (labels >= drowsy_min)
        frame["label"] = (labels >= drowsy_min).astype(int)
        frame["dataset_id"] = "seed-vig"
        frame["source_is_synthetic"] = synthetic
        frames.append(frame.loc[keep])
        provenance.append({**item, "rows_total": len(labels), "rows_kept": int(keep.sum()),
                           "rows_excluded_ambiguous": int((~keep).sum())})
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("No rows retained with these thresholds")
    return result, {
        "dataset": "seed-vig", "feature_names": names,
        "label_policy": {"alert_max": alert_max, "drowsy_min": drowsy_min,
                         "meaning": "User-defined PERCLOS binary mapping, NOT KSS"},
        "sessions": provenance,
        "warning": ("SYNTHETIC: software tests only. " if synthetic else "")
                   + "Precomputed features; verify smoothing causality. Not raw 512 Hz or a live NeuroSky model.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("file", type=Path)
    convert = sub.add_parser("convert")
    convert.add_argument("manifest", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--alert-max", type=float, required=True)
    convert.add_argument("--drowsy-min", type=float, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_mat(args.file), indent=2))
    else:
        sidecar = args.output.with_suffix(".manifest.json")
        if args.output.exists() or sidecar.exists():
            parser.error("Output exists; choose a new filename")
        frame, provenance = convert_manifest(args.manifest, args.alert_max, args.drowsy_min)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        sidecar.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(frame)} rows. Explicit feature names are in {sidecar}")


if __name__ == "__main__":
    main()
