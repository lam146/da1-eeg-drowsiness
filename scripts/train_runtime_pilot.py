"""Train a provisional C++ JSON model with the runtime's FFT features."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_pilot_model import FEATURES
from n16_eeg.blink import blink_window_features
from n16_eeg.model import evaluate_loso, fit_logistic_model
from n16_eeg.signal_features import bandpower_features


FS = 512


def run(pilot, output):
    source = pd.read_csv(pilot / "all_subjects_features_PROVISIONAL.csv")
    manifest = json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    recordings = {(entry["subject_id"], entry["session_id"]): entry
                  for entry in manifest["sessions"]}
    if (set(source.dataset_id) != {"openvibe"} or set(source.signal_unit) != {"count"}
            or set(source.fs_hz) != {FS} or not source.time_alignment_verified.eq(False).all()
            or not source.source_is_synthetic.eq(False).all()):
        raise ValueError("Expected the reviewed provisional 512 Hz OpenViBE dataset")
    rows = []
    for key, group in source.groupby(["subject_id", "session_id"], sort=False):
        recording = recordings[key]
        path = Path(recording["source"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != recording["source_sha256"] or set(group.source_file) != {path.name}:
            raise ValueError(f"Raw file no longer matches the screening audit: {path.name}")
        raw = pd.read_csv(path, usecols=["Electrode"]).Electrode.to_numpy(float)
        for row in group.itertuples(index=False):
            start = round(row.window_start_ms * FS / 1000)
            end = round(row.window_end_ms * FS / 1000)
            if not 0 <= start < end <= len(raw) or end - start != 4 * FS:
                raise ValueError(f"Invalid 4-second window in {path.name}")
            values = raw[start:end]
            features = {**bandpower_features(values, FS), **blink_window_features(values, FS)}
            rows.append(dict(subject_id=row.subject_id, session_id=row.session_id,
                             window_start_ms=row.window_start_ms, window_end_ms=row.window_end_ms,
                             label=row.label, **{name: features[name] for name in FEATURES}))
    frame = pd.DataFrame(rows)
    x = frame[FEATURES].to_numpy(float)
    labels = frame.label.to_numpy(int)
    groups = frame.subject_id.to_numpy(str)
    folds = evaluate_loso(x, labels, groups, FEATURES)
    model = fit_logistic_model(x, labels, FEATURES)
    result = dict(status="PROVISIONAL_OFFLINE_NOT_VALIDATED_FOR_LIVE_USE",
                  dataset="OpenViBE raw counts with provisional KSS",
                  source=str(pilot.resolve()), samples=len(frame), feature_branch="runtime_fft",
                  folds=[asdict(fold) for fold in folds],
                  mean_balanced_accuracy=float(np.mean([fold.balanced_accuracy for fold in folds])))
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "features_runtime_PROVISIONAL.csv", index=False)
    model.save(output / "runtime_model_PROVISIONAL.json")
    (output / "metrics_PROVISIONAL.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.pilot, args.output), indent=2))
