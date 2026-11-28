"""Retrain the N16 Windows model using the reviewed pilot and a session manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_n16_systems import train
from evaluate_python_alert_latency import FEATURES, FS, feature_vector
from prepare_two_subjects_pilot import binary, intervals, note_data

from n16_eeg.csv_test import raw_artifacts
from n16_eeg.ml_workbench import read_csv_file, validate_intervals
from n16_eeg.model import LogisticModel

WINDOW = 3 * FS
STRIDE = 4 * FS  # Retain the cadence of the original training windows.
KEY = ["subject_id", "session_id"]
TIME = "Time:512Hz"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(path):
    raw = pd.read_csv(path, usecols=[TIME, "Electrode"])
    times = raw[TIME].to_numpy(float)
    values = raw.Electrode.to_numpy(float)
    if (len(values) < WINDOW or not np.isfinite(raw.to_numpy()).all()
            or not np.allclose(np.diff(times), 1 / FS, rtol=0, atol=1e-9)):
        raise ValueError(f"Expected continuous 512 Hz raw EEG in {path}")
    return values


def make_features(subject, session, starts, labels, raw, basis):
    return pd.DataFrame(
        [(subject, session, int(start), int(label), basis,
          *feature_vector(raw[start:start + WINDOW])) for start, label in zip(starts, labels)],
        columns=[*KEY, "start_sample", "label", "label_basis", *FEATURES],
    )


def pilot_features(raw_dir, pilot_dir):
    sources = json.loads((pilot_dir / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    hashes = {(item["subject_id"], item["session_id"]): item["source_sha256"]
              for item in sources["sessions"]}
    frame = pd.read_csv(pilot_dir / "all_subjects_features_PROVISIONAL.csv")
    for (subject, session), rows in frame.groupby(KEY):
        path = raw_dir / subject / rows.source_file.unique().item()
        digest = sha256(path)
        if digest != hashes[subject, session]:
            raise ValueError(f"Pilot raw recording changed: {path}")
        raw = read_raw(path)
        starts = np.rint(rows.window_end_ms.to_numpy() * FS / 1000).astype(int) - WINDOW
        if starts.min() < 0 or starts.max() + WINDOW > len(raw):
            raise ValueError(f"Pilot feature window outside {path}")
        yield make_features(subject, session, starts, rows.label, raw, "nominal_kss"), digest


def recording_labels(raw_dir, item, path, digest, raw_length, starts, stops):
    labels = np.full(len(starts), -1, dtype=int)
    if "note" in item:
        parent = Path(item["note"]).parent.name.upper()
        if parent.startswith("S") and parent[1:].isdigit() and parent != path.parent.name.upper():
            raise ValueError(f"KSS note belongs to {parent}, recording is in {path.parent.name}: {path}")
    if "annotations" in item:
        annotation = raw_dir.parents[1] / item["annotations"]
        if digest != item["recording_sha256"] or sha256(annotation) != item["annotations_sha256"]:
            raise ValueError(f"Recording or KSS intervals changed: {path}")
        spans = validate_intervals(read_csv_file(annotation), raw_length / FS)
        for span in spans.loc[spans.quality_status.eq("usable")].itertuples(index=False):
            label = binary(span.kss)
            covered = (starts >= span.start_s * FS - 1e-6) & (stops <= span.end_s * FS + 1e-6)
            if label is not None:
                labels[covered] = label
        return labels, "confirmed_kss_interval"
    if "note" in item:
        if digest != item["recording_sha256"]:
            raise ValueError(f"Recording associated with the KSS note changed: {path}")
        _, points = note_data(raw_dir.parent / item["note"])
        for span in intervals(points, raw_length / FS):
            covered = (starts >= span["start_s"] * FS) & (stops <= span["end_s"] * FS)
            labels[covered] = span["label"]
        return labels, "nominal_kss"
    if item.get("session_label") not in (0, 1):
        raise ValueError(f"Missing binary session label or KSS intervals for {path}")
    return np.full(len(starts), item["session_label"], dtype=int), "provisional_session_label"


def additional_features(raw_dir, manifest, prior_hashes):
    for item in manifest:
        path = raw_dir / item["file"]
        subject, session = path.parent.name, path.stem
        digest = sha256(path)
        if digest in prior_hashes:
            raise ValueError(f"Duplicate raw recording: {path}")
        prior_hashes.add(digest)
        raw = read_raw(path)
        stops = np.arange(STRIDE, len(raw) + 1, STRIDE)
        starts = stops - WINDOW
        artifacts = raw_artifacts(raw, np.zeros(len(raw), dtype=int), FS)
        clean = artifacts[starts] == artifacts[stops]
        labels, basis = recording_labels(raw_dir, item, path, digest, len(raw), starts, stops)
        kept = clean & (labels >= 0)
        if not kept.any():
            raise ValueError(f"No usable labeled windows in {path}")
        audit = dict(**item, sha256=digest, duration_s=len(raw) / FS, subject_id=subject,
                     label_basis=basis, artifact_windows=int(np.count_nonzero(~clean)),
                     unlabeled_clean_windows=int(np.count_nonzero(clean & (labels < 0))),
                     kept_windows=int(np.count_nonzero(kept)))
        print(f"{session}: {audit['kept_windows']} usable windows", flush=True)
        yield make_features(subject, session, starts[kept], labels[kept], raw, basis), audit


def evaluate(pilot, combined, threshold):
    folds, predictions = [], []
    for subject, test in combined.groupby("subject_id"):
        test = test.copy()
        for variant, dataset in (("previous_data", pilot), ("updated_data", combined)):
            training = dataset.loc[dataset.subject_id.ne(subject)]
            model = train(training, "logistic_combined")
            score = model.predict_proba(test[FEATURES].to_numpy())
            predicted = (score >= threshold).astype(int)
            test[f"{variant}_score"] = score
            test[f"{variant}_prediction"] = predicted
            recall = {}
            for label in sorted(test.label.unique()):
                in_class = test.label.eq(label)
                recall[str(label)] = float(np.mean(predicted[in_class] == label))
            folds.append({
                "subject": subject,
                "variant": variant,
                "recall": recall,
                "balanced_accuracy": float(np.mean(list(recall.values()))) if len(recall) == 2 else None,
            })
        predictions.append(test)
    summary = {}
    for variant in ("previous_data", "updated_data"):
        rows = [fold for fold in folds if fold["variant"] == variant]
        class_recall = {label: float(np.mean([row["recall"][label] for row in rows
                                             if label in row["recall"]]))
                        for label in ("0", "1")}
        summary[variant] = {
            "recall_by_class": class_recall,
            "balanced_accuracy": float(np.mean(list(class_recall.values()))),
        }
    return folds, summary, pd.concat(predictions, ignore_index=True)


def run(raw_dir, pilot_dir, previous_release, session_manifest, output):
    previous = json.loads((previous_release / "release.json").read_text(encoding="utf-8-sig"))
    profile = previous["variants"]["LowRepeat"]
    prior_path = previous_release / profile["file"]
    if sha256(prior_path) != profile["sha256"]:
        raise ValueError("Previous packaged model does not match its manifest")
    prior_model = LogisticModel.load(prior_path)
    entries = json.loads(session_manifest.read_text(encoding="utf-8"))
    pilot_parts, pilot_hashes = zip(*pilot_features(raw_dir, pilot_dir))
    pilot = pd.concat(pilot_parts, ignore_index=True)
    additional, audit = zip(*additional_features(raw_dir, entries, set(pilot_hashes)))
    combined = pd.concat([pilot, *additional], ignore_index=True)
    threshold = profile["threshold"]
    folds, summary, predictions = evaluate(pilot, combined, threshold)

    baseline = train(pilot, "logistic_combined")
    features = combined[FEATURES].to_numpy()
    prior_delta = float(np.max(np.abs(baseline.predict_proba(features)
                                      - prior_model.predict_proba(features))))
    model = train(combined, "logistic_combined")
    model.threshold = threshold
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "python_openvibe_3s_PROVISIONAL.json"
    model.save(model_path)
    release = {key: previous[key] for key in ("runtime", "window_s", "step_s", "consecutive")}
    release.update(release=output.name, status="PROVISIONAL_OPENVIBE",
                   variants={"LowRepeat": dict(profile, file=model_path.name, sha256=sha256(model_path))})
    class_counts = combined.groupby("subject_id").label.value_counts().unstack(fill_value=0)
    report = {
        "status": release["status"],
        "pilot_windows": len(pilot),
        "total_windows": len(combined),
        "new_recordings": audit,
        "threshold": threshold,
        "class_counts": class_counts.to_dict(orient="index"),
        "summaries": summary,
        "folds": folds,
        "reconstructed_previous_model_max_score_delta": prior_delta,
        "summary_method": "Mean recall by person within each class, then mean across classes",
        "limitation": (
            "New session labels are provisional across whole recordings unless a KSS note is listed. "
            "KSS alignment is nominal. Reported metrics evaluate window classifications, "
            "and provisional session labels can reflect recording conditions."
        ),
    }
    for name, content in (("release.json", release), ("metrics_PROVISIONAL.json", report),
                          ("sessions.json", entries)):
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions.to_csv(output / "predictions_LOSO_PROVISIONAL.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw_dir", "pilot_dir", "previous_release", "session_manifest", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
