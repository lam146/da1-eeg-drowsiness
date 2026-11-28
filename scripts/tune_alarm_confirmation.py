"""Replay held-out OpenViBE traces and simulated onsets to compare alarm confirmation."""

import argparse
import json
import shutil
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from update_openvibe_model import FS, WINDOW, feature_vector, read_raw, recording_labels, sha256
from upgrade_openvibe_model import FEATURES, fit, predict

from n16_eeg.alarm import AlarmPolicy
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.quality import RawArtifactGuard
from n16_eeg.model import load_model
from n16_eeg.simulate import generate_session

STEP = FS // 4


def scores(raw, predict_values):
    stops = np.arange(WINDOW, len(raw) + 1, STEP)
    values = pd.DataFrame([feature_vector(raw[stop - WINDOW:stop]) for stop in stops], columns=FEATURES)
    guard = RawArtifactGuard(WINDOW, FS, True)
    rejected = np.fromiter((guard.update(sample) for sample in raw), dtype=bool, count=len(raw))
    valid = ~rejected[stops - 1]
    return stops, predict_values(values), valid


def evaluate_trace(values, valid, threshold, count, release_count, support=None):
    floor, minimum = support or (None, 1)
    policy = AlarmPolicy(threshold, consecutive=count, release_consecutive=release_count,
                         support_threshold=floor, support_consecutive=minimum)
    return np.array([policy.update(score if keep else None) for score, keep in zip(values, valid)])


def collect(source, raw_dir, fixture):
    frame = pd.read_csv(source / "predictions_LOSO_PROVISIONAL.csv")
    sessions = json.loads((source / "sessions.json").read_text(encoding="utf-8"))
    people = {Path(session["file"]).parent.name for session in sessions}
    models = {person: fit(frame.loc[frame.subject_id.ne(person)]) for person in people}
    real = []
    for session in sessions:
        relative = Path(session["file"])
        person = relative.parent.name
        path = raw_dir / relative
        raw = read_raw(path)
        stops, values, valid = scores(raw, partial(predict, models[person]))
        labels, _ = recording_labels(raw_dir, session, path, sha256(path), len(raw), stops - WINDOW, stops)
        for block in np.split(np.arange(len(stops)), np.flatnonzero(np.diff(labels)) + 1):
            if len(block) and labels[block[0]] >= 0 and valid[block].any():
                real.append((person, int(labels[block[0]]), values[block], valid[block]))
        print(relative.name, flush=True)
    profile = json.loads((source / "release.json").read_text())["variants"]["LowRepeat"]
    model = load_model(source / profile["file"])
    synthetic = []
    fixtures = [pd.read_csv(fixture)]
    fixtures.extend(generate_session(alert_seconds=12 + phase, drowsy_seconds=12, seed=seed)
                    for seed in (51, 71, 103, 127) for phase in (.05, .23, .47, .71, .93))
    for rows in fixtures:
        stops, values, valid = scores(rows.raw_eeg.to_numpy(float), lambda features: np.array(
            [model.predict_feature_dict(row) for row in features.to_dict(orient="records")]))
        onset = rows.loc[rows.event.eq("STATE_DROWSY"), "timestamp_ms"].iloc[0]
        synthetic.append((1000 * (stops - 1) / FS - onset, values, valid))
    return real, synthetic


def compare(real, synthetic, threshold, count, release_count, support=None):
    measured = []
    for subject, label, values, valid in real:
        alarm = evaluate_trace(values, valid, threshold, count, release_count, support)
        edges = np.flatnonzero(np.diff(np.r_[False, alarm, False]))
        durations = edges[1::2] - edges[::2]
        minutes = len(alarm) / 240
        measured.append((subject, label, len(durations) / minutes,
                         np.count_nonzero(durations <= 2 * FS // STEP) / minutes,
                         float(alarm[valid].mean())))
    table = pd.DataFrame(measured, columns=["subject", "label", "episodes_per_minute",
                                            "short_episodes_per_minute", "alarm_fraction"])
    rates = table.groupby(["label", "subject"]).mean().groupby("label").mean()
    delays, early = [], 0
    for time, values, valid in synthetic:
        alarm = evaluate_trace(values, valid, threshold, count, release_count, support)
        timing = alarm_onset_metrics(time, alarm, 0)
        early += int(timing["pre_onset_alarm"])
        delays.append(timing["onset_to_alarm_ms"])
    return {
        "threshold": threshold,
        "confirmation_windows": count,
        "release_windows": release_count,
        "support": support,
        "awake_episodes_per_minute": float(rates.loc[0, "episodes_per_minute"]),
        "awake_short_episodes_per_minute": float(rates.loc[0, "short_episodes_per_minute"]),
        "awake_alarm_fraction": float(rates.loc[0, "alarm_fraction"]),
        "drowsy_alarm_fraction": float(rates.loc[1, "alarm_fraction"]),
        "early_synthetic_alarms": early,
        "synthetic_under_3s": sum(delay is not None and delay < 3000 for delay in delays),
        "synthetic_alarms": sum(delay is not None for delay in delays),
        "synthetic_cases": len(delays),
        "synthetic_worst_ms": max((delay for delay in delays if delay is not None), default=None),
        "synthetic_demo_ms": delays[0],
    }


def run(source, raw_dir, fixture, output, release_directory=None):
    real, synthetic = collect(source, raw_dir, fixture)
    baseline = compare(real, synthetic, .57, 3, 3)
    candidates = [compare(real, synthetic, threshold, count, 3)
                  for threshold, count in product((.55, .56, .57, .58, .59, .60), range(4, 12))]
    candidates.extend(compare(real, synthetic, .57, count, 3, (floor, minimum))
                      for count, floor, minimum
                      in product((2, 3, 4), (.535, .545, .55, .555, .56, .565), range(5, 13)))
    # Every case must have a fresh alarm strictly within the N16 deadline.
    eligible = [item for item in candidates if item["threshold"] == baseline["threshold"]
                and item["support"] is not None and item["support"][1] >= 6
                and item["early_synthetic_alarms"] == 0
                and item["synthetic_alarms"] == item["synthetic_cases"]
                and item["synthetic_under_3s"] == item["synthetic_cases"] and item["drowsy_alarm_fraction"] >= .58]
    selected = min(eligible, key=lambda item: (item["awake_short_episodes_per_minute"],
                                              -item["synthetic_under_3s"],
                                              -item["drowsy_alarm_fraction"]), default=None)
    report = {"scope": "Development comparison. Raw OpenViBE sessions use provisional labels "
                       "and a model fit with the replay subject held out. Timing is synthetic.",
              "baseline": baseline, "selected": selected, "candidates": candidates}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if release_directory is not None and selected is None:
        raise ValueError("No candidate met the release criteria. See the comparison report.")
    if release_directory is not None:
        release = json.loads((source / "release.json").read_text(encoding="utf-8-sig"))
        profile = release["variants"]["LowRepeat"]
        floor, minimum = selected["support"]
        profile.update(support_threshold=floor, support_consecutive=minimum,
                       release_consecutive=selected["release_windows"])
        release.update(release=release_directory.name, consecutive=selected["confirmation_windows"])
        release_directory.mkdir(parents=True, exist_ok=False)
        for filename in (profile["file"], "sessions.json"):
            shutil.copyfile(source / filename, release_directory / filename)
        (release_directory / "release.json").write_text(
            json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "raw_dir", "fixture", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--release-directory", type=Path)
    run(**vars(parser.parse_args()))
