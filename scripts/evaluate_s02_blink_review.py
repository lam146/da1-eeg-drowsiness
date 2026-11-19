"""Score the reviewed S02 blink video against the matching OpenViBE recording."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_event_pairs, match_events


FS = 512
TOLERANCE = .20
CALIBRATION = (27.0, 120.0)
REVIEW = (120.0, 238.0)
DEFAULT = (6.0, .50, .08)


def in_range(times, limits):
    start, stop = limits
    return times[(times >= start) & (times < stop)]


def align_from_calibration(device, reference):
    """Estimate video = scale * EEG + offset from early OpenViBE events only."""
    reference = in_range(reference, CALIBRATION)
    device = in_range(device, (CALIBRATION[0] - 3, CALIBRATION[1] + 3))
    candidates = []
    for offset in np.linspace(-3, 3, 601):
        pairs = match_event_pairs(device + offset, reference, TOLERANCE)
        if pairs:
            error = sum(abs(a - b) for a, b in pairs)
            candidates.append((-len(pairs), error, offset, pairs))
    if not candidates:
        raise ValueError("No alignment candidates in the calibration interval")
    _, _, initial, pairs = min(candidates, key=lambda candidate: candidate[:3])
    if len(pairs) < 4:
        raise ValueError("Insufficient calibration anchors")
    eeg, video = np.asarray(pairs).T
    eeg -= initial
    scale, offset = np.polyfit(eeg, video, 1)
    return float(scale), float(offset), list(zip(eeg.tolist(), video.tolist()))


def score_detector(x, limits, scale, offset, params):
    # Use only samples inside this interval, including when estimating the MAD.
    start, stop = np.clip(np.ceil((np.asarray(limits) - offset) / scale * FS).astype(int), 0, len(x))
    threshold, refractory, width = params
    events = detect_blinks(x[start:stop], fs=FS, threshold_mad=threshold,
                           refractory_s=refractory, min_width_s=width)
    video_times = np.array([(start / FS + event.time_seconds) * scale + offset for event in events])
    return in_range(video_times, limits)


def run(session, annotations, output):
    raw = pd.read_csv(session)
    timeline = raw["Time:512Hz"].to_numpy(float)
    x = raw["Electrode"].to_numpy(float)
    if (not np.isfinite(timeline).all() or not np.isfinite(x).all()
            or not np.allclose(np.diff(timeline), 1 / FS, rtol=0, atol=1e-7)):
        raise ValueError("Expected finite OpenViBE data on a 512 Hz sample grid")
    device = np.sort(raw["Event Date"].dropna().to_numpy(float) - timeline[0])
    if not np.isfinite(device).all():
        raise ValueError("Invalid OpenViBE event times")
    reference = np.sort(pd.read_csv(annotations)["timestamp_ms"].to_numpy(float) / 1000)
    if not np.isfinite(reference).all() or np.unique(reference).size != len(reference):
        raise ValueError("Expected unique, finite video timestamps")
    scale, offset, anchors = align_from_calibration(device, reference)
    if REVIEW[1] > len(x) / FS * scale + offset:
        raise ValueError("Review interval extends beyond the estimated overlap")
    calibration_refs = in_range(reference, CALIBRATION)
    holdout_refs = in_range(reference, REVIEW)
    all_default = np.array([event.time_seconds for event in detect_blinks(x, fs=FS)])
    aligned_device = device * scale + offset
    trials = []
    for params in product([4., 5., 6., 7., 8.], [.25, .35, .50], [.05, .08, .10]):
        predictions = score_detector(x, CALIBRATION, scale, offset, params)
        scores = match_events(predictions, calibration_refs, TOLERANCE)
        trials.append(dict(threshold_mad=params[0], refractory_s=params[1],
                           min_width_s=params[2], **scores))
    best = max(trials, key=lambda trial: (trial["f1"], trial["precision"],
                                         -abs(trial["threshold_mad"] - DEFAULT[0]),
                                         -abs(trial["refractory_s"] - DEFAULT[1]),
                                         -abs(trial["min_width_s"] - DEFAULT[2])))
    params = tuple(best[key] for key in ("threshold_mad", "refractory_s", "min_width_s"))
    predictions = {
        "default_split": score_detector(x, REVIEW, scale, offset, DEFAULT),
        "tuned_split": score_detector(x, REVIEW, scale, offset, params),
        "openvibe": in_range(aligned_device, REVIEW),
    }
    report = dict(
        scope="Single-session exploratory comparison using estimated synchronization",
        synchronization_verified=False, video_reference_events=len(reference),
        raw_events_full=len(all_default), openvibe_events_full=len(device),
        video_time_formula={"scale": scale, "offset_s": offset},
        calibration_anchors=[{"eeg_s": a, "video_s": b, "residual_ms": (a * scale + offset - b) * 1000}
                             for a, b in anchors],
        tolerance_ms=TOLERANCE * 1000, calibration_video_s=CALIBRATION,
        holdout_video_s=REVIEW, holdout_reference_count=len(holdout_refs),
        selected_parameters=best,
        stable_interval_default=match_events(
            in_range(all_default * scale + offset, (CALIBRATION[0], REVIEW[1])),
            in_range(reference, (CALIBRATION[0], REVIEW[1])), TOLERANCE),
        holdout={name: match_events(times, holdout_refs, TOLERANCE)
                 for name, times in predictions.items()},
    )
    report["holdout_offset_sensitivity"] = [
        dict(additional_offset_ms=shift * 1000, **match_events(
            in_range(predictions["default_split"] + shift, REVIEW), holdout_refs, TOLERANCE))
        for shift in [-.10, -.05, 0, .05, .10]
    ]
    rows = []
    for name, times in predictions.items():
        pairs = match_event_pairs(times, holdout_refs, TOLERANCE)
        used_predictions, used_refs = map(set, zip(*pairs)) if pairs else (set(), set())
        rows.extend(dict(detector=name, status="matched", prediction_video_s=p,
                         reference_video_s=r, error_ms=(p - r) * 1000) for p, r in pairs)
        rows.extend(dict(detector=name, status="prediction_only", prediction_video_s=p)
                    for p in times if p not in used_predictions)
        rows.extend(dict(detector=name, status="reference_only", reference_video_s=r)
                    for r in holdout_refs if r not in used_refs)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(output / "calibration_trials.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "holdout_event_matches.csv", index=False)
    (output / "video_blink_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.session, args.annotations, args.output), indent=2))
