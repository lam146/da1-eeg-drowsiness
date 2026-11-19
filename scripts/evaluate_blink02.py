"""Evaluate BLINK_02 with frozen defaults and early-only estimated synchronization.

Video annotations are reviewed separately from EEG. This is an exploratory,
computer-assisted reference, not an independently double-annotated benchmark.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from n16_eeg.blink import detect_blinks
from n16_eeg.blink_eval import match_event_pairs, match_events
from evaluate_s02_blink_review import in_range, score_detector, DEFAULT


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/blink02_s02_20261107"
SESSION = ROOT / "raw/Raw data/S02/S02_07112026_Blink02.csv"
VIDEO = ROOT / "raw/video/S02_07112026_Blink02.mp4"
CALIBRATION = (0., 60.)
HOLDOUT = (60., 239.)
TOLERANCE = .20


def run():
    raw = pd.read_csv(SESSION)
    timeline = raw["Time:512Hz"].to_numpy(float)
    x = raw["Electrode"].to_numpy(float)
    if (not np.isfinite(timeline).all() or not np.isfinite(x).all()
            or not np.allclose(np.diff(timeline), 1/512, rtol=0, atol=1e-7)):
        raise ValueError("Expected finite, continuous 512 Hz samples")
    device = raw.loc[pd.to_numeric(raw["Event Id"], errors="coerce") == 33025,
                     "Event Date"].to_numpy(float) - timeline[0]
    ref = pd.read_csv(OUT / "video_blinks_confirmed.csv").timestamp_ms.to_numpy(float)/1000
    # Offset search and clock fit use seven early device/video anchors only.
    early_d = in_range(device, (0., 60.))
    early_r = in_range(ref, CALIBRATION)
    candidates = []
    for offset in np.linspace(-2, 2, 401):
        pairs = match_event_pairs(early_d + offset, early_r, TOLERANCE)
        candidates.append((-len(pairs), sum(abs(a-b) for a,b in pairs), offset, pairs))
    _, _, initial, pairs = min(candidates, key=lambda c:c[:3])
    if len(pairs) < 4:
        raise ValueError("Insufficient early alignment anchors")
    eeg, vid = np.array(pairs).T
    eeg -= initial
    scale, offset = np.polyfit(eeg, vid, 1)
    if scale <= 0 or HOLDOUT[1] > min(240.20, len(x)/512*scale+offset):
        raise ValueError("Invalid estimated overlap")
    full = np.array([e.time_seconds for e in detect_blinks(x, fs=512)])
    predictions = {
        "raw_default_split": score_detector(x, HOLDOUT, scale, offset, DEFAULT),
        "raw_default_whole_session": in_range(full*scale+offset, HOLDOUT),
        "device_event": in_range(device*scale+offset, HOLDOUT),
    }
    # Alternative alignment audit: EEG waveform peaks need not share device
    # event latency. Fit the same two-parameter model on early raw peaks only;
    # retain both results rather than selecting the better holdout score.
    candidates_raw = []
    for shift in np.linspace(-2, 2, 401):
        paired = match_event_pairs(in_range(full, CALIBRATION)+shift, early_r, TOLERANCE)
        candidates_raw.append((-len(paired), sum(abs(a-b) for a,b in paired), shift, paired))
    _, _, initial_raw, raw_pairs = min(candidates_raw, key=lambda c:c[:3])
    if len(raw_pairs) < 4:
        raise ValueError("Insufficient early raw alignment anchors")
    raw_eeg, raw_vid = np.array(raw_pairs).T
    raw_eeg -= initial_raw
    raw_scale, raw_offset = np.polyfit(raw_eeg, raw_vid, 1)
    predictions["raw_default_raw_anchor_alignment"] = score_detector(
        x, HOLDOUT, raw_scale, raw_offset, DEFAULT)
    targets = in_range(ref, HOLDOUT)
    rows=[]
    for name, times in predictions.items():
        matched = match_event_pairs(times, targets, TOLERANCE)
        used_p={p for p,r in matched};used_r={r for p,r in matched}
        rows += [dict(detector=name,status="matched",prediction_video_s=p,
                      reference_video_s=r,error_ms=(p-r)*1000) for p,r in matched]
        rows += [dict(detector=name,status="prediction_only",prediction_video_s=p)
                 for p in times if p not in used_p]
        rows += [dict(detector=name,status="reference_only",reference_video_s=r)
                 for r in targets if r not in used_r]
    report = dict(
        session=str(SESSION), samples=len(x), sample_rate_hz=512,
        sample_count_duration_s=len(x)/512, video_duration_s=240.20,
        user_sync_statement="Video recording started together with START",
        synchronization_verified=False,
        reference_method="All 7206 sampled video frames scored at 30 fps; 29 candidates visually confirmed; six additional low-score candidates visually excluded. Computer-assisted single-reviewer annotation, potentially incomplete.",
        reference_count=len(ref), device_events_full=len(device), raw_default_events_full=len(full),
        calibration_video_s=CALIBRATION, holdout_video_s=HOLDOUT,
        holdout_reference_count=len(targets), tolerance_ms=200,
        frozen_parameters=dict(threshold_mad=6.,refractory_s=.5,min_width_s=.08,max_width_s=.6),
        video_time_formula=dict(scale=float(scale),offset_s=float(offset)),
        alternative_raw_anchor_formula=dict(scale=float(raw_scale),offset_s=float(raw_offset)),
        alternative_raw_anchors=[dict(eeg_s=a,video_s=b,residual_ms=(a*raw_scale+raw_offset-b)*1000)
                                for a,b in zip(raw_eeg,raw_vid)],
        calibration_anchors=[dict(eeg_s=a,video_s=b,residual_ms=(a*scale+offset-b)*1000)
                             for a,b in zip(eeg,vid)],
        holdout={name:match_events(times,targets,TOLERANCE) for name,times in predictions.items()},
        raw_split_offset_sensitivity=[dict(extra_offset_ms=s*1000,**match_events(
            in_range(predictions["raw_default_split"]+s,HOLDOUT),targets,TOLERANCE))
            for s in [-.1,-.05,0.,.05,.1]],
        raw_anchor_offset_sensitivity=[dict(extra_offset_ms=s*1000,**match_events(
            in_range(predictions["raw_default_raw_anchor_alignment"]+s,HOLDOUT),targets,TOLERANCE))
            for s in [-.1,-.05,0.,.05,.1]],
        limitations=["Clock fit uses device events, whose latency is not a hardware synchronization marker.",
                     "Device events are not independent labels; video is the reference.",
                     "No detector parameters were tuned on BLINK_02.",
                     "Count agreement does not measure correct event detection.",
                     "One new session from S02; not a cross-person benchmark.",
                     "Blink sessions do not carry KSS drowsiness labels and are excluded from drowsiness training."],
    )
    report["input_sha256"]={str(p.relative_to(ROOT)):hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
                             for p in [SESSION, VIDEO, OUT/"video_blinks_confirmed.csv"]}
    pd.DataFrame(rows).to_csv(OUT/"holdout_event_matches.csv",index=False)
    (OUT/"video_blink_metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    run()
