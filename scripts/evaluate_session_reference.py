"""Evaluate first-minute spectral centering against an equally truncated LOSO baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from develop_pilot_model import FEATURES
from evaluate_repeat_wearers import load_local
from evaluate_welch_alarms import CUT, FS, WINDOW, STEP, alarm_trace, extract
from n16_eeg.quality import RawArtifactGuard
from tune_runtime_pilot import fit


REFERENCE = 60 * FS
SPECTRAL = FEATURES[:-3]
KEY = ["subject_id", "session_id"]


def reference(path):
    raw = pd.read_csv(path, usecols=["Electrode"], nrows=REFERENCE).Electrode.to_numpy(float)
    guard = RawArtifactGuard(WINDOW, FS, True)
    bad = np.fromiter((guard.update(sample) for sample in raw), bool, count=len(raw))
    ends = np.arange(WINDOW, len(raw) + 1, WINDOW)
    clean = ends[~bad[ends - 1]]
    center = np.full(len(SPECTRAL), np.nan)
    if len(clean) >= 10:
        features = extract(raw, clean, literature=True)["welch_blend"]
        center = np.median(features[:, :len(SPECTRAL)], axis=0)
    return len(clean), center


def run(pilot, cache, output):
    frame = load_local(pilot, cache)
    # Keep the entire EEG input window after the reference, even for the comparator.
    frame = frame.loc[frame.end_sample - WINDOW >= REFERENCE].copy()
    sessions = json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))["sessions"]
    refs = []
    for meta in sessions:
        clean, center = reference(meta["source"])
        refs.append({**{key: meta[key] for key in KEY}, "clean_windows": clean,
                     **dict(zip(SPECTRAL, center, strict=True))})
    references = pd.DataFrame(refs)
    table = frame.merge(references, on=KEY, suffixes=("", "_ref"), validate="many_to_one")
    ref_cols = [name + "_ref" for name in SPECTRAL]
    results, counts = [], []
    for train, test in LeaveOneGroupOut().split(table, groups=table.subject_id):
        donor, held = table.iloc[train], table.iloc[test]
        center = donor.drop_duplicates(KEY)[ref_cols].median().to_numpy()
        if not np.isfinite(center).all():
            raise ValueError("No usable reference in training sessions")
        for weight in (0., .5, 1.):
            def adjusted(rows):
                rows = rows.copy()
                shift = rows[ref_cols].to_numpy() - center
                rows[SPECTRAL] = rows[SPECTRAL].to_numpy() - weight * np.nan_to_num(shift)
                return rows

            model = fit(adjusted(donor.loc[donor.anchor & donor.valid]))
            scored = held.assign(score=model.predict_proba(adjusted(held)[FEATURES].to_numpy()))
            anchors = scored.loc[scored.valid & scored.anchor]
            pred = anchors.score.ge(CUT)
            awake = (~pred[anchors.label.eq(0)]).mean()
            sleepy = pred[anchors.label.eq(1)].mean()
            person = held.subject_id.iloc[0]
            results.append(dict(person=person, weight=weight, anchors=len(anchors), awake_recall=awake,
                                drowsy_recall=sleepy, balanced_accuracy=(awake+sleepy)/2))
            for (session, block), rows in scored.groupby(["session_id", "block"]):
                rows = rows.sort_values("end_sample")
                if rows.label.nunique() != 1 or not rows.end_sample.diff().dropna().eq(STEP).all():
                    raise ValueError("Non-contiguous labelled block")
                active = alarm_trace(rows.score, rows.valid)
                counts.append(dict(person=person, weight=weight, session=session, block=block,
                                   label=rows.label.iloc[0], valid=int(rows.valid.sum()),
                                   active=int(active.sum()), onsets=int((active & ~np.r_[False, active[:-1]]).sum())))
    folds = pd.DataFrame(results)
    blocks = pd.DataFrame(counts)
    summary = folds.groupby("weight")[["awake_recall", "drowsy_recall", "balanced_accuracy"]].mean()
    for label, name in [(0, "awake"), (1, "drowsy")]:
        stats = blocks.loc[blocks.label.eq(label)].groupby(["weight", "person"])[["onsets","valid","active"]].sum()
        summary[name+"_onsets"] = stats.onsets.groupby(level=0).sum()
        summary[name+"_minutes"] = stats.valid.groupby(level=0).sum() * STEP / FS / 60
        summary[name+"_fraction_active"] = (stats.active / stats.valid).groupby(level=0).mean()
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output/"folds.csv", index=False)
    blocks.to_csv(output/"blocks.csv", index=False)
    references[KEY+["clean_windows"]].to_csv(output/"references.csv", index=False)
    summary.to_csv(output/"summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("pilot", "cache", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
