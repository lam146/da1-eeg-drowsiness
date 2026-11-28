"""Replay a new recording on the locked reference, personal and pooled Welch models."""
import argparse
import hashlib
import json
from pathlib import Path
from threading import Event

import pandas as pd

from n16_eeg.csv_test import test_recording
from n16_eeg.live import LiveDetector
from n16_eeg.model import load_model


def replay(folder, person, source, output):
    manifest = json.loads((folder / "profiles.json").read_text(encoding="utf-8"))
    if person not in manifest["people"]:
        raise ValueError(f"No personal profile for {person}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest in manifest["training_raw_hashes"]:
        raise ValueError("This recording belongs to the development cohort; use a new recording for this comparison.")
    sample = pd.read_csv(source, sep=None, engine="python", nrows=1, encoding="utf-8-sig")
    if "subject_id" in sample and len(sample):
        recorded = str(sample.subject_id.iloc[0]).strip().upper()
        if recorded not in (person, "UNASSIGNED"):
            raise ValueError(f"Subject in recording: {recorded}; requested: {person}")
    models = {}
    for method, spec in manifest["people"][person].items():
        path = folder / spec["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Frozen profile changed: {path}")
        models[method] = load_model(path)
    output.mkdir(parents=True,exist_ok=False)
    comparisons = []
    for method,model in models.items():
        detector = LiveDetector(model, fs=manifest["fs"], window_s=manifest["window_s"],
                                step_s=manifest["step_s"], consecutive=manifest["consecutive"],
                                release_consecutive=manifest["release_consecutive"], screen_raw=True)
        result = test_recording(source,output/f"{method}.csv",detector,Event())
        comparisons.append(dict(person=person,source_sha256=digest,model=method,**result))
    table=pd.DataFrame(comparisons)
    table.to_csv(output/"comparison.csv",index=False)
    print(table[["model","windows","episodes","alarm_windows","poor_windows"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("person")
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    replay(**vars(parser.parse_args()))
