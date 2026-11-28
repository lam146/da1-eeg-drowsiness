"""Run the packaged Windows app on the local inventory and check its bundled model."""

import argparse
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pandas as pd

from n16_live_app import load_detector
from n16_eeg.csv_test import test_recording


def run(executable, profile, inventory, output):
    output.mkdir(parents=True, exist_ok=False)
    executable = executable.resolve()
    subprocess.run([str(executable), "--check"], cwd=output, check=True, timeout=120)

    def replay(source, destination):
        subprocess.run([str(executable), "--csv", str(source.resolve()), "--output", str(destination.resolve())],
                       cwd=output, check=True, timeout=240)
        return pd.read_csv(destination)

    fixture = profile / "demo_session_SYNTHETIC.csv"
    packaged = replay(fixture, output / "demo_exe.csv")
    test_recording(fixture, output / "demo_python.csv", load_detector(profile), Event())
    expected = pd.read_csv(output / "demo_python.csv")
    pd.testing.assert_frame_equal(packaged.drop(columns="latency_ms"), expected.drop(columns="latency_ms"))

    def inspect(row):
        source = Path(row.file)
        if hashlib.sha256(source.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Recording changed: {source}")
        predictions = replay(source, output / (source.parent.name + "_" + source.name))
        alarms = predictions.alarm.eq(1)
        invalid = predictions.risk.isna()
        if (len(predictions) != row.windows + row.poor_windows or alarms[invalid].any()
                or predictions.raw_artifact.sum() != row.artifact_windows):
            raise ValueError(f"Prediction grid or quality differs: {source}")
        return dict(file=row.file, person=row.person, decisions=len(predictions),
                    artifact_windows=int(predictions.raw_artifact.sum()), alarm_windows=int(alarms.sum()),
                    episodes=int((alarms & ~alarms.shift(fill_value=False)).sum()))

    records = list(pd.read_csv(inventory).itertuples(index=False))
    with ThreadPoolExecutor(max_workers=3) as workers:
        result = pd.DataFrame(workers.map(inspect, records))
    result.to_csv(output / "inventory.csv", index=False)
    print(f"EXE startup OK; {len(packaged)} fixture decisions match Python; {len(result)} recordings passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("executable", "profile", "inventory", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
