"""Export and replay the post-hoc public-assisted development candidate."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from develop_pilot_model import session_weights
from try_cui_transfer import SPECTRAL, features
from n16_eeg.alarm import AlarmPolicy
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.simulate import generate_session


def run(folder):
    source = pd.read_csv(folder / "public_fp1_features.csv")
    target = pd.read_csv(folder / "target_features_PROVISIONAL.csv")
    columns = SPECTRAL + [name for name in source if name.startswith(("ar_", "sample_entropy_"))]
    sw, tw = session_weights(source), session_weights(target)
    data = pd.concat([target, source], ignore_index=True)
    weights = np.r_[.5 * tw, .5 * sw / sw.sum() * tw.sum()]
    scaler = StandardScaler().fit(data[columns], sample_weight=weights)
    model = LogisticRegression(C=.1, max_iter=3000).fit(scaler.transform(data[columns]), data.label, sample_weight=weights)
    joblib.dump(dict(scaler=scaler, estimator=model, feature_names=columns,
                     source_fraction=.5, threshold=.54, window_s=3, input_fs=512, feature_fs=128,
                     status="POST_HOC_DEVELOPMENT_ONLY"), folder / "candidate_OFFLINE.joblib")
    records = []
    for seed in (401, 409, 419, 431, 443, 457):
        for phase in (.05, .29, .56, .87):
            raw = generate_session(alert_seconds=12 + phase, drowsy_seconds=10, seed=seed)
            onset = raw.loc[raw.event.eq("STATE_DROWSY"), "timestamp_ms"].iloc[0]
            signal = raw.raw_eeg.to_numpy(float)
            ends = np.arange(1536, len(signal) + 1, 128)
            rows = pd.DataFrame([features(resample_poly(signal[end - 1536:end], 1, 4)) for end in ends])
            probability = model.predict_proba(scaler.transform(rows[columns]))[:, 1]
            times = raw.timestamp_ms.to_numpy()[ends - 1]
            for threshold in (.5, .54):
                policy = AlarmPolicy(threshold, 3, 3)
                alarm = np.array([policy.update(value) for value in probability])
                measured = alarm_onset_metrics(times, alarm, onset)
                records.append(dict(seed=seed, phase=phase, threshold=threshold,
                                    delay_ms=measured['onset_to_alarm_ms'],
                                    premature=measured['pre_onset_alarm'],
                                    valid_under_3s=measured['valid_under_3s']))
    table = pd.DataFrame(records)
    table.to_csv(folder / "candidate_synthetic.csv", index=False)
    summary = {str(threshold): dict(cases=len(group), premature=int(group.premature.sum()),
                                   valid_under_3s=int(group.valid_under_3s.sum()), invalid_or_missing=int(group.delay_ms.isna().sum()))
               for threshold, group in table.groupby("threshold")}
    (folder / "candidate_synthetic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
