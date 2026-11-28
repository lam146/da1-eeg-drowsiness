"""Compare the N16 spectrum with two fixed Welch estimators on held-out wearers."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from develop_pilot_model import FEATURES
from tune_runtime_pilot import fit
from n16_eeg.alarm import AlarmPolicy
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.blink import blink_window_features
from n16_eeg.preprocess import preprocess_window
from n16_eeg.quality import RawArtifactGuard
from n16_eeg.signal_features import BANDS, bandpower_features
from n16_eeg.simulate import generate_session

FS, WINDOW, STEP = 512, 1536, 128
METHODS = ("current", "welch_mean", "welch_median", "welch_blend")
CUT = .54
ALPHA = ["alpha_local_max", "alpha_local_range"]
SLOPE = ["log_slope_30_45"]
LITERATURE = {"welch_blend": [], "alpha": ALPHA, "slope": SLOPE, "alpha_slope": ALPHA + SLOPE,
              "log_relative": []}


def local_spectral_features(windows):
    """Five causal one-second subframes per window; frequencies fall on integer Hz."""
    frames = np.lib.stride_tricks.sliding_window_view(np.asarray(windows), FS, axis=-1)[:, ::FS//2, :]
    tapered = (frames - frames.mean(axis=-1, keepdims=True)) * np.hanning(FS + 1)[:-1]
    power = np.abs(np.fft.rfft(tapered, axis=-1)) ** 2
    eps = np.finfo(float).tiny
    relative = power[:, :, 8:13].sum(axis=2) / np.maximum(power[:, :, 4:30].sum(axis=2), eps)
    # Fit on the raw spectrum so the 45 Hz preprocessing cutoff cannot create a slope.
    centered = np.log(np.arange(30, 46, dtype=float))
    centered -= centered.mean()
    log_power = np.log(np.maximum(power[:, :, 30:46].mean(axis=1), eps))
    slope = ((log_power - log_power.mean(axis=1, keepdims=True)) @ centered) / (centered @ centered)
    return np.column_stack((relative.max(axis=1), np.ptp(relative, axis=1), slope))


def spectral_columns(freq, psd):
    band = (freq >= .5) & (freq <= 45)
    eps = np.finfo(float).eps
    total = np.maximum(np.trapezoid(psd[:, band], freq[band], axis=1), eps)
    powers = {}
    for name, (low, high) in BANDS.items():
        mask = (freq >= low) & (freq < high)
        powers[name] = np.trapezoid(psd[:, mask], freq[mask], axis=1)
    normalized = psd[:, band] / np.maximum(psd[:, band].sum(axis=1, keepdims=True), eps)
    entropy = -(normalized * np.log2(normalized + eps)).sum(axis=1) / np.log2(band.sum())
    relative = [powers[name] / total for name in BANDS]
    ratios = [powers[a] / np.maximum(powers[b], eps) for a, b in
              (("theta", "alpha"), ("theta", "beta"), ("alpha", "beta"))]
    return np.column_stack(relative + ratios + [entropy])


def extract(raw, ends, literature=False):
    """Every feature uses exactly 3 seconds ending at its decision time."""
    features = {name: [] for name in (LITERATURE if literature else METHODS)}
    for offset in range(0, len(ends), 128):
        windows = [raw[end - WINDOW:end] for end in ends[offset:offset + 128]]
        blink = np.array([[row[name] for name in FEATURES[-3:]] for row in
                         (blink_window_features(x, FS) for x in windows)])
        if not literature:
            current = np.array([[row[name] for name in FEATURES[:-3]] for row in
                                (bandpower_features(x, FS) for x in windows)])
            features["current"].append(np.column_stack((current, blink)))
        filtered = np.stack([preprocess_window(x, FS) for x in windows])
        spectra = []
        for average in ("mean", "median"):
            freq, psd = welch(filtered, FS, window="hann", nperseg=FS, noverlap=FS // 2,
                              nfft=WINDOW, average=average, axis=-1)
            spectra.append(psd)
            if not literature:
                features["welch_" + average].append(np.column_stack((spectral_columns(freq, psd), blink)))
        blend = np.column_stack((spectral_columns(freq, (spectra[0] + spectra[1]) / 2), blink))
        features["welch_blend"].append(blend)
        if literature:
            extra = local_spectral_features(windows)
            log_relative = blend.copy()
            # Log relative powers and power ratios. Keep entropy and blink in their native units.
            log_relative[:, :8] = np.log(np.maximum(log_relative[:, :8], 1e-12))
            features["log_relative"].append(log_relative)
            features["alpha"].append(np.column_stack((blend, extra[:, :2])))
            features["slope"].append(np.column_stack((blend, extra[:, 2])))
            features["alpha_slope"].append(np.column_stack((blend, extra)))
    return {name: np.vstack(rows) for name, rows in features.items()}


def alarm_trace(risks, valid, options=None):
    policy = AlarmPolicy(CUT, **(options or dict(consecutive=3, release_consecutive=3)))
    return np.array([policy.update(score if good else None) for score, good in zip(risks, valid, strict=True)])


def collect(pilot, literature=False, anchors_only=False):
    manifest = json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    sessions = {(s["subject_id"], s["session_id"]): s for s in manifest["sessions"]}
    table = pd.read_csv(pilot / "all_subjects_features_PROVISIONAL.csv")
    if (set(table.subject_id) != {"S01", "S02"} or set(table.fs_hz) != {FS}
            or set(table.label) != {0, 1} or table.duplicated(["subject_id","session_id","window_end_ms"]).any()
            or not (table.window_end_ms - table.window_start_ms).eq(4000).all()):
        raise ValueError("Unexpected pilot cohort or window grid")
    blocks = []
    for (person, session), group in table.groupby(["subject_id", "session_id"]):
        source = sessions[person, session]
        path = Path(source["source"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["source_sha256"]:
            raise ValueError(f"Recording changed: {path}")
        raw = pd.read_csv(path, usecols=["Electrode"]).Electrode.to_numpy(float)
        guard = RawArtifactGuard(WINDOW, FS, True)
        bad = np.fromiter((guard.update(value) for value in raw), bool, count=len(raw))
        rows = group.sort_values("window_start_ms")
        split = rows.window_start_ms.ne(rows.window_end_ms.shift()) | rows.label.ne(rows.label.shift())
        for number, part in rows.groupby(split.cumsum()):
            start = round(part.window_start_ms.iloc[0] * FS / 1000)
            end = round(part.window_end_ms.iloc[-1] * FS / 1000)
            if not 0 <= start <= end <= len(raw):
                raise ValueError("Label interval outside the recording")
            ends = np.arange(start + WINDOW, end + 1, STEP)
            anchors = np.isin(ends, np.rint(part.window_end_ms * FS / 1000).astype(int))
            if anchors.sum() != len(part):
                raise ValueError("Missing training window")
            info = pd.DataFrame(dict(subject_id=person, session_id=session, block=int(number),
                                label=int(part.label.iloc[0]), end_sample=ends, anchor=anchors, valid=~bad[ends-1]))
            if anchors_only:
                ends = ends[anchors]
                info = info.loc[anchors]
            blocks.append((info, extract(raw, ends, literature)))
        print(person, session, flush=True)
    return blocks


def check_public_transfer(pilot, dataset, output):
    """Score untouched individual public epochs with a frozen N16 fit for each feature set."""
    from scipy.io import loadmat
    from scipy.signal import resample_poly

    checksum = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != "53cc4ef14b1343f7f3fb5322dd2b541c031c6c2297bddf1817aed04dc687a6a4":
        raise ValueError("Expected the reviewed Cui v3 dataset")
    content = loadmat(dataset)
    eeg = content["EEGsample"]
    truth = content["substate"].ravel()
    people = content["subindex"].ravel()
    if eeg.shape != (2022, 30, 384) or set(truth) != {0, 1} or len(np.unique(people)) != 11:
        raise ValueError("Unexpected public EEG shape or labels")
    # Each 384-sample epoch is resampled in isolation. No cross-epoch windows.
    public = resample_poly(eeg[:, 0, :], 4, 1, axis=1)
    vectors = extract(public.ravel(), WINDOW * np.arange(1, len(public) + 1), literature=True)
    blocks = collect(pilot, literature=True, anchors_only=True)
    info = pd.concat([meta for meta, _ in blocks], ignore_index=True)
    folds = []
    for method, extra in LITERATURE.items():
        cols = FEATURES + extra
        features = np.vstack([matrix[method] for _, matrix in blocks])
        model = fit(pd.concat([info, pd.DataFrame(features, columns=cols)], axis=1), cols)
        predicted = model.predict_proba(vectors[method]) >= CUT
        for person in np.unique(people):
            selected = people == person
            awake = float((~predicted[selected & (truth == 0)]).mean())
            drowsy = float(predicted[selected & (truth == 1)].mean())
            folds.append(dict(method=method, subject=int(person), epochs=int(selected.sum()),
                              awake_recall=awake, drowsy_recall=drowsy, balanced_accuracy=(awake+drowsy)/2))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(folds)
    frame.to_csv(output / "cui_external_by_person.csv", index=False)
    summary = frame.groupby("method")[["awake_recall", "drowsy_recall", "balanced_accuracy"]].mean()
    summary.to_csv(output / "cui_external.csv")
    print(summary, flush=True)


def run(pilot, output, literature=False, alarm_smoothing=False):
    output.mkdir(parents=True, exist_ok=True)
    blocks = collect(pilot, literature or alarm_smoothing)
    all_info = pd.concat([info for info, _ in blocks], ignore_index=True)
    anchor = all_info.anchor.to_numpy()
    folds, episodes, timing = [], [], []
    full_models = {}
    policies = {"welch_blend": {}, "mean3": dict(smoothing_window=3, consecutive=2, release_consecutive=2),
                "mean5": dict(smoothing_window=5, consecutive=2, release_consecutive=2)}
    methods = dict.fromkeys(policies, []) if alarm_smoothing else (LITERATURE if literature else dict.fromkeys(METHODS, []))
    for method, extra in methods.items():
        columns = FEATURES + extra
        feature_set = "welch_blend" if alarm_smoothing else method
        options = policies[method] if alarm_smoothing else None
        values = np.vstack([vectors[feature_set] for _, vectors in blocks])
        training = pd.concat([all_info.loc[anchor].reset_index(drop=True),
                              pd.DataFrame(values[anchor], columns=columns)], axis=1)
        full_models[method] = fit(training, columns)
        for person in sorted(all_info.subject_id.unique()):
            model = fit(training.loc[training.subject_id.ne(person)], columns)
            test = all_info.subject_id.eq(person)
            marked = all_info.loc[test & anchor]
            positive = model.predict_proba(values[test & anchor]) >= CUT
            awake = float((~positive[marked.label.eq(0)]).mean())
            drowsy = float(positive[marked.label.eq(1)].mean())
            folds.append(dict(method=method, person=person, anchors=len(marked), awake_recall=awake,
                              drowsy_recall=drowsy, balanced_accuracy=(awake + drowsy) / 2))
            for info, vectors in blocks:
                if info.subject_id.iloc[0] != person:
                    continue
                active = alarm_trace(model.predict_proba(vectors[feature_set]), info.valid, options)
                episodes.append(dict(method=method, person=person, session=info.session_id.iloc[0],
                    block=int(info.block.iloc[0]), label=int(info.label.iloc[0]), decisions=len(info),
                    valid=int(info.valid.sum()), active=int(active.sum()),
                    onsets=int(np.count_nonzero(active & ~np.r_[False, active[:-1]]))))
    for seed in (401, 409, 419, 431, 443, 457, 2003, 2011, 2017, 2027, 2029, 2039):
        for phase in (.05, .29, .56, .87):
            signal = generate_session(alert_seconds=12 + phase, drowsy_seconds=10, seed=seed)
            raw = signal.raw_eeg.to_numpy(float)
            ends = np.arange(WINDOW, len(raw) + 1, STEP)
            vectors = extract(raw, ends, literature or alarm_smoothing)
            guard = RawArtifactGuard(WINDOW, FS, True)
            valid = ~np.fromiter((guard.update(x) for x in raw), bool)[ends-1]
            if signal.poor_signal.gt(50).any():
                raise ValueError("Synthetic fixture unexpectedly contains poor contact")
            onset = signal.loc[signal.event.eq("STATE_DROWSY"), "timestamp_ms"].item()
            for method, model in full_models.items():
                feature_set = "welch_blend" if alarm_smoothing else method
                active = alarm_trace(model.predict_proba(vectors[feature_set]), valid,
                                     policies[method] if alarm_smoothing else None)
                timing.append(dict(method=method, seed=seed, phase=phase, suite="fresh" if seed >= 2000 else "previous",
                    **alarm_onset_metrics(signal.timestamp_ms.to_numpy()[ends-1], active, onset)))
    folds, episodes, timing = map(pd.DataFrame, (folds, episodes, timing))
    rates = episodes.groupby(["method","person","label"])[["decisions","onsets","active","valid"]].sum()
    rates["onsets_per_hour_valid"] = rates.onsets * 3600 * FS / (STEP * rates.valid)
    rates["fraction_active_valid"] = rates.active / rates.valid
    summary = folds.groupby("method")[["awake_recall","drowsy_recall","balanced_accuracy"]].mean()
    by_state = rates.groupby(["method","label"]).mean()
    for state, label in [(0, "awake"), (1, "drowsy")]:
        group = by_state.xs(state, level="label")
        summary[label + "_onsets_per_hour_valid"] = group.onsets_per_hour_valid
        summary[label + "_fraction_active_valid"] = group.fraction_active_valid
    summary["synthetic_early"] = timing.groupby("method").pre_onset_alarm.sum()
    summary["synthetic_under_3s"] = timing.groupby("method").valid_under_3s.sum()
    summary["worst_valid_delay_ms"] = timing.groupby("method").onset_to_alarm_ms.max()
    for name, frame in dict(summary=summary.reset_index(), folds=folds, blocks=episodes,
                            by_person_state=rates.reset_index(), synthetic=timing).items():
        frame.to_csv(output / (name + ".csv"), index=False)
    if literature:
        sessions = episodes.loc[episodes.method.eq("welch_blend")].groupby(
            ["person", "label", "session"], as_index=False)[["valid", "onsets", "active"]].sum()
        sessions["scored_minutes"] = sessions.valid * STEP / (60 * FS)
        sessions["onsets_per_minute"] = sessions.onsets / sessions.scored_minutes
        sessions["fraction_active"] = sessions.active / sessions.valid
        sessions.sort_values(["label", "onsets_per_minute"], ascending=[True, False]).to_csv(
            output / "kiem_tra_tung_phien.csv", index=False)
    if literature or alarm_smoothing:
        print(summary.to_string(), flush=True)
        return
    for method in ("welch_median", "welch_blend"):
        model = full_models[method]
        model.threshold = CUT
        model.spectral_estimator = method
        model.format_version = 2
        profile = output / method
        profile.mkdir(exist_ok=True)
        model_path = profile / "combined_PROVISIONAL.json"
        model.save(model_path)
        fixture = Path(__file__).resolve().parents[1] / "output/N16_Reviewed_20261127/demo_session_SYNTHETIC.csv"
        shutil.copyfile(fixture, profile / fixture.name)
        manifest = dict(release=method, status="EXPERIMENTAL_S01_S02", window_s=3, step_s=.25, consecutive=3,
            fixture=dict(file=fixture.name, sha256=hashlib.sha256(fixture.read_bytes()).hexdigest()),
            variants=dict(LowRepeat=dict(file=model_path.name, sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                         threshold=CUT, release_consecutive=3, screen_raw=True)))
        (profile / "release.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--literature", action="store_true")
    mode.add_argument("--alarm-smoothing", action="store_true")
    mode.add_argument("--public-test", type=Path, help="Score Cui Fp1 epochs using the N16-only fit")
    args = parser.parse_args()
    if args.public_test:
        check_public_transfer(args.pilot, args.public_test, args.output)
    else:
        run(args.pilot, args.output, args.literature, args.alarm_smoothing)
