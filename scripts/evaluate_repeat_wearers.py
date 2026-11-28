"""Evaluate prior-day labelled calibration with fixed rules and a calendar-causal donor."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_ensembles import alarm_states, public_training_frame
from develop_pilot_model import FEATURES
from evaluate_welch_alarms import FS, WINDOW, STEP, CUT, extract
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.quality import RawArtifactGuard
from n16_eeg.simulate import generate_session
from tune_runtime_pilot import fit
from pilot_calendar import measurement_date
from n16_eeg.model import load_model


def load_local(pilot, cache):
    rows = pd.read_csv(cache / "window_index.csv")
    values = np.load(cache / "welch.npy", allow_pickle=False)
    if values.shape != (len(rows), len(FEATURES)) or not np.isfinite(values).all():
        raise ValueError("Unexpected Welch cache")
    meta = pd.read_csv(pilot / "all_subjects_features_PROVISIONAL.csv")
    keys = ["subject_id","session_id","end_sample","label"]
    anchors = meta.assign(end_sample=np.rint(meta.window_end_ms * FS / 1000).astype(int))
    actual = rows.loc[rows.anchor, keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, anchors[keys].sort_values(keys).reset_index(drop=True))
    dates = []
    for session in json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))["sessions"]:
        raw = Path(session["source"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != session["source_sha256"]:
            raise ValueError(f"Raw recording has changed: {raw}")
        date, basis = measurement_date(session["note"], raw.name)
        dates.append(dict(subject_id=session["subject_id"], session_id=session["session_id"], day=date,
                          date_basis=basis, source_file=raw.name))
    return pd.concat([rows, pd.DataFrame(values, columns=FEATURES)],axis=1).merge(
        pd.DataFrame(dates), on=["subject_id","session_id"], validate="many_to_one")


def score_centres(model, frame):
    margin = ((frame[FEATURES].to_numpy() - model.mean) / model.scale) @ model.weights + model.bias
    sessions = frame.assign(margin=margin).groupby(["label","subject_id","session_id"]).margin.median()
    return sessions.groupby(level=["label","subject_id"]).mean().groupby(level="label").mean()


def variants(donor, model, history):
    """Estimate a personal intercept from classes actually observed on earlier days."""
    out = {"cold": model}
    delta = 0.
    if not history.empty:
        target = score_centres(model, history)
        source = score_centres(model, donor)
        delta = float(np.clip((source.loc[target.index] - target).mean(), -1., 1.))
    for name, weight in (("half_shift", .5), ("full_shift", 1.)):
        adjusted = deepcopy(model)
        adjusted.bias += weight * delta
        out[name] = adjusted
    out["pooled"] = fit(pd.concat([donor, history], ignore_index=True)) if len(history) else model
    # Combine the affine margins in a common feature coordinate system.
    # Averaging separately standardized coefficient vectors would be incorrect.
    pooled = out["pooled"]
    slope = np.asarray(pooled.weights) / pooled.scale
    pooled_at_source_center = pooled.bias + (np.asarray(model.mean) - pooled.mean) @ slope
    for fraction in (.25, .50, .75):
        mixed = deepcopy(model)
        mixed.weights = ((1-fraction)*np.asarray(model.weights) + fraction*slope*model.scale).tolist()
        mixed.bias = (1-fraction)*model.bias + fraction*pooled_at_source_center
        out[f"pooled_{fraction:.2f}"] = mixed
    return out


def synthetic_cases():
    cases = []
    for seed in (10007, 10009, 10037, 10039, 10061, 10067):
        for phase in (.05, .29, .56, .87):
            s = generate_session(alert_seconds=12 + phase, drowsy_seconds=10, seed=seed)
            raw = s.raw_eeg.to_numpy()
            ends = np.arange(WINDOW, len(s)+1, STEP)
            guard = RawArtifactGuard(WINDOW, FS, True)
            valid = ~np.fromiter((guard.update(x) for x in raw), bool)[ends-1]
            if s.poor_signal.gt(50).any():
                raise ValueError("Unexpected poor-quality simulation")
            cases.append((extract(raw, ends, True)["welch_blend"],valid,s.timestamp_ms.to_numpy()[ends-1],
                          s.loc[s.event.eq("STATE_DROWSY"),"timestamp_ms"].item()))
    return cases


def audit_locked_profiles(output, reference=None):
    """Fresh seed per trial; test frozen models on a broader simulator sample."""
    profiles = json.loads((output/"profiles.json").read_text(encoding="utf-8"))
    models = {(person,name):load_model(output/spec["file"]) for person,entries in profiles["people"].items()
              for name,spec in entries.items()}
    if reference:
        release = json.loads((reference/"release.json").read_text(encoding="utf-8"))
        spec = release["variants"]["LowRepeat"]
        path = reference/spec["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Changed reference model")
        if (release["window_s"], release["step_s"], release["consecutive"], spec["release_consecutive"]) != (3,.25,3,3):
            raise ValueError("Unexpected reference configuration")
        models["N16","app_Welch"] = load_model(path)
    records = []
    for i in range(128):
        seed = 10300+i
        fixture = generate_session(seed=seed, alert_seconds=12+(i%4)*.25,drowsy_seconds=12)
        raw = fixture.raw_eeg.to_numpy()
        ends = np.arange(WINDOW,len(raw)+1,STEP)
        guard = RawArtifactGuard(WINDOW,FS,True)
        valid = ~np.fromiter((guard.update(x) for x in raw),bool)[ends-1]
        values = extract(raw,ends,True)["welch_blend"]
        onset = fixture.loc[fixture.event.eq("STATE_DROWSY"),"timestamp_ms"].item()
        for (person,name),model in models.items():
            high = model.predict_proba(values)>=model.threshold
            metrics = alarm_onset_metrics(fixture.timestamp_ms.to_numpy()[ends-1],alarm_states(high,valid),onset)
            records.append(dict(seed=seed,person=person,model=name,**metrics))
    result=pd.DataFrame(records)
    result.to_csv(output/"audit_128.csv",index=False)
    print(result.groupby(["person","model"]).agg(cases=("seed","count"),early=("pre_onset_alarm","sum"),
          under_3s=("valid_under_3s","sum"),worst=("onset_to_alarm_ms","max")).to_string())


def evaluate(pilot, cache, public, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = load_local(pilot, cache)
    anchors = frame.loc[frame.anchor]
    public_rows = public_training_frame(public)
    public_model = fit(public_rows)
    cases = synthetic_cases()
    rows, latency = [], []
    for person, wearer in frame.groupby("subject_id"):
        other = anchors.loc[anchors.subject_id.ne(person)]
        archived = fit(other)
        for day, test in wearer.groupby("day"):
            past = wearer.loc[wearer.anchor & wearer.day.lt(day)]
            older_other = other.loc[other.day.lt(day)]
            for scenario in ("archived_other", "calendar"):
                donor, base, name = other, archived, "other_N16"
                if scenario == "calendar" and older_other.label.nunique() < 2:
                    donor, base, name = public_rows, public_model, "Cui_Fp1"
                elif scenario == "calendar" and len(older_other) != len(other):
                    donor, base = older_other, fit(older_other)
                for method, model in variants(donor,base,past).items():
                    common = dict(scenario=scenario,person=person,day=str(day.date()),method=method,donor=name,
                                  prior_sessions=past.session_id.nunique(),prior_classes=past.label.nunique())
                    for (session,block), group in test.groupby(["session_id","block"]):
                        valid, anchor = group.valid.to_numpy(), group.anchor.to_numpy()
                        high = model.predict_proba(group[FEATURES].to_numpy()) >= CUT
                        active = alarm_states(high,valid)
                        label = int(group.label.unique().item())
                        rows.append(dict(**common,session=session,block=block,label=label,
                                         correct=int((high[anchor]==label).sum()), anchors=int(anchor.sum()),
                                         valid=int(valid.sum()),active=int(active.sum()),
                                         episodes=int((active & ~np.r_[False,active[:-1]]).sum())))
                    results = [alarm_onset_metrics(times,alarm_states(model.predict_proba(x)>=CUT,valid),onset)
                               for x,valid,times,onset in cases]
                    delays = [r["onset_to_alarm_ms"] for r in results if r["onset_to_alarm_ms"] is not None]
                    latency.append(dict(**common, cases=len(cases), early=sum(r["pre_onset_alarm"] for r in results),
                                        under_3s=sum(r["valid_under_3s"] for r in results),worst_ms=max(delays,default=None)))
    measured = pd.DataFrame(rows)
    measured.to_csv(output/"session_results.csv", index=False)
    pd.DataFrame(latency).to_csv(output/"timing_by_day.csv", index=False)
    anchors[["subject_id","session_id","source_file","day","date_basis","label"]].drop_duplicates().to_csv(
        output/"days.csv",index=False)
    summaries = []
    for stage, subset in (("all", measured),("has_history",measured.loc[measured.prior_sessions.gt(0)]),
                          ("both_in_history",measured.loc[measured.prior_classes.eq(2)])):
        sums = subset.groupby(["scenario","method","person","label"])[["correct","anchors","valid","active","episodes"]].sum()
        sums["recall"] = sums.correct / sums.anchors
        sums["active_fraction"] = sums.active / sums.valid
        sums.reset_index().to_csv(output/f"by_person_{stage}.csv",index=False)
        for (scenario,method), g in sums.groupby(level=["scenario","method"]):
            states = g.groupby(level="label").mean()
            has_paired = all(group.index.get_level_values("label").nunique()==2 for _,group in g.groupby(level="person"))
            summaries.append(dict(stage=stage,scenario=scenario,method=method,
                awake=states.loc[0,"recall"],drowsy=states.loc[1,"recall"],
                ba=states.recall.mean() if has_paired else None,
                awake_time=states.loc[0,"active_fraction"],drowsy_time=states.loc[1,"active_fraction"],
                awake_episodes=int(g.xs(0,level="label").episodes.sum())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/"summary.csv",index=False)
    # Freeze the prespecified full-intercept correction for paired future replay.
    # Each wearer gets the same other-person reference used in the retrospective comparison.
    manifest = json.loads((pilot / "experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    registry = dict(training_raw_hashes=[s["source_sha256"] for s in manifest["sessions"]],
                    fs=FS, window_s=3, step_s=.25, consecutive=3, release_consecutive=3, people={})
    for person, own in anchors.groupby("subject_id"):
        other = anchors.loc[anchors.subject_id.ne(person)]
        models = variants(other, fit(other), own)
        registry["people"][person] = {}
        for method in ("cold", "full_shift", "pooled"):
            model = deepcopy(models[method])
            model.threshold, model.format_version, model.spectral_estimator = CUT, 2, "welch_blend"
            path = output / f"{person}_{method}.json"
            model.save(path)
            restored = load_model(path)
            np.testing.assert_allclose(restored.predict_proba(own[FEATURES]),models[method].predict_proba(own[FEATURES]))
            registry["people"][person][method] = dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output/"profiles.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","cache","public","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--audit-locked",action="store_true")
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    if args.audit_locked:
        audit_locked_profiles(args.output,args.reference)
    else:
        evaluate(args.pilot,args.cache,args.public,args.output)
