"""Compare fixed score mixtures, with outer wearer and inner recording-day holdouts."""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from develop_pilot_model import FEATURES, session_weights
from evaluate_welch_alarms import FS, WINDOW, STEP, collect, extract
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.quality import RawArtifactGuard
from n16_eeg.simulate import generate_session
from tune_runtime_pilot import fit
from pilot_calendar import measurement_date

PARTS = ("joint", "spectral", "blink", "rbf", "public")
WEIGHTS = np.array([v for v in product(range(5), repeat=len(PARTS)) if sum(v) == 4]) / 4
GRID = [(w, cut) for w in WEIGHTS for cut in (.50, .54, .58)]
BASE = next(i for i, (w, cut) in enumerate(GRID) if w[0] == 1 and cut == .54)


def component_models(frame, public_model):
    rbf = make_pipeline(RobustScaler(), SVC(C=.5, gamma="scale"))
    rbf.fit(frame[FEATURES], frame.label, svc__sample_weight=session_weights(frame))
    return [fit(frame), fit(frame, FEATURES[:-3]), fit(frame, FEATURES[-3:]), rbf, public_model]


def margins(models, frame):
    result = []
    for model in models:
        if hasattr(model, "decision_function"):
            result.append(model.decision_function(frame[FEATURES]))
        else:
            x = frame[model.feature_names].to_numpy()
            result.append(((x - model.mean) / model.scale) @ model.weights + model.bias)
    return np.column_stack(result)


def public_training_frame(path):
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    if digest.hexdigest() != "53cc4ef14b1343f7f3fb5322dd2b541c031c6c2297bddf1817aed04dc687a6a4":
        raise ValueError("Unexpected public dataset version")
    data = loadmat(path)
    eeg = resample_poly(data["EEGsample"][:, 0, :], 4, 1, axis=1)
    values = extract(eeg.ravel(), WINDOW * np.arange(1, len(eeg)+1), True)["welch_blend"]
    frame = pd.DataFrame(values, columns=FEATURES)
    frame["label"] = data["substate"].ravel()
    frame["subject_id"] = frame["session_id"] = data["subindex"].ravel()
    return frame


def train_public(path):
    return fit(public_training_frame(path))


def decisions(scores, candidate):
    weights, cut = GRID[candidate]
    return scores @ weights >= np.log(cut / (1 - cut))


def alarm_states(high, valid, on=3, off=3):
    """Consecutive highs to start; consecutive lows or a quality gap to clear."""
    high = np.asarray(high, bool)
    valid = np.asarray(valid, bool)
    up, down = high & valid, ~high & valid
    indices = np.arange(len(high))
    start, stop = np.zeros(len(high), bool), ~valid.copy()
    if len(up) >= on:
        start[on-1:] = np.lib.stride_tricks.sliding_window_view(up, on).all(axis=1)
    if len(down) >= off:
        stop[off-1:] |= np.lib.stride_tricks.sliding_window_view(down, off).all(axis=1)
    return np.maximum.accumulate(np.where(start, indices, -1)) > np.maximum.accumulate(np.where(stop, indices, -1))


def class_recalls(truth, high):
    return float((~high[truth == 0]).mean()), float(high[truth == 1].mean())


def both_modalities(i):
    joint, spectral, blink, rbf, public = GRID[i][0]
    return joint + rbf + public > 0 or (spectral > 0 and blink > 0)


def inner_choice(frame, public_model):
    scores = np.empty((len(frame), len(PARTS)))
    for day in sorted(frame.date.unique()):
        hold = frame.date.eq(day)
        train = frame.loc[~hold]
        if train.label.nunique() != 2:
            raise ValueError("Inner training split lacks a class")
        scores[hold] = margins(component_models(train, public_model), frame.loc[hold])
    performance = [np.mean(class_recalls(frame.label.to_numpy(), decisions(scores, i))) for i in range(len(GRID))]
    return max((i for i in range(len(GRID)) if both_modalities(i)),
               key=lambda i: (performance[i], -abs(GRID[i][1] - .54), GRID[i][0][0]))


def synthetic_scores(models, seeds):
    cases = []
    for seed in seeds:
        for phase in (.05, .29, .56, .87):
            s = generate_session(alert_seconds=12+phase, drowsy_seconds=10, seed=seed)
            raw = s.raw_eeg.to_numpy(float)
            ends = np.arange(WINDOW, len(raw)+1, STEP)
            values = extract(raw, ends, True)["welch_blend"]
            guard = RawArtifactGuard(WINDOW, FS, True)
            valid = ~np.fromiter((guard.update(v) for v in raw), bool)[ends-1]
            if s.poor_signal.gt(50).any():
                raise ValueError("Unexpected poor contact in timing fixture")
            cases.append((s.timestamp_ms.to_numpy()[ends-1], s.loc[s.event.eq("STATE_DROWSY"),"timestamp_ms"].item(),
                          margins(models, pd.DataFrame(values, columns=FEATURES)), valid))
    return cases


def timing(cases, candidate, on=3, off=3):
    results = [alarm_onset_metrics(t, alarm_states(decisions(scores, candidate), valid, on, off), onset)
               for t, onset, scores, valid in cases]
    valid = [r["onset_to_alarm_ms"] for r in results if r["onset_to_alarm_ms"] is not None]
    return dict(cases=len(cases), under_3s=sum(r["valid_under_3s"] for r in results),
                early=sum(r["pre_onset_alarm"] for r in results), worst_ms=max(valid, default=None))


def sweep_alarm(out):
    table = pd.read_csv(out/"comparison.csv").set_index("candidate")
    base = table.loc[BASE]
    # Changing alarm persistence cannot change the classifier's window recalls.
    viable = [i for i,row in table.iterrows() if both_modalities(i) and row.ba >= base.ba
              and row.drowsy >= base.drowsy]
    frame = pd.concat([pd.read_csv(out/"window_index.csv"),
                       pd.DataFrame(np.load(out/"welch.npy"), columns=FEATURES)],axis=1)
    fit_all = joblib.load(out/"candidate_OFFLINE.joblib")["models"]
    oof = []
    for person, test in frame.groupby("subject_id"):
        train = frame.loc[frame.anchor & frame.subject_id.ne(person)]
        values = margins(component_models(train, fit_all[-1]), test)
        for indices in test.groupby(["session_id","block"]).indices.values():
            block = test.iloc[indices]
            oof.append((person,int(block.label.iloc[0]),values[indices],block.valid.to_numpy()))
    dev = synthetic_scores(fit_all,(401,409,419,431,443,457,2003,2011,2017,2027,2029,2039))
    rows = []
    for i in viable:
        for on,off in product((2,3,4,5), (2,3,4,5)):
            states = []
            for person,label,scores,valid in oof:
                active = alarm_states(decisions(scores,i),valid,on,off)
                starts = (active & ~np.r_[False,active[:-1]]).sum()
                states.append((person,label,valid.sum(),active.sum(),starts))
            counts = pd.DataFrame(states,columns=["person","label","valid","active","starts"]).groupby(["label","person"]).sum()
            fractions = (counts.active/counts.valid).groupby("label").mean()
            rows.append(dict(candidate=i,on=on,off=off,ba=table.loc[i,"ba"],awake=table.loc[i,"awake"],
                             drowsy=table.loc[i,"drowsy"],awake_time=fractions[0],drowsy_time=fractions[1],
                             awake_episodes=int(counts.loc[0,"starts"].sum()),**timing(dev,i,on,off)))
    policies = pd.DataFrame(rows)
    winner_pool = policies.loc[(policies.under_3s == 48)&(policies.awake_time<=base.awake_time+1e-12)
        &(policies.drowsy_time>=base.drowsy_time-1e-12)&(policies.awake_episodes<=base.awake_episodes)]
    by_rank = winner_pool.sort_values(["ba","awake_episodes","awake_time"],ascending=[False,True,True])
    winner = by_rank.iloc[0]
    policies.to_csv(out/"alarm_policies.csv",index=False)
    i,on,off = (int(winner[k]) for k in ("candidate","on","off"))
    extra = synthetic_scores(fit_all,(8111,8123,8147,8161,8171,8191))
    output = dict(candidates_passing_classifier_screen=len(viable), policies_tested=len(policies),
                  qualified=len(by_rank), candidate=i,on=on,off=off, weights=dict(zip(PARTS,GRID[i][0])),
                  threshold=GRID[i][1], metrics=winner.to_dict(), audit=timing(extra,i,on,off),
                  baseline_audit=timing(extra,BASE))
    (out/"policy_result.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2),flush=True)


def run(pilot, public, out, reuse):
    out.mkdir(exist_ok=True, parents=True)
    if not reuse:
        blocks = collect(pilot, literature=True)
        pd.concat([m for m,_ in blocks], ignore_index=True).to_csv(out/"window_index.csv", index=False)
        np.save(out/"welch.npy", np.vstack([v["welch_blend"] for _,v in blocks]))
    meta = pd.read_csv(out/"window_index.csv")
    frame = pd.concat([meta, pd.DataFrame(np.load(out/"welch.npy"), columns=FEATURES)], axis=1)
    manifest = json.loads((pilot/"experiment_PROVISIONAL.json").read_text(encoding="utf-8"))
    dates = pd.DataFrame([dict(subject_id=s["subject_id"],session_id=s["session_id"],
                               date=measurement_date(s["note"],s["source"])[0]) for s in manifest["sessions"]])
    training = frame.loc[frame.anchor].merge(dates[["subject_id","session_id","date"]].drop_duplicates(),
        on=["subject_id","session_id"], validate="many_to_one")
    if training.date.isna().any() or len(training) != 3339 or set(training.subject_id) != {"S01","S02"}:
        raise ValueError("Unexpected cohort")
    public_model = train_public(public)
    folds, selected = [], {}
    for person in sorted(training.subject_id.unique()):
        train = training.loc[training.subject_id.ne(person)]
        selected[person] = inner_choice(train, public_model)
        test = frame.loc[frame.subject_id.eq(person)].reset_index(drop=True)
        scores = margins(component_models(train, public_model), test)
        segments = list(test.groupby(["session_id","block"]).indices.values())
        label, anchor, valid = (test[c].to_numpy() for c in ("label","anchor","valid"))
        for i in range(len(GRID)):
            high = decisions(scores, i)
            awake, drowsy = class_recalls(label[anchor], high[anchor])
            active = np.zeros(len(test), bool)
            starts = np.zeros(len(test), bool)
            for idx in segments:
                trace = alarm_states(high[idx], valid[idx])
                active[idx] = trace
                starts[idx] = trace & ~np.r_[False,trace[:-1]]
            folds.append(dict(candidate=i, person=person, awake=awake, drowsy=drowsy, ba=(awake+drowsy)/2,
                         awake_episodes=int(starts[label==0].sum()), awake_time=active[(label==0)&valid].mean(),
                         drowsy_time=active[(label==1)&valid].mean()))
        print("held out", person, "inner choice", selected[person], flush=True)
    folds = pd.DataFrame(folds)
    folds.to_csv(out/"folds.csv",index=False)
    table = folds.groupby("candidate").agg({**dict.fromkeys(["awake","drowsy","ba","awake_time","drowsy_time"], "mean"),
                                           "awake_episodes":"sum"})
    full = component_models(training, public_model)
    dev = synthetic_scores(full, (401,409,419,431,443,457,2003,2011,2017,2027,2029,2039))
    for i,(w,cut) in enumerate(GRID):
        for name,value in zip(PARTS,w):
            table.loc[i,name] = value
        table.loc[i,"threshold"] = cut
        for name,value in timing(dev,i).items():
            table.loc[i,name] = value
    base = table.loc[BASE]
    eligible = table.loc[[both_modalities(i) for i in table.index]]
    eligible = eligible.loc[(eligible.under_3s==48) & (eligible.awake_episodes<=base.awake_episodes)
        & (eligible.awake_time<=base.awake_time) & (eligible.drowsy>=base.drowsy) & (eligible.ba>=base.ba)]
    best = int(eligible.sort_values(["ba","awake_episodes"],ascending=[False,True]).index[0])
    highest = int(table.ba.idxmax())
    nested = pd.concat([folds.loc[(folds.person==p)&(folds.candidate==i)] for p,i in selected.items()])
    nested.to_csv(out/"nested.csv",index=False)
    audit = synthetic_scores(full, (6073,6079,6089,6091,6101,6113))
    checked = {str(i):timing(audit,i) for i in sorted({highest,best,BASE})}
    table.sort_values("ba",ascending=False).to_csv(out/"comparison.csv")
    for filename,choice in [("candidate_OFFLINE.joblib",best), ("highest_BA_OFFLINE.joblib",highest)]:
        joblib.dump(dict(parts=PARTS, models=full, features=FEATURES, weights=GRID[choice][0],
                         threshold=GRID[choice][1], status="DEVELOPMENT", spectral_estimator="welch_blend"),
                    out/filename)
    report = dict(baseline=BASE, highest_ba=highest, constrained_best=best, eligible=len(eligible),
                  nested_selection=selected, extra_synthetic=checked, configs=len(table))
    (out/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report, table.loc[[BASE,highest,best]].to_string(),nested.to_string(),sep="\n",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--reuse", action="store_true", help="Reuse this run's unchanged Welch feature cache")
    parser.add_argument("--sweep-alarm", action="store_true", help="Compare alarm persistence for eligible classifiers from a completed run")
    args=parser.parse_args()
    if args.sweep_alarm:
        sweep_alarm(args.out)
    else:
        run(args.pilot,args.public,args.out,args.reuse)
