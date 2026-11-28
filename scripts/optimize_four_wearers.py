"""Wearer-held-out comparison for the four-person NeuroSky development cohort."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from compare_ensembles import alarm_states
from develop_pilot_model import FEATURES
from evaluate_repeat_wearers import load_local
from evaluate_welch_alarms import FS, STEP, WINDOW, extract
from tune_runtime_pilot import fit
from n16_eeg.model import LogisticModel, load_model
from n16_eeg.alarm_eval import alarm_onset_metrics
from n16_eeg.quality import RawArtifactGuard
from n16_eeg.simulate import generate_session


METHODS = ("refit", "balanced", "bands_blink", "theta_alpha_blink", "shallow_boost")
SHAPE = ["raw_log_mad", "raw_mobility", "raw_peak_ratio", "raw_slow_shift", "raw_hjorth_complexity"]
CUTS = (.48, .54, .60)
REF = Path("output/N16_Research_20261128/welch_blend")
NEW = Path("output/N16_NewRaw_20261128")


def reference():
    spec = json.loads((REF/"release.json").read_text())["variants"]["LowRepeat"]
    path = REF/spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Reference model hash mismatch")
    return load_model(path)


def shape_features(raw, ends):
    """Robust amplitude, roughness and slow deflection from the same causal 3 s."""
    windows = np.lib.stride_tricks.sliding_window_view(raw, WINDOW)
    blocks = []
    for i in range(0, len(ends), 128):
        x = windows[ends[i:i+128]-WINDOW].astype(float)
        absolute = np.abs(x - np.median(x, axis=1, keepdims=True))
        mad = np.median(absolute, axis=1)
        dx = np.diff(x)
        mobility = dx.std(axis=1)/np.maximum(x.std(axis=1), 1e-8)
        complexity = np.diff(dx).std(axis=1)/np.maximum(dx.std(axis=1)*mobility, 1e-8)
        subframes = np.median(x.reshape(-1, 6, WINDOW//6), axis=2)
        blocks.append(np.column_stack((np.log1p(mad), mobility,
            np.quantile(absolute,.98,axis=1)/np.maximum(mad,1.),
            np.ptp(subframes,axis=1)/np.maximum(mad,1.), complexity)))
    return np.vstack(blocks)


def cohort(with_shape):
    old = load_local(Path("output/two_subjects_20261105_v1"), Path("output/N16_Ensemble_20261128"))
    if with_shape:
        for (person,name), group in old.groupby(["subject_id","source_file"]):
            raw = pd.read_csv(Path("raw/Raw data")/person/name,usecols=["Electrode"]).Electrode.to_numpy()
            old.loc[group.index,SHAPE] = shape_features(raw,group.end_sample.to_numpy())
    chunks = [old]
    manifest = pd.read_csv(NEW/"archive_manifest.csv").set_index("member")
    sessions = pd.read_csv(NEW/"session_results.csv")
    for row in sessions.loc[sessions.use.eq("đối chiếu")].itertuples():
        person = row.person_folder
        if person not in ("S04", "S05") or row.label not in ("tỉnh","buồn ngủ"):
            raise ValueError(f"Unexpected scope {row.file}")
        for name in (row.file, row.note):
            entry = manifest.loc[f"{person}/{name}"]
            path = Path(entry.destination)
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
                raise ValueError(f"Source changed: {name}")
        raw = pd.read_csv(Path("raw/Raw data")/person/row.file, usecols=["Electrode"]).Electrode.to_numpy()
        replay = pd.read_csv(NEW/"predictions"/row.file)
        replay = replay.loc[replay.sample_index+1 <= round(row.label_end_s*FS)]
        ends = replay.sample_index.to_numpy()+1
        if not np.array_equal(ends, np.arange(WINDOW, ends[-1]+1, STEP)):
            raise ValueError("Unexpected replay timing grid")
        values = extract(raw, ends, literature=True)["welch_blend"]
        valid = replay.risk.notna().to_numpy()
        risk = reference().predict_proba(values)
        np.testing.assert_allclose(risk[valid], replay.risk.to_numpy()[valid], rtol=0, atol=5.001e-5)
        frame = pd.DataFrame(values, columns=FEATURES)
        if with_shape:
            frame[SHAPE] = shape_features(raw,ends)
        frame = frame.assign(subject_id=person, session_id=Path(row.file).stem, block=1,
                             end_sample=ends, anchor=(ends % (4*FS) == 0), valid=valid,
                             label=int(row.label=="buồn ngủ"))
        assert (frame.valid & frame.anchor).sum() == row.anchors
        chunks.append(frame)
    return pd.concat(chunks, ignore_index=True)


def train_model(frame, method):
    if method == "refit":
        return fit(frame)
    # Give each class equal total weight, each wearer within class equal weight,
    # and balance the sessions of each wearer. S05 has drowsy sessions only.
    keys = ["label","subject_id","session_id"]
    counts = frame.groupby(keys).label.transform("size")
    persons = frame.groupby("label").subject_id.transform("nunique")
    sessions = frame.groupby(keys[:2]).session_id.transform("nunique")
    weights = 1/(counts*sessions*persons)
    weights = (weights/weights.mean()).to_numpy()
    cols = FEATURES
    if method == "bands_blink":
        cols = FEATURES[:5] + FEATURES[8:]
    elif method == "theta_alpha_blink":
        cols = ["rel_theta","rel_alpha","theta_alpha"] + FEATURES[-3:]
    x = frame[cols].to_numpy()
    y = frame.label.to_numpy()
    if method in ("shape_linear","shape_boost"):
        cols = FEATURES + SHAPE
        x = frame[cols].to_numpy()
    if method in ("shallow_boost","shape_boost"):
        model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=5, min_samples_leaf=50,
                    learning_rate=.05, l2_regularization=2., early_stopping=False, random_state=16)
        return model.fit(frame[cols], y, sample_weight=weights)
    scaler = RobustScaler().fit(x)
    clf = LogisticRegression(C=.03, max_iter=2000).fit(scaler.transform(x), y, sample_weight=weights)
    return LogisticModel(cols, scaler.center_.tolist(), scaler.scale_.tolist(), clf.coef_[0].tolist(),
                         float(clf.intercept_[0]), format_version=2, spectral_estimator="welch_blend")


def predict(model, frame):
    if isinstance(model, LogisticModel):
        return model.predict_proba(frame[model.feature_names].to_numpy())
    return model.predict_proba(frame[list(model.feature_names_in_)])[:,1]


def anchored_update(prior, update, weight=.20):
    """Blend the two margins and export one equivalent linear runtime model."""
    if prior.feature_names != FEATURES or update.feature_names != FEATURES:
        raise ValueError("Incompatible Welch models")
    old = np.array(prior.weights)/prior.scale
    new = np.array(update.weights)/update.scale
    coeff = (1-weight)*old + weight*new
    intercept = (1-weight)*(prior.bias-old@prior.mean) + weight*(update.bias-new@update.mean)
    return LogisticModel(FEATURES,[0.]*len(coeff),[1.]*len(coeff),coeff.tolist(),float(intercept),
                         threshold=.54, format_version=2,spectral_estimator="welch_blend")


def recalls(rows, positive):
    y = rows.label.to_numpy()
    return {col:float((positive[y==k]==k).mean()) if (y==k).any() else np.nan
            for k,col in enumerate(("awake","drowsy"))}


def run(out, with_shape):
    out.mkdir(exist_ok=True, parents=True)
    frame = cohort(with_shape)
    methods = METHODS + (("shape_linear","shape_boost") if with_shape else ())
    train = frame.loc[frame.valid & frame.anchor]
    people = tuple(sorted(train.subject_id.unique()))
    assert people == ("S01","S02","S04","S05")
    train.groupby(["subject_id","session_id","label"]).size().rename("anchors").to_csv(out/"cohort.csv")

    @lru_cache(None)
    def model_for(donors, method):
        return train_model(train.loc[train.subject_id.isin(donors)], method)

    @lru_cache(None)
    def folds(donors, person, method):
        test = train.loc[train.subject_id.eq(person)]
        scores = predict(model_for(donors, method), test)
        return [recalls(test, scores>=cut) for cut in CUTS]

    def choose(donors):
        options = []
        for method in methods:
            per_person = [folds(tuple(p for p in donors if p != held), held, method) for held in donors]
            for i,cut in enumerate(CUTS):
                avg = pd.DataFrame([r[i] for r in per_person]).mean()
                if avg.isna().any():
                    raise ValueError("Both classes needed for model selection")
                # Reward balanced recall and the weaker of the two recalls.
                objective = (avg.mean()+avg.min())/2
                options.append((objective,-abs(cut-.54),-methods.index(method),method,cut))
        return max(options)[-2:]

    records, per_session, candidates = [], [], []
    for person in people:
        donors = tuple(p for p in people if p != person)
        choice, threshold = choose(donors)
        test = frame.loc[frame.subject_id.eq(person)].copy()
        for method in methods:
            for cut, stats in zip(CUTS,folds(donors,person,method)):
                candidates.append(dict(person=person,method=method,threshold=cut,**stats))
        # Legacy recipe for old wearers is held-out; new wearers use actual frozen HUD.
        historic = (fit(train.loc[train.subject_id.isin({"S01","S02"}-{person})])
                    if person in ("S01","S02") else reference())
        anchored = anchored_update(historic,model_for(donors,"balanced"))
        for variant,model,cut,name,on,off in [
            ("legacy",historic,.54,"legacy",3,3),
            ("refit",model_for(donors,"refit"),.54,"refit",3,3),
            ("nested",model_for(donors,choice),threshold,choice,3,3),
            ("anchored",anchored,.54,"anchored",3,3),
            ("anchored_fast",anchored,.54,"anchored",2,3),
            ("legacy_fast",historic,.54,"legacy",2,3)]:
            scores = predict(model,test)
            high = scores >= cut
            anchor = (test.anchor & test.valid).to_numpy()
            stats = recalls(test.loc[anchor],high[anchor])
            for (session,block),idx in test.groupby(["session_id","block"],sort=False).indices.items():
                segment=test.iloc[idx]
                if not segment.end_sample.diff().dropna().eq(STEP).all() or segment.label.nunique()!=1:
                    raise ValueError("Unexpected block boundary")
                valid=segment.valid.to_numpy()
                active=alarm_states(high[idx],valid,on,off)
                per_session.append(dict(variant=variant, person=person, session=session, block=block,
                    label=segment.label.iloc[0],valid=int(valid.sum()),active=int(active.sum()),
                    episodes=int((active & ~np.r_[False,active[:-1]]).sum()),
                    anchors=int(anchor[idx].sum()),positive=int(high[idx][anchor[idx]].sum())))
            records.append(dict(variant=variant,person=person,method=name,threshold=cut,**stats))
        print("Held out",person,choice,threshold,flush=True)
    result=pd.DataFrame(records)
    result.to_csv(out/"held_out.csv",index=False)
    pd.DataFrame(per_session).to_csv(out/"blocks.csv",index=False)
    pd.DataFrame(candidates).to_csv(out/"candidate_folds.csv",index=False)
    final_method,cut = choose(people)
    chosen = model_for(people,final_method)
    refit = model_for(people,"refit")
    frozen = reference()
    anchored = anchored_update(frozen,model_for(people,"balanced"))
    anchored.save(out/"anchored.json")
    for name, model, threshold in [("candidate",chosen,cut),("refit",refit,.54)]:
        if isinstance(model,LogisticModel):
            model.threshold=threshold
            model.spectral_estimator="welch_blend"
            model.format_version=2
            model.save(out/(name+".json"))
        else:
            joblib.dump(dict(model=model,features=list(model.feature_names_in_),threshold=threshold,
                             spectral_estimator="welch_blend"),out/(name+".joblib"))
    timing=[]
    for i in range(128):
        seed=21000+i
        s=generate_session(seed=seed,alert_seconds=12+(i%4)*.25,drowsy_seconds=12)
        raw=s.raw_eeg.to_numpy()
        ends=np.arange(WINDOW,len(raw)+1,STEP)
        values=pd.DataFrame(extract(raw,ends,True)["welch_blend"],columns=FEATURES)
        if with_shape:
            values[SHAPE] = shape_features(raw,ends)
        guard=RawArtifactGuard(WINDOW,FS,True)
        valid=~np.fromiter((guard.update(v) for v in raw),bool)[ends-1]
        if s.poor_signal.gt(50).any():
            raise ValueError("Unexpected poor contact in timing fixture")
        onset=s.loc[s.event.eq("STATE_DROWSY"),"timestamp_ms"].item()
        for variant,model,th,on in [("legacy",frozen,.54,3),("refit",refit,.54,3),("nested",chosen,cut,3),
                ("anchored",anchored,.54,3),("anchored_fast",anchored,.54,2),("legacy_fast",frozen,.54,2)]:
            active=alarm_states(predict(model,values)>=th,valid,on,3)
            timing.append(dict(variant=variant,seed=seed,
                **alarm_onset_metrics(s.timestamp_ms.to_numpy()[ends-1],active,onset)))
    pd.DataFrame(timing).to_csv(out/"timing.csv",index=False)
    (out/"selection.json").write_text(json.dumps(dict(method=final_method,threshold=cut,
        training_people=people,window_s=3,step_s=.25,confirmation=3,release=3),indent=2),encoding="utf8")
    print(result.to_string(index=False),flush=True)
    print("Selected for all people",final_method,cut,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--shape", action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out,args.shape)
