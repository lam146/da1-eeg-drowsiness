"""Select a confirmation policy on development recordings, then test fresh onsets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from optimize_four_wearers import (FS, STEP, WINDOW, REF, alarm_states, anchored_update, cohort,
                                   fit, predict, reference, train_model, extract, RawArtifactGuard,
                                   generate_session, alarm_onset_metrics)
from n16_eeg.model import load_model


def cases(start, count, models):
    for i in range(count):
        seed=start+i
        signal=generate_session(seed=seed,alert_seconds=12+(i%4)*.25,drowsy_seconds=12)
        ends=np.arange(WINDOW,len(signal)+1,STEP)
        raw=signal.raw_eeg.to_numpy()
        guard=RawArtifactGuard(WINDOW,FS,True)
        valid=~np.fromiter((guard.update(v) for v in raw),bool)[ends-1]
        assert not signal.poor_signal.gt(50).any()
        x=extract(raw,ends,True)["welch_blend"]
        yield (seed,{name:m.predict_proba(x) for name,m in models.items()},valid,
               signal.timestamp_ms.to_numpy()[ends-1],signal.loc[signal.event.eq("STATE_DROWSY"),"timestamp_ms"].item())


def run(out):
    data=cohort(False)
    train=data.loc[data.valid & data.anchor]
    base=reference()
    updated=anchored_update(base,train_model(train,"balanced"))
    np.testing.assert_allclose(updated.predict_proba(train[base.feature_names].to_numpy()),
        load_model(out/"anchored.json").predict_proba(train[base.feature_names].to_numpy()))
    blocks=[]
    for person,rows in data.groupby("subject_id"):
        prior=fit(train.loc[train.subject_id.isin({"S01","S02"}-{person})]) if person in ("S01","S02") else base
        model=anchored_update(prior,train_model(train.loc[train.subject_id.ne(person)],"balanced"))
        p=rows.assign(legacy=predict(prior,rows),anchored=predict(model,rows))
        blocks.extend(v for _,v in p.groupby(["session_id","block"]))
    dev=list(cases(21000,128,{"legacy":base,"anchored":updated}))
    configs=[("legacy",.54,3,3)] + [("anchored",t,n,r) for t in (.54,.545,.55,.555,.56) for n in (2,3) for r in (2,3,4)]
    totals=[]
    detailed=[]
    for kind,t,n,r in configs:
        records=[]
        for block in blocks:
            y=int(block.label.iloc[0])
            high=block[kind].to_numpy()>=t
            valid=block.valid.to_numpy()
            active=alarm_states(high,valid,n,r)
            marked=block.anchor.to_numpy() & valid
            records.append(dict(person=block.subject_id.iloc[0],label=y,valid=valid.sum(),
                active=active.sum(),anchors=marked.sum(),correct=(high[marked]==y).sum(),
                episodes=(active & ~np.r_[False,active[:-1]]).sum()))
        group=pd.DataFrame(records).groupby(["label","person"]).sum()
        rates=(group.correct/group.anchors).unstack("label")
        for (label,person),row in group.iterrows():
            detailed.append(dict(kind=kind,cut=t,on=n,off=r,person=person,label=label,**row.to_dict()))
        awake=group.loc[0].sum()
        sleepy=group.loc[1].sum()
        delays=[alarm_onset_metrics(ts,alarm_states(scores[kind]>=t,valid,n,r),onset)
                for _,scores,valid,ts,onset in dev]
        totals.append(dict(kind=kind,cut=t,on=n,off=r,ba=rates.dropna().mean(axis=1).mean(),
            awake_recall=rates[0].mean(),drowsy_recall=rates[1].mean(),awake_episodes=int(awake.episodes),
            awake_active=awake.active/awake.valid,drowsy_active=sleepy.active/sleepy.valid,
            timing_ok=sum(d["valid_under_3s"] for d in delays)))
    table=pd.DataFrame(totals)
    old=table.iloc[0]
    table.to_csv(out/"policy_grid.csv",index=False)
    pd.DataFrame(detailed).to_csv(out/"policy_by_person.csv",index=False)
    # Require no increase in nuisance alarms, at most two percentage points lost
    # on sleepy recall and no loss in mean balanced accuracy on complete wearers.
    eligible=table.loc[table.kind.eq("anchored") & (table.awake_episodes<=old.awake_episodes)
        & (table.awake_active<=old.awake_active) & (table.drowsy_recall>=old.drowsy_recall-.02)
        & (table.ba>=old.ba) & (table.timing_ok>=old.timing_ok)]
    if eligible.empty:
        raise RuntimeError("No anchored candidate met the development criteria")
    best=eligible.sort_values(["timing_ok","ba","awake_episodes"],ascending=[False,False,True]).iloc[0]
    final=[]
    for seed,scores,valid,ts,onset in cases(31000,256,{"legacy":base,"anchored":updated}):
        for name,row in [("legacy",old),("candidate",best)]:
            active=alarm_states(scores[row.kind]>=row.cut,valid,int(row.on),int(row.off))
            final.append(dict(variant=name,seed=seed,**alarm_onset_metrics(ts,active,onset)))
    result=pd.DataFrame(final)
    result.to_csv(out/"fresh_256.csv",index=False)
    profile=out/"profile"
    profile.mkdir(exist_ok=True)
    updated.threshold=float(best.cut)
    path=profile/"combined.json"
    updated.save(path)
    manifest=json.loads((REF/"release.json").read_text())
    manifest.update(release="N16_Welch_Anchored_4",status="DEVELOPMENT_ONLY",consecutive=int(best.on))
    manifest["variants"]["LowRepeat"].update(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        threshold=updated.threshold,release_consecutive=int(best.off))
    shutil.copyfile(REF/manifest["fixture"]["file"],profile/manifest["fixture"]["file"])
    (profile/"release.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (out/"selected_policy.json").write_text(best.to_json(indent=2),encoding="utf-8")
    print(best.to_string(),flush=True)
    print(result.groupby("variant").agg(ok=("valid_under_3s","sum"),early=("pre_onset_alarm","sum"),
                                        max_ms=("onset_to_alarm_ms","max")).to_string(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
