"""Prospective replay for S04 using only that wearer's labelled earlier days."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_four_wearers import (FEATURES, FS, STEP, WINDOW, NEW, alarm_states, anchored_update,
                                   extract, reference, train_model, predict, alarm_onset_metrics,
                                   RawArtifactGuard, generate_session)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    base = reference()
    sessions = pd.read_csv(NEW/"session_results.csv")
    inventory = pd.read_csv(NEW/"archive_manifest.csv").set_index("member")
    parts = []
    for row in sessions.loc[sessions.person_folder.eq("S04") & sessions.use.eq("đối chiếu")].itertuples():
        for name in (row.file,row.note):
            entry=inventory.loc["S04/"+name]
            if hashlib.sha256(Path(entry.destination).read_bytes()).hexdigest()!=entry.sha256:
                raise ValueError(f"Changed study input: {name}")
        raw=pd.read_csv(Path("raw/Raw data/S04")/row.file,usecols=["Electrode"]).Electrode.to_numpy()
        replay=pd.read_csv(NEW/"predictions"/row.file)
        replay=replay.loc[replay.sample_index+1<=round(row.label_end_s*FS)]
        ends=replay.sample_index.to_numpy()+1
        part=pd.DataFrame(extract(raw,ends,True)["welch_blend"],columns=FEATURES)
        part=part.assign(subject_id="S04",session_id=row.file,label=int(row.label=="buồn ngủ"),
                         anchor=(ends%(4*FS)==0),valid=replay.risk.notna().to_numpy(),
                         day=pd.to_datetime(row.file.split("_")[1],format="%d%m%Y"))
        parts.append(part)
    frame=pd.concat(parts,ignore_index=True)
    history=frame.loc[frame.anchor & frame.valid]
    records=[]
    for day,data in frame.groupby("day"):
        past=history.loc[history.day<day]
        if past.label.nunique()!=2:
            continue
        personal=anchored_update(base,train_model(past,"balanced"),weight=.5)
        for session,test in data.groupby("session_id"):
            for name,model in [("legacy",base),("personal",personal)]:
                high=predict(model,test)>=.54
                valid=test.valid.to_numpy()
                active=alarm_states(high,valid)
                marked=test.anchor & test.valid
                records.append(dict(day=day.date(),session=session,variant=name,prior_sessions=past.session_id.nunique(),
                                    label=test.label.iloc[0],anchors=int(marked.sum()),positive=int(high[marked].sum()),
                                    valid=int(valid.sum()),active=int(active.sum()),
                                    onsets=int((active & ~np.r_[False,active[:-1]]).sum())))
    result=pd.DataFrame(records)
    result.to_csv(out/"future_sessions.csv",index=False)
    final=anchored_update(base,train_model(history,"balanced"),weight=.5)
    final.save(out/"S04.json")
    timing=[]
    for i in range(128):
        s=generate_session(seed=21000+i,alert_seconds=12+(i%4)*.25,drowsy_seconds=12)
        raw=s.raw_eeg.to_numpy()
        ends=np.arange(WINDOW,len(raw)+1,STEP)
        guard=RawArtifactGuard(WINDOW,FS,True)
        valid=~np.fromiter((guard.update(v) for v in raw),bool)[ends-1]
        x=extract(raw,ends,True)["welch_blend"]
        onset=s.loc[s.event.eq("STATE_DROWSY"),"timestamp_ms"].item()
        states=alarm_states(final.predict_proba(x)>=.54,valid)
        timing.append(dict(seed=21000+i,**alarm_onset_metrics(s.timestamp_ms.to_numpy()[ends-1],states,onset)))
    pd.DataFrame(timing).to_csv(out/"timing.csv",index=False)
    print(result.to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    run(parser.parse_args().out)
