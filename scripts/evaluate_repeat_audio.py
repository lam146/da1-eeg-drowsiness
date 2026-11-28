"""Replay the fixed four-wearer model to measure the cost of grouping nearby tones."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from optimize_four_wearers import FS, STEP, alarm_states, anchored_update, cohort, fit, predict, reference, train_model


class RepeatSound:
    """Debounce only audible requests. A renewed sustained alert still gets a tone."""

    def __init__(self, interval_s=0.):
        self.interval_s = interval_s
        self.reset()

    def reset(self):
        self.last_request = None
        self.played_this_episode = False

    def update(self, now, alarm, valid=True, enabled=True):
        if not valid:
            self.reset()
            return False
        if not alarm:
            self.played_this_episode = False
            return False
        if not enabled:
            # Muting cancels a pending tone. Re-enabling applies to the next episode.
            self.played_this_episode = True
            self.last_request = None
            return False
        if self.played_this_episode:
            return False
        if self.last_request is not None and now - self.last_request < self.interval_s:
            return False
        self.last_request = now
        self.played_this_episode = True
        return True


def run(out):
    frame = cohort(False)
    train = frame.loc[frame.anchor & frame.valid]
    rows = []
    for person, data in frame.groupby("subject_id"):
        prior = fit(train.loc[train.subject_id.isin({"S01","S02"}-{person})]) if person in ("S01","S02") else reference()
        model = anchored_update(prior, train_model(train.loc[train.subject_id.ne(person)], "balanced"))
        for (session, block), part in data.groupby(["session_id","block"]):
            valid = part.valid.to_numpy()
            times = (part.end_sample.to_numpy()-1)/FS
            if not np.allclose(np.diff(times), STEP/FS) or part.label.nunique()!=1:
                raise ValueError("Invalid replay block")
            active = alarm_states(predict(model, part)>=.545,valid,3,3)
            onsets = np.flatnonzero(active & ~np.r_[False,active[:-1]])
            for interval in (0,2,3,5,8):
                gate = RepeatSound(interval)
                requested = np.array([gate.update(t,a,v) for t,a,v in zip(times,active,valid,strict=True)])
                sounded = np.flatnonzero(requested)
                index = np.searchsorted(onsets,sounded,side="right")-1
                delays = times[sounded]-times[onsets[index]]
                if (len(np.unique(index))!=len(index) or (len(onsets) and (not len(sounded) or sounded[0]!=onsets[0]))):
                    raise ValueError("Unexpected audio event mapping")
                if interval==0 and not np.array_equal(onsets,sounded):
                    raise ValueError("Default audio differs")
                rows.append(dict(person=person,session=session,block=block,label=part.label.iloc[0],interval=interval,
                    valid_s=valid.sum()*STEP/FS,episodes=len(onsets),requests=len(sounded),
                    immediate=int((delays==0).sum()),delayed=int((delays>0).sum()),delayed_ge3=int((delays>=3).sum()),
                    skipped=len(onsets)-len(sounded),max_delay_s=delays.max(initial=0.),
                    repeat_gaps_lt5=int((np.diff(times[onsets])<5).sum())))
    result=pd.DataFrame(rows)
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"audio_by_block.csv",index=False)
    summary=result.groupby(["label","interval"]).agg({**dict.fromkeys(["episodes","requests","delayed","skipped","delayed_ge3","repeat_gaps_lt5","valid_s"],"sum"),"max_delay_s":"max"})
    summary.to_csv(out/"audio_summary.csv")
    print(summary.to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
