import csv
import json
from threading import Event
from zipfile import ZipFile

import numpy as np
import pytest

from n16_eeg.live import LiveDetector
from n16_eeg.model import LogisticModel
from n16_eeg.paired_replay import DETAILS, REPORT, compare_recording, kss_intervals


def detectors():
    return tuple(LiveDetector(LogisticModel(["theta_alpha"],[0.],[1.],[0.],bias),
                              window_s=3,step_s=.25,consecutive=3,screen_raw=True) for bias in (-4.,4.))


def recording(tmp_path, subject="TEST"):
    size=512*24
    values=(50*np.sin(np.arange(size)/9.)).astype(int)
    marks={0:3,6*512:4,8*512:6,10*512:8,16*512:7,20*512:7,23*512:8}
    with (tmp_path/"raw.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["sample_index","raw_segment","event","kss","raw_eeg","signal_time_ms",
                         "poor_signal","subject_id","session_id"])
        for i,value in enumerate(values):
            writer.writerow([i,int(i>=18*512),"KSS" if i in marks else "",marks.get(i,""),value,
                             i*1000/512,0,"TEST","TEST_PAIR"])
    archive=tmp_path/"session.zip"
    with ZipFile(archive,"w") as z:
        z.write(tmp_path/"raw.csv","raw.csv")
        z.writestr("raw.session.json",json.dumps(dict(samples=size,subject_id=subject,session_id="TEST_PAIR")))
    return archive


def test_zip_compares_identical_windows_and_abstains_across_gaps(tmp_path):
    target=tmp_path/"paired"
    result=compare_recording(recording(tmp_path),target,detectors(),Event())
    assert result["episodes",0]==0 and result["episodes",1]==2
    with (target/DETAILS).open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        end=int(row["sample_index"])
        start=end-1535
        expected=("0" if start>=0 and end<3072 else "1" if
                  (start>=5120 and end<8192 or start>=10240 and end<11776) else "")
        assert row["kss_class"]==expected
    assert result[0]>0 and result[1]>0 and (target/REPORT).exists()
    assert result[0,"active",1]<result[0] and result[1,"active",1]==result[1]
    assert kss_intervals([(0,0,0),(361*512,0,0)],512)==[]


def test_cancel_or_mixed_session_never_publishes_partial_result(tmp_path):
    source=recording(tmp_path,subject="OTHER")
    dest=tmp_path/"output"
    with pytest.raises(ValueError,match="không khớp"):
        compare_recording(source,dest,detectors(),Event())
    assert not dest.exists()
    stop=Event()
    stop.set()
    assert compare_recording(source,dest,detectors(),stop) is None and not dest.exists()
