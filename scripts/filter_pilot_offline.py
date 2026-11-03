"""Create an offline diagnostic EEG copy, never an automatically labeled ML dataset."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, tf2sos, sosfiltfilt


def line_fraction(samples, fs=512):
    n=2048
    if len(samples)<n:
        return None
    z=np.asarray(samples[:len(samples)//n*n]).reshape(-1,n)
    good=np.isfinite(z).all(axis=1)
    if not good.any():return None
    z=z[good]-z[good].mean(axis=1,keepdims=True)
    w=np.hanning(n);f=np.fft.rfftfreq(n,1/fs)
    p=np.abs(np.fft.rfft(z*w,axis=1))**2/(fs*np.sum(w*w));p[:,1:-1]*=2
    def integ(lo,hi):
        m=(f>=lo)&(f<hi)
        return np.trapezoid(p[:,m],f[m],axis=1)
    return float(np.median(integ(49,51)/np.maximum(integ(.5,100),np.finfo(float).eps)))


def run(source, target):
    source=source.resolve()
    original=source.read_bytes();sha=hashlib.sha256(original).hexdigest()
    d=pd.read_csv(source)
    t=d['Time:512Hz'].to_numpy(float);x=d['Electrode'].to_numpy(float)
    if not np.isfinite(t).all() or not np.isfinite(x).all() or not np.allclose(np.diff(t),1/512,atol=1e-9,rtol=0):
        raise ValueError('Need finite continuous 512 Hz input; do not fill missing samples automatically')
    flags=np.zeros(len(x),dtype=bool)
    rail=(x<=-2048)|(x>=2047)
    outside=(x < -2048)|(x>2047)
    blocks=[]
    for a in range(0,len(x),2048):
        b=min(a+2048,len(x))
        if rail[a:b].any():
            flags[a:b]=True
            blocks.append(dict(start_s=a/512,end_s=b/512,reason='raw_outside_expected_asic_range' if outside[a:b].any() else 'raw_at_expected_asic_rail'))
    # Flag prolonged identical samples before filtering, without guessing the cause.
    edges=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1,len(x)]
    for a,b in zip(edges[:-1],edges[1:]):
        if b-a>=128:
            left=a//2048*2048;right=min(len(x),((b+2047)//2048)*2048)
            flags[left:right]=True
            blocks.append(dict(start_s=left/512,end_s=right/512,reason='identical_raw_run_at_least_0p25s'))
    bn,an=iirnotch(50,30,fs=512)
    sos=np.vstack([tf2sos(bn,an),butter(4,[.5,45],btype='bandpass',fs=512,output='sos')])
    filtered=np.full(len(x),np.nan)
    good=~flags
    e=np.flatnonzero(np.diff(np.r_[False,good,False]))
    segments=[]
    edge_guard=np.zeros(len(x),dtype=bool)
    for a,b in zip(e[::2],e[1::2]):
        if b-a<2048:
            flags[a:b]=True
            blocks.append(dict(start_s=a/512,end_s=b/512,reason='continuous_segment_shorter_than_4s'))
            continue
        filtered[a:b]=sosfiltfilt(sos,x[a:b])
        # Mark conservative filter-boundary regions for review, never call them usable.
        edge_guard[a:min(b,a+2048)]=True;edge_guard[max(a,b-2048):b]=True
        segments.append(dict(start_s=a/512,end_s=b/512))
    status=np.where(flags,'exclude_screening',np.where(edge_guard,'review_filter_edge','not_flagged_by_screening'))
    target.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(dict(sample_index=np.arange(len(x)),time_s=t,
                      eeg_filtered_count=filtered,quality_screen=status)).to_csv(target/'eeg_filtered_offline.csv',index=False)
    pd.DataFrame(blocks,columns=['start_s','end_s','reason']).to_csv(target/'screening_exclusions.csv',index=False)
    before=x.copy();before[flags]=np.nan
    report=dict(source=str(source),source_sha256=sha,rows=len(x),fs_hz=512,duration_s=len(x)/512,
                raw_min=float(x.min()),raw_max=float(x.max()),outside_expected_asic_range_samples=int(outside.sum()),
                expected_rail_samples=int(((x==-2048)|(x==2047)).sum()),
                screening_excluded_s=int(flags.sum())/512,filter_edge_review_s=int(edge_guard.sum())/512,
                segments_filtered=segments,exclusions=blocks,
                filter=dict(method='offline zero-phase SOS forward/backward',notch_hz=50,notch_Q=30,
                            bandpass_hz=[.5,45],butterworth_design_order=4),
                median_line_fraction_before=line_fraction(before),median_line_fraction_after=line_fraction(filtered),
                labels_created=False,quality_usable_confirmed=False,
                scope='Diagnostic spectral EEG only. NOT blink validation, training features, or a real-time filter.',
                warning='Do not reconstruct excluded samples. No wall-clock/KSS alignment inferred. Absence of screening flags does not prove clean EEG.',
                created_at_utc=datetime.now(timezone.utc).isoformat())
    (target/'processing.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha,'Source changed'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('target',type=Path)
    a=p.parse_args();run(a.source,a.target)
