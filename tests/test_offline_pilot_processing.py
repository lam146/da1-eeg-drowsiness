"""Offline processing must not repair, relabel, or overwrite original observations."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    pytest.importorskip('scipy.signal')
    spec=importlib.util.spec_from_file_location('filter_pilot_offline',Path(__file__).parents[1]/'scripts/filter_pilot_offline.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_filter_suppresses_line_preserves_alpha_and_does_not_fill_artifact(tmp_path):
    mod=module()
    t=np.arange(24*512)/512
    clean=70*np.sin(2*np.pi*10*t)+300*np.sin(2*np.pi*50*t)
    raw=clean.copy();raw[529]=-21846
    source=tmp_path/'raw.csv';out=tmp_path/'derived'
    pd.DataFrame({'Time:512Hz':t,'Electrode':raw}).to_csv(source,index=False)
    original=source.read_bytes()
    report=mod.run(source,out)
    data=pd.read_csv(out/'eeg_filtered_offline.csv')
    assert source.read_bytes()==original
    assert len(data)==len(t) and np.allclose(data.time_s,t)
    assert data.loc[data.time_s<4,'eeg_filtered_count'].isna().all()
    keep=(data.time_s>=8)&(data.time_s<20)
    filtered=data.loc[keep,'eeg_filtered_count'].to_numpy()
    tt=data.loc[keep,'time_s'].to_numpy()
    amp=lambda freq:2*abs(np.mean(filtered*np.exp(-2j*np.pi*freq*tt)))
    assert 60<amp(10)<80
    assert amp(50)<3
    assert report['screening_excluded_s']==4
    assert report['labels_created'] is False
    assert 'label' not in data
    assert 'not_flagged_by_screening' in set(data.quality_screen)
    with pytest.raises(FileExistsError):mod.run(source,out)


def test_discontinuous_record_is_not_interpolated(tmp_path):
    mod=module()
    t=np.arange(4096)/512;t[2048:]+=1
    source=tmp_path/'broken.csv';out=tmp_path/'derived'
    pd.DataFrame({'Time:512Hz':t,'Electrode':np.sin(t)}).to_csv(source,index=False)
    with pytest.raises(ValueError,match='continuous'):mod.run(source,out)
    assert not out.exists()
