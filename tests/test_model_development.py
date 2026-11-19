"""Check train-only weighting and transforms used in provisional development."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from develop_pilot_model import session_weights, estimator


def test_repeating_a_session_does_not_increase_its_total_weight():
    f=pd.DataFrame({'subject_id':['S01']*7,'label':[0,0,0,1,1,1,1],
                    'session_id':['A','A','B','C','D','D','D']})
    original=f.copy(deep=True)
    f['weight']=session_weights(f)
    totals=f.groupby(['label','session_id']).weight.sum()
    assert np.allclose(totals.to_numpy(),len(f)/4)
    assert np.allclose(f.groupby('label').weight.sum().to_numpy(),len(f)/2)
    pd.testing.assert_frame_equal(f.drop(columns='weight'),original)


def test_robust_scaler_is_not_refit_by_extreme_test_values():
    X=np.array([[0.,1.],[1.,2.],[5.,7.],[8.,9.]])
    y=np.array([0,0,1,1])
    model=estimator(False);model.fit(X,y)
    expected=np.median(np.log1p(X),axis=0)
    model.predict_proba([[1e12,1e12]])
    assert np.allclose(model.named_steps['scale'].center_,expected)
