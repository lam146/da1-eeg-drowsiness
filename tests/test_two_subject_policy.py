"""Guard the provisional pilot policy against invented tail labels and KSS transitions."""
import sys
from pathlib import Path

import pytest

pytest.importorskip('scipy')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_two_subjects_pilot import intervals


def points(*values):
    return {minute:{'kss':score} for minute,score in values}


def test_missing_end_of_record_does_not_extend_or_compress_labels():
    rows=intervals(points((0,7),(5,8),(10,8),(15,8)),780.0625)
    assert [(r['start_s'],r['end_s']) for r in rows]==[(0,300),(300,600)]
    assert all(r['time_alignment_verified'] is False for r in rows)
    assert all(r['label']==1 for r in rows)


def test_ambiguous_score_and_state_transition_are_not_binary_training_intervals():
    assert intervals(points((0,6),(5,7)),600)==[]
    assert intervals(points((0,5),(5,7)),600)==[]
    assert intervals(points((0,7),(10,7)),900)==[]


def test_consistent_alert_endpoints_remain_alert_without_filename_inference():
    rows=intervals(points((0,3),(5,4),(10,5)),700)
    assert len(rows)==2
    assert all(r['label']==0 for r in rows)
    assert rows[-1]['end_s']==600
