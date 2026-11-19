import numpy as np
import pytest

from n16_eeg.blink_eval import match_event_pairs, match_events


def test_assignment_does_not_lose_valid_blinks_to_nearest_first_match():
    # Greedy nearest matching consumes 0.1 for the first prediction and loses
    # the second. Both references can be matched in chronological order.
    pairs = match_event_pairs([0.06, 0.16], [0.0, 0.1], tolerance_s=0.07)
    assert pairs == [(0.06, 0.0), (0.16, 0.1)]
    assert match_events([0.06, 0.16], [0.0, 0.1], .07)["f1"] == 1


def test_assignment_prefers_smaller_error_among_equal_match_counts():
    assert match_event_pairs([1.0], [.85, 1.02], .2) == [(1.0, 1.02)]
    assert match_events([], [1.0])["false_negative"] == 1


def test_assignment_rejects_nonfinite_event_times():
    with pytest.raises(ValueError):
        match_event_pairs([np.nan], [1.0])
