import pytest

from n16_eeg.alarm_eval import alarm_onset_metrics


def test_alarm_already_active_at_onset_has_no_valid_detection_delay():
    result = alarm_onset_metrics([0, 250, 500, 750, 1000], [False, True, True, True, True], 500)
    assert result['pre_onset_alarm']
    assert result['first_active_after_onset_ms'] == 0
    assert result['onset_to_alarm_ms'] is None
    assert not result['valid_under_3s']


def test_new_alarm_delay_and_three_second_limit():
    result = alarm_onset_metrics([0, 1000, 3499, 4000], [False, False, True, True], 500)
    assert result['onset_to_alarm_ms'] == 2999
    assert result['valid_under_3s']
    result = alarm_onset_metrics([0, 1000, 3500, 4000], [False, False, True, True], 500)
    assert not result['valid_under_3s']


def test_early_alarm_then_later_retrigger_remains_reported_as_premature():
    result = alarm_onset_metrics([0, 1000, 2000, 3000], [True, False, False, True], 2000)
    assert result['first_new_alarm_after_onset_ms'] == 1000
    assert result['onset_to_alarm_ms'] is None


def test_missing_alarm_and_bad_time_grid():
    assert alarm_onset_metrics([], [], 10)['onset_to_alarm_ms'] is None
    with pytest.raises(ValueError):
        alarm_onset_metrics([1, 1], [False, True], 1)
