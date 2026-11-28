"""Measure a new alarm episode after a known reference onset."""

import numpy as np


def alarm_onset_metrics(timestamps_ms, alarms, onset_ms):
    times = np.asarray(timestamps_ms, dtype=float)
    active = np.asarray(alarms, dtype=bool)
    if (times.ndim != 1 or active.shape != times.shape or not np.isfinite(times).all()
            or not np.isfinite(onset_ms) or np.any(np.diff(times) <= 0)):
        raise ValueError('Expected matching alarm states and increasing finite timestamps')
    early = bool(active[times < onset_ms].any())
    rising = active & ~np.r_[False, active[:-1]] if len(active) else active
    episodes = times[rising & (times >= onset_ms)] - onset_ms
    after = times[active & (times >= onset_ms)] - onset_ms
    new_alarm = float(episodes[0]) if len(episodes) else None
    delay = new_alarm if not early else None
    return dict(pre_onset_alarm=early, onset_to_alarm_ms=delay,
                first_new_alarm_after_onset_ms=new_alarm,
                first_active_after_onset_ms=float(after[0]) if len(after) else None,
                valid_under_3s=bool(delay is not None and delay < 3000))
