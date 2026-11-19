"""State machine for starting and clearing a drowsiness alarm."""

from collections import deque


class AlarmPolicy:
    def __init__(self, threshold=0.5, consecutive=3, release_consecutive=1,
                 release_threshold=None, confirmation_window=None,
                 support_threshold=None, support_consecutive=1):
        release_threshold = threshold if release_threshold is None else release_threshold
        confirmation_window = consecutive if confirmation_window is None else confirmation_window
        support_threshold = threshold if support_threshold is None else support_threshold
        if (not 0 <= release_threshold <= threshold <= 1 or consecutive < 1
                or release_consecutive < 1 or confirmation_window < consecutive
                or not 0 <= support_threshold <= threshold or support_consecutive < 1):
            raise ValueError('Invalid alarm thresholds or confirmation counts')
        self.threshold = threshold
        self.consecutive = consecutive
        self.release_threshold = release_threshold
        self.release_consecutive = release_consecutive
        self.support_threshold = support_threshold
        self.support_consecutive = support_consecutive
        self.recent_high = deque(maxlen=confirmation_window)
        self.reset()

    def reset(self):
        self.recent_high.clear()
        self.active = False
        self.low_streak = 0
        self.support_streak = 0

    def update(self, risk):
        if risk is None:
            self.reset()
        elif self.active:
            self.low_streak = self.low_streak + 1 if risk < self.release_threshold else 0
            if self.low_streak >= self.release_consecutive:
                self.reset()
        else:
            self.recent_high.append(risk >= self.threshold)
            self.support_streak = self.support_streak + 1 if risk >= self.support_threshold else 0
            if (risk >= self.threshold and sum(self.recent_high) >= self.consecutive
                    and self.support_streak >= self.support_consecutive):
                self.active = True
        return self.active
