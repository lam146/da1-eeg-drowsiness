"""Watch for stalled delivery of decoded raw EEG using a monotonic clock."""

RAW_GAP_S = 0.5
SERIAL_POLL_S = 0.05


class RawStreamWatchdog:
    def __init__(self, started: float) -> None:
        self.last_raw = started
        self.stalled = False

    def check(self, now: float, has_raw: bool) -> bool:
        """Report a new stall, including one that ends in this batch."""
        new_stall = not self.stalled and now - self.last_raw >= RAW_GAP_S
        if has_raw:
            self.last_raw = now
            self.stalled = False
        elif new_stall:
            self.stalled = True
        return new_stall
