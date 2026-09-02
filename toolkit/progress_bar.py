from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        # `smoothing` only tunes tqdm's internal EMA; the rate actually shown
        # is overridden in `format_dict` below to a stable run average.
        kwargs.setdefault('smoothing', 0.7)
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()

    @property
    def format_dict(self):
        # Replace tqdm's jittery exponential-moving-average rate with a stable
        # run average (completed steps / elapsed wall time). `format_meter`
        # uses this single value for BOTH the displayed s/it and the derived
        # time-remaining estimate, so one override makes the rate human-
        # readable and the ETA accurate (no separate "avg" label needed).
        d = super().format_dict
        done = d.get('n', 0) - d.get('initial', 0)
        elapsed = d.get('elapsed', 0)
        if done > 0 and elapsed:
            d['rate'] = done / elapsed  # iterations/second (run average)
        return d

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
