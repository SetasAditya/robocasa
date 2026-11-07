"""Temporal scoring independent of predicate polling frequency; JSON replay state."""
from copy import deepcopy
import math


class ScoreState:
    def __init__(self, stable_seconds=1.):
        self.stable_seconds = stable_seconds
        self.reset()

    def reset(self):
        self.last_time = None
        self.valid_since = None
        self.failed = False
        self.covered = set()
        self.articulated = False
        self.success = False

    def advance(self, time, *, goal, protected=True, penetration=False, cells=(), articulated=False):
        if not math.isfinite(time):
            raise ValueError('Nonfinite simulation time')
        if self.last_time is not None and time < self.last_time:
            raise ValueError('Time moved backwards without restoring scoring state')
        if self.last_time == time:
            return self.success
        self.last_time = time
        self.failed |= not protected or penetration
        self.covered.update(cells)
        self.articulated |= articulated
        if goal and not self.failed:
            if self.valid_since is None:
                self.valid_since = time
            self.success = time-self.valid_since >= self.stable_seconds-1e-9
        else:
            self.valid_since = None
            self.success = False
        return self.success

    def snapshot(self):
        d = deepcopy(self.__dict__)
        d['covered'] = sorted(self.covered)
        return d

    def restore(self, record):
        expected = set(self.snapshot())
        if set(record) != expected:
            raise ValueError('Incomplete scoring state')
        self.__dict__.update(deepcopy(record))
        self.covered = set(self.covered)
