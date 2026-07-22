"""
utils/timing.py

    timer = StageTimer()
    with timer.stage("clip_embed"):
        ...
    result["stage_timings_ms"] = timer.as_dict()
"""

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self._timings: dict = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = round((time.perf_counter() - start) * 1000.0, 2)

    def as_dict(self) -> dict:
        self._timings["total"] = round(sum(self._timings.values()), 2)
        return dict(self._timings)
