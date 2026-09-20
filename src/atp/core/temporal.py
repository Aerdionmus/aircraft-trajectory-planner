"""Pure temporal utilities for the discrete planner state.

This module contains only validation and conversion logic for integer time-bucket
semantics.  It intentionally does not include traversal-time solves, wind
sampling, fixed-point iteration, or dynamic transition semantics.  Those belong
in a future M4.2.2 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_TIME_ORIGIN_H: float = 0.0


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Explicit temporal configuration for a discrete-time planner state.

    Dynamic mode must provide a valid positive ``time_step_h``.  Static M1/M2/M3
    planning does not require a temporal configuration at all.
    """

    time_origin_h: float = DEFAULT_TIME_ORIGIN_H
    time_step_h: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_origin_h):
            raise ValueError("time_origin_h must be finite")
        if not math.isfinite(self.time_step_h) or self.time_step_h <= 0.0:
            raise ValueError("time_step_h must be finite and > 0")

    def time_at_bucket(self, bucket: int) -> float:
        if type(bucket) is not int or bucket < 0:
            raise ValueError("bucket must be a non-negative int")
        return self.time_origin_h + float(bucket) * self.time_step_h

    def bucket_count_for_duration(self, duration_h: float) -> int:
        if not math.isfinite(duration_h):
            raise ValueError("duration_h must be finite")
        if duration_h <= 0.0:
            return 0
        return int(math.ceil(duration_h / self.time_step_h))

    def planned_duration_h(self, duration_h: float) -> float:
        return self.bucket_count_for_duration(duration_h) * self.time_step_h
