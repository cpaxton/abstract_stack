"""Measure progress independently of the contact skill's recovery policy."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContactProgress:
    """Track measured displacement along a world axis after contact begins.

    Brief loss of force does not reset the clock. A caller can explicitly reset
    the window when changing contact mode or starting a bounded recovery.
    """

    direction: tuple[float, float, float]
    minimum_advance_m: float = 0.001
    _origin_m: float | None = field(default=None, init=False)
    _since_s: float | None = field(default=None, init=False)

    def __post_init__(self):
        if len(self.direction) != 3 or not np.isclose(np.linalg.norm(self.direction), 1):
            raise ValueError("Progress direction must be a unit world vector")
        if not np.isfinite(self.minimum_advance_m) or self.minimum_advance_m <= 0:
            raise ValueError("Minimum progress must be finite and positive")

    def reset(self):
        self._origin_m = self._since_s = None

    def observe(self, position, time_s, *, contact):
        """Return elapsed seconds and signed advance in the current window."""
        coordinate = float(np.dot(self.direction, position))
        if self._since_s is None:
            if not contact:
                return None
            self._origin_m, self._since_s = coordinate, time_s
        advance = coordinate - self._origin_m
        if advance >= self.minimum_advance_m:
            self._origin_m, self._since_s = coordinate, time_s
            advance = 0.0
        return time_s - self._since_s, advance
