"""Soft global-depth scheduling cost, independent of chemistry and reroot labels.

Depth is the shortest unit-edge distance in the currently observed state graph,
not distance from a local anchor, model-call count, or a route optimality claim.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthPenaltyConfig:
    enabled: bool = True
    start_depth: int = 20
    strong_depth: int = 40
    soft_slope: float = .05
    strong_slope: float = 1.

    def validate(self):
        if not self.enabled:
            raise ValueError('Full SReP requires depth regularization')
        if (not isinstance(self.start_depth, int)
                or not isinstance(self.strong_depth, int)
                or not 0 <= self.start_depth <= self.strong_depth):
            raise ValueError('Depth thresholds must be ordered nonnegative integers')
        if any(not math.isfinite(x) or x <= 0
               for x in (self.soft_slope, self.strong_slope)):
            raise ValueError('Depth penalty slopes must be finite and positive')

    @property
    def active(self):
        return True

    def penalty(self, depth):
        if not math.isfinite(depth) or depth < 0:
            raise ValueError('A selectable state must have a finite global depth')
        soft_steps = min(max(0, depth - self.start_depth),
                         self.strong_depth - self.start_depth)
        strong_steps = max(0, depth - self.strong_depth)
        return self.soft_slope * soft_steps + self.strong_slope * strong_steps

    def score(self, log_r, depth):
        return log1p_r(log_r) + self.penalty(depth)


def log1p_r(log_r):
    """Compute log(1 + R) from log(R), including R=0 and huge R safely."""
    if math.isnan(log_r):
        raise ValueError('Reroot priority cannot be NaN')
    if log_r == -math.inf:
        return 0.
    return max(0., log_r) + math.log1p(math.exp(-abs(log_r)))
