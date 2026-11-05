"""Additive route-cost rerooting, with A and B stored in log space.

A=C_a/w_a, B=1/w_a, where C_a sums the baseline's clipped negative-log
edge costs from the anchor. Across an edge with cost c: A'=A+B*c, B'=B.
This replaces inverse-prefix-probability search costs, not the fixed
rerooter weights, mixture schedule, or state/label dominance rules.
"""
import math
from dataclasses import dataclass


REROOT_COST_MODE = 'negative_log_sum'


def logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


@dataclass(frozen=True)
class Label:
    log_a: float
    log_b: float
    anchor: int

    @classmethod
    def start(cls, anchor, log_weight):
        return cls(-math.inf, -log_weight, anchor)

    def advance(self, edge_cost):
        if not math.isfinite(edge_cost) or edge_cost < 0:
            raise ValueError('Edge cost must be finite and >= 0')
        if edge_cost == 0:
            # p=1 produces -log(p)=0 in the unchanged route-cost definition.
            # Do not add an epsilon/step penalty or evaluate log(0).
            return self
        return Label(logaddexp(self.log_a, self.log_b + math.log(edge_cost)),
                     self.log_b, self.anchor)


class Envelope:
    """Keep all nondominated (A, B) labels, not only today's smallest A."""
    def __init__(self):
        self.labels = []
        self.accepted = 0
        self.rejected = 0
        self.removed = 0

    def add(self, label):
        # No epsilon: an approximate dominance rule could destroy a useful
        # future label. Exact equality is a duplicate and is safely rejected.
        if any(x.log_a <= label.log_a and x.log_b <= label.log_b
               for x in self.labels):
            self.rejected += 1
            return False
        kept = [x for x in self.labels
                if not (label.log_a <= x.log_a and label.log_b <= x.log_b)]
        self.removed += len(self.labels) - len(kept)
        kept.append(label)
        self.labels = kept
        self.accepted += 1
        return True

    def best(self):
        return min(self.labels, key=lambda x: (x.log_a, x.log_b, x.anchor))

    def contains(self, label):
        return label in self.labels
