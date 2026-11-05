"""Exact unresolved-task states with separate route and search labels."""
import heapq
import math
from dataclasses import dataclass, field

from depth_penalty import DepthPenaltyConfig
from reroot_priority import Envelope


def state_key(molecules, stock):
    # Deliberately preserve the baseline's set semantics and SMILES strings.
    # This does not assert stoichiometric equivalence or molecular similarity.
    return tuple(sorted({m for m in molecules if m not in stock}))


@dataclass(frozen=True)
class Proof:
    state: tuple
    g: float
    depth: int = 0
    parent: object = None
    reaction: str = None
    template: str = None
    outcome_index: int = None

    def contains(self, key):
        node = self
        while node is not None:
            if node.state == key:
                return True
            node = node.parent
        return False

    def route(self):
        rows, node = [], self
        while node.parent is not None:
            rows.append({'reaction': node.reaction, 'template': node.template,
                         'outcome_index': node.outcome_index,
                         'state': list(node.state), 'g': node.g})
            node = node.parent
        rows.reverse()
        return rows


@dataclass(frozen=True)
class Edge:
    child: int
    log_probability: float
    cost: float
    reaction: str
    template: str
    outcome_index: int
    # Individual alternatives stay available as route evidence; probabilities
    # are summed only when their COMPLETE remaining tasks are exactly equal.
    alternatives: tuple = ()


@dataclass
class StateRecord:
    id: int
    key: tuple
    h: float
    born_call: int
    community: int
    g: float = math.inf
    proof: object = None
    labels: Envelope = field(default_factory=Envelope)
    edges: list = field(default_factory=list)
    expanded: bool = False
    in_open: bool = False
    version: int = 0
    anchor_created: bool = False
    anchor_log_weight: object = None
    selected_call: object = None
    min_step_depth: float = math.inf

    @property
    def f(self):
        return self.g + self.h


class StateGraph:
    def __init__(self, depth_penalty_config=None):
        self.depth_penalty = depth_penalty_config or DepthPenaltyConfig()
        self.depth_penalty.validate()
        self.records = []
        self.by_key = {}
        self.undirected_edges = set()
        self.f_heap = []
        self.r_heap = []
        # Records the unpenalized ordering for search diagnostics.
        self.raw_r_heap = []
        self.open_size = 0
        self.max_open = 0
        self.stale_heap_entries = 0
        self.graph_self_loops = 0

    def create(self, key, h, calls, community):
        if key in self.by_key:
            return self.records[self.by_key[key]], False
        record = StateRecord(len(self.records), key, h, calls, community)
        self.by_key[key] = record.id
        self.records.append(record)
        return record, True

    def add_connection(self, source, target):
        if source == target:
            self.graph_self_loops += 1
        else:
            self.undirected_edges.add((min(source, target), max(source, target)))

    def refresh(self, record):
        if record.expanded or not record.key or record.proof is None:
            return
        if not record.in_open:
            record.in_open = True
            self.open_size += 1
            self.max_open = max(self.max_open, self.open_size)
        record.version += 1
        heapq.heappush(self.f_heap, (record.f, record.id, record.version))
        if record.labels.labels:
            heapq.heappush(self.r_heap,
                           (self.reroot_key(record), record.id, record.version))
            heapq.heappush(self.raw_r_heap,
                           (record.labels.best().log_a, record.id, record.version))

    def reroot_key(self, record):
        log_r = record.labels.best().log_a
        # Original log(R) breaks numeric ties after the softplus transform.
        return (self.depth_penalty.score(log_r, record.min_step_depth), log_r)

    def _peek(self, heap):
        while heap:
            _, state_id, version = heap[0]
            record = self.records[state_id]
            if record.in_open and record.version == version:
                return record
            heapq.heappop(heap)
            self.stale_heap_entries += 1
        return None

    def peek_f(self):
        return self._peek(self.f_heap)

    def peek_unpenalized_reroot(self):
        return self._peek(self.raw_r_heap)

    def pop(self, lane):
        heap = self.f_heap if lane == 'retrostar' else self.r_heap
        record = self._peek(heap)
        if record is None:
            return None
        heapq.heappop(heap)
        record.in_open = False
        self.open_size -= 1
        record.expanded = True
        record.version += 1
        return record

    def compact_heaps(self):
        # Bound lazy-entry memory without restricting OPEN or label sets.
        if max(len(self.f_heap), len(self.r_heap), len(self.raw_r_heap)) <= 4 * max(1, self.open_size) + 1024:
            return
        live = [r for r in self.records if r.in_open]
        self.f_heap = [(r.f, r.id, r.version) for r in live]
        self.r_heap = [(self.reroot_key(r), r.id, r.version)
                       for r in live if r.labels.labels]
        self.raw_r_heap = [(r.labels.best().log_a, r.id, r.version)
                           for r in live if r.labels.labels]
        heapq.heapify(self.f_heap)
        heapq.heapify(self.r_heap)
        heapq.heapify(self.raw_r_heap)
