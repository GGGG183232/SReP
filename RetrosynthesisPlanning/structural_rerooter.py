"""Leiden structure weights; each expanded state's anchor weight is fixed."""
import math
import time
from collections import Counter
from dataclasses import dataclass

from reroot_priority import logaddexp


@dataclass(frozen=True)
class RerootConfig:
    structure_weight: float = 1.0
    heuristic_weight: float = 1.0
    alpha: float = 10.0
    first_cluster_expansions: int = 32
    leiden_resolution: float = 1.0
    leiden_iterations: int = 2

    def validate(self):
        if self.structure_weight != 1. or self.heuristic_weight != 1.:
            raise ValueError('Full SReP uses both starting-point terms with unit coefficients')
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('Alpha must be finite and positive')
        if self.first_cluster_expansions < 1 or self.leiden_resolution <= 0:
            raise ValueError('Invalid graph-clustering configuration')
        if self.leiden_iterations < 1:
            raise ValueError('Leiden iterations must be positive')


def leiden_membership(node_count, edges, seed, resolution, iterations):
    import igraph
    import leidenalg
    graph = igraph.Graph(n=node_count, edges=sorted(edges), directed=False)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, n_iterations=iterations,
        seed=int(seed) % (2 ** 31 - 1))
    return list(partition.membership)


class StructuralRerooter:
    def __init__(self, config, root_h, seed, partitioner=leiden_membership):
        config.validate()
        self.config = config
        self.root_h = max(float(root_h), 1e-12)
        self.seed = seed
        self.partitioner = partitioner
        self.cluster_sizes = Counter({0: 1})
        self.delta = Counter()
        self.next_update = config.first_cluster_expansions
        self.updates = 0
        self.seconds = 0.0
        self.last_communities = 1
        self.max_communities = 1

    def new_state(self, record):
        # Before the first partition we know the actual one-color graph size.
        # Afterwards sizes are from the latest partition; new vertices inherit
        # their discovery parent's color until the next geometric refresh.
        if self.updates == 0:
            self.cluster_sizes[record.community] += 1

    def log_weight(self, record):
        if record.id == 0:
            return 0.0
        cfg = self.config
        size = max(1, self.cluster_sizes[record.community] + self.delta[record.community])
        structural = -math.log(size)
        heuristic = -cfg.alpha * record.h / self.root_h
        return logaddexp(structural, heuristic)

    def on_expansion(self, record):
        self.delta[record.community] += 1

    def maybe_update(self, graph, expansions):
        if expansions < self.next_update:
            return None
        started = time.monotonic()
        colors = self.partitioner(len(graph.records), graph.undirected_edges,
                                  self.seed + self.updates,
                                  self.config.leiden_resolution,
                                  self.config.leiden_iterations)
        if len(colors) != len(graph.records):
            raise ValueError('Partitioner returned an invalid membership array')
        for record, color in zip(graph.records, colors):
            record.community = int(color)
        self.cluster_sizes = Counter(colors)
        self.delta.clear()
        self.last_communities = len(self.cluster_sizes)
        self.max_communities = max(self.max_communities, self.last_communities)
        self.updates += 1
        self.seconds += time.monotonic() - started
        while self.next_update <= expansions:
            self.next_update *= 2
        return {'event': 'leiden_update', 'state_expansions': expansions,
                'graph_nodes': len(graph.records),
                'graph_edges': len(graph.undirected_edges),
                'communities': self.last_communities,
                'next_update': self.next_update}
