"""Hybrid min-f / additive-cost rerooting on a shared exact-state graph.

Chemistry, the stock, Top-50 policy, Vcost and clipped route costs are frozen.
The exploration lane directly minimizes reroot priority with a
non-resetting global-depth scheduling penalty. It does NOT sample candidates
and subsequently override that choice with original f.
The reroot base cost is a sum of Edge.cost, not inverse path probabilities;
state h is not added to the reroot priority (it still informs anchor weights).
"""
import math
import random
import time
from collections import Counter, deque
from dataclasses import asdict

from depth_penalty import DepthPenaltyConfig, log1p_r
from reroot_priority import Label, REROOT_COST_MODE
from state_graph import Edge, Proof, StateGraph, state_key
from structural_rerooter import RerootConfig, StructuralRerooter


GLOBAL_LOCAL_STAGES = (
    (0, 100, .65),
    (100, 200, .40),
    (200, 300, .20),
    (300, 500, .00),
)


class StructureSearch:
    def __init__(self, target, stock, value_fn, policy_fn, seed=20260920,
                 max_budget=500, reroot_config=None,
                 trace_fn=None, partitioner=None, depth_penalty_config=None):
        self.target = target
        self.stock = stock
        self.value_fn = value_fn
        self.policy_fn = policy_fn
        self.max_budget = int(max_budget)
        self.schedule = GLOBAL_LOCAL_STAGES
        self.rng = random.Random(seed)
        self.trace_fn = trace_fn
        self.depth_penalty = depth_penalty_config or DepthPenaltyConfig()
        self.depth_penalty.validate()
        self.graph = StateGraph(self.depth_penalty)
        self.values = {}
        self.policies = {}
        self.route_queue = deque()
        self.label_queue = deque()
        self.depth_queue = deque()
        self.model_calls = 0
        self.expansions = 0
        self.effective_expansions = 0
        self.generated = 0
        self.selection_counts = Counter({'retrostar': 0, 'reroot': 0})
        self.call_counts = Counter({'retrostar': 0, 'reroot': 0})
        self.counters = Counter()
        self.timings = Counter()
        self.max_labels = 0
        self.label_count = 0
        self.peak_total_labels = 0
        self.goal = None
        self.termination_reason = None
        self.max_generated_route_depth = 0
        self.max_selected_route_depth = 0
        self.max_selected_min_step_depth = 0
        self.depth_penalty_totals = Counter()
        self.selected_depth_histogram = Counter()
        self.real_call_depth_histogram = Counter()
        self.cache_depth_histogram = Counter()
        self.selected_route_depth_histogram = Counter()
        self.real_call_route_depth_histogram = Counter()
        key = state_key([target], stock)
        root, _ = self.graph.create(key, self._h(key), 0, 0)
        root.proof = Proof(key, 0.)
        root.g = 0.
        root.min_step_depth = 0
        options = {} if partitioner is None else {'partitioner': partitioner}
        self.rerooter = StructuralRerooter(reroot_config or RerootConfig(), root.h,
                                         seed, **options)
        root.anchor_created = True
        root.anchor_log_weight = 0.
        self._add_label(root, Label.start(root.id, 0.))
        self.graph.refresh(root)
        if not key:
            self.goal = root.proof

    def _h(self, key):
        unknown = [m for m in key if m not in self.values]
        if unknown:
            started = time.monotonic()
            values = list(self.value_fn(unknown))
            self.timings['value_seconds'] += time.monotonic() - started
            if len(values) != len(unknown):
                raise ValueError('Value model returned a mismatched batch')
            for molecule, value in zip(unknown, values):
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Vcost must be finite and nonnegative')
                self.values[molecule] = value
        return sum(self.values[m] for m in key)

    def _emit(self, event):
        if self.trace_fn:
            self.trace_fn(event)

    def _add_label(self, record, label):
        before = len(record.labels.labels)
        accepted = record.labels.add(label)
        self.counters['labels_accepted' if accepted else 'labels_dominated'] += 1
        if accepted:
            self.label_count += len(record.labels.labels) - before
            self.peak_total_labels = max(self.peak_total_labels, self.label_count)
            self.max_labels = max(self.max_labels, len(record.labels.labels))
            if record.expanded:
                self.label_queue.append((record.id, label))
            else:
                self.graph.refresh(record)
        return accepted

    def _relax_route(self, parent, edge):
        child = self.graph.records[edge.child]
        if parent.proof is None:
            return
        if parent.proof.contains(child.key):
            self.counters['cyclic_route_paths_rejected'] += 1
            return
        new_g = parent.g + edge.cost
        if new_g >= child.g:
            self.counters['route_costs_dominated'] += 1
            return
        child.g = new_g
        child.proof = Proof(child.key, new_g, parent.proof.depth + 1,
                            parent.proof, edge.reaction, edge.template,
                            edge.outcome_index)
        self.max_generated_route_depth = max(self.max_generated_route_depth,
                                             child.proof.depth)
        self.counters['route_costs_improved'] += 1
        if not child.key:
            self.goal = child.proof
        elif child.expanded:
            self.route_queue.append(child.id)
        else:
            self.graph.refresh(child)

    def _relax_depth(self, parent, edge):
        """Unit-edge distance is independent of the lowest-g proof and labels."""
        child = self.graph.records[edge.child]
        depth = parent.min_step_depth + 1
        if depth >= child.min_step_depth:
            return
        child.min_step_depth = depth
        self.counters['global_depth_improvements'] += 1
        if child.expanded:
            self.depth_queue.append(child.id)
        else:
            self.graph.refresh(child)

    def _propagate(self):
        started = time.monotonic()
        # Better arrival labels at a materialized state reuse its stored edges.
        # They never reopen the state or call the chemical model again.
        while self.route_queue:
            parent = self.graph.records[self.route_queue.popleft()]
            for edge in parent.edges:
                self._relax_route(parent, edge)
                if self.goal is not None:
                    self.timings['priority_seconds'] += time.monotonic() - started
                    return
        # A shorter but more costly arrival can improve scheduling depth even
        # when lowest-g route dominance rejects it. Propagate through CLOSED
        # states without reopening them or invoking the single-step model.
        while self.depth_queue:
            parent = self.graph.records[self.depth_queue.popleft()]
            for edge in parent.edges:
                self._relax_depth(parent, edge)
        while self.label_queue:
            state_id, label = self.label_queue.popleft()
            record = self.graph.records[state_id]
            if not record.labels.contains(label):
                continue
            for edge in record.edges:
                self.counters['label_edge_relaxations'] += 1
                self._add_label(self.graph.records[edge.child],
                                label.advance(edge.cost))
        self.timings['priority_seconds'] += time.monotonic() - started
        self.graph.compact_heaps()

    def probability(self):
        for lower, upper, probability in self.schedule:
            if lower <= self.model_calls < upper:
                return probability
        return 0.

    def _select_lane(self):
        probability = self.probability()
        if probability == 0.:
            return 'reroot'
        return 'retrostar' if self.rng.random() < probability else 'reroot'

    def _policy(self, molecule, lane):
        if molecule in self.policies:
            self.counters['policy_cache_hits'] += 1
            return self.policies[molecule], False
        # An attempted real call consumes budget even if it raises an error.
        self.model_calls += 1
        self.call_counts[lane] += 1
        started = time.monotonic()
        try:
            policy = self.policy_fn(molecule)
        finally:
            self.timings['policy_seconds'] += time.monotonic() - started
        self.policies[molecule] = policy
        return policy, True

    def _materialize(self, record, lane):
        molecule = record.key[0]  # Same deterministic order as the baseline.
        policy, real_call = self._policy(molecule, lane)
        if policy is None or not policy.get('scores'):
            self.counters['empty_policy_states'] += 1
            return real_call
        scores, reactants = policy['scores'], policy['reactants']
        templates = policy['template']
        if len(scores) != len(reactants) or len(scores) != len(templates):
            raise ValueError('Invalid policy output lengths')
        self.generated += len(scores)
        grouped = {}
        for i, (score, reactant, template) in enumerate(zip(scores, reactants, templates)):
            score = float(score)
            if not math.isfinite(score) or score < 0 or score > 1.000001:
                raise ValueError('Invalid policy probability')
            key = state_key(reactant.split('.') + list(record.key[1:]), self.stock)
            cost = -math.log(min(1., max(1e-3, score)))
            option = (cost, reactant + '>>' + molecule, template, i, score)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(option)
        if sum(float(s) for s in scores) > 1.000001:
            raise ValueError('Policy probabilities sum to more than one')
        self.effective_expansions += 1
        started = time.monotonic()
        for key, options in grouped.items():
            child, fresh = self.graph.create(
                key, self._h(key) if key not in self.graph.by_key else 0.,
                self.model_calls, record.community)
            if fresh:
                self.rerooter.new_state(child)
            else:
                self.counters['duplicate_state_arrivals'] += 1
            self.graph.add_connection(record.id, child.id)
            if len(options) > 1:
                self.counters['outcomes_merged_by_complete_state'] += len(options) - 1
            probability = min(1., math.fsum(x[4] for x in options))
            if probability == 0:
                # Numerical support only; do not use the route-cost 1e-3 floor
                # for search probabilities or renormalize after pruning.
                self.counters['zero_probability_edges'] += 1
            log_p = math.log(max(probability, 1e-300))
            best = min(options, key=lambda x: (x[0], x[3]))
            edge = Edge(child.id, log_p, *best[:4], tuple(options))
            if edge.cost == 0:
                self.counters['zero_cost_edges'] += 1
            record.edges.append(edge)
            self._relax_depth(record, edge)
            self._relax_route(record, edge)
            if self.goal is not None:
                # Match the baseline's first-complete-route termination: do
                # not evaluate unrelated later outcomes after a proven goal.
                self.timings['materialization_seconds'] += time.monotonic() - started
                return real_call
        self.timings['materialization_seconds'] += time.monotonic() - started
        for label in list(record.labels.labels):
            self.label_queue.append((record.id, label))
        return real_call

    def search(self):
        if self.goal is not None:
            self.termination_reason = 'success'
            return True
        while self.model_calls < self.max_budget:
            self._propagate()
            if self.goal is not None:
                break
            baseline = self.graph.peek_f()
            if baseline is None:
                self.termination_reason = 'open_exhausted'
                return False
            lane = self._select_lane()
            raw_reroot = self.graph.peek_unpenalized_reroot() if lane == 'reroot' else None
            record = self.graph.pop(lane)
            if record is None:
                raise RuntimeError('Reachable OPEN state is missing a reroot label')
            self.selection_counts[lane] += 1
            selected_label = record.labels.best()
            selected_depth = int(record.min_step_depth)
            penalty = self.depth_penalty.penalty(selected_depth) if lane == 'reroot' else 0.
            adjusted_score = (self.depth_penalty.score(selected_label.log_a, selected_depth)
                              if lane == 'reroot' else record.f)
            changed_by_penalty = bool(self.depth_penalty.active and lane == 'reroot'
                                      and raw_reroot is not None and raw_reroot.id != record.id)
            self.selected_depth_histogram[selected_depth] += 1
            self.selected_route_depth_histogram[record.proof.depth] += 1
            self.max_selected_min_step_depth = max(self.max_selected_min_step_depth, selected_depth)
            self.max_selected_route_depth = max(self.max_selected_route_depth, record.proof.depth)
            if lane == 'reroot':
                self.depth_penalty_totals['reroot_selections'] += 1
                self.depth_penalty_totals['penalized_selections'] += int(penalty > 0)
                self.depth_penalty_totals['strong_penalty_selections'] += int(
                    penalty > 0 and selected_depth > self.depth_penalty.strong_depth)
                self.depth_penalty_totals['selected_penalty_sum'] += penalty
                self.depth_penalty_totals['changed_selections'] += int(changed_by_penalty)
            if lane == 'reroot' and selected_label.anchor != 0:
                self.counters['local_anchor_selections'] += 1
            if record.id != baseline.id:
                self.counters['selections_different_from_min_f'] += 1
            record.selected_call = self.model_calls
            self.expansions += 1
            # Eq. (5) includes the current expansion in delta.
            self.rerooter.on_expansion(record)
            # Selection uses ancestor labels; only after it is selected does
            # this state create its own local anchor. It cannot select itself.
            if not record.anchor_created:
                log_w = self.rerooter.log_weight(record)
                record.anchor_created = True
                record.anchor_log_weight = log_w
                if log_w != -math.inf:
                    self._add_label(record, Label.start(record.id, log_w))
            before = self.model_calls
            event = {'event': 'selection', 'state_id': record.id,
                     'state': list(record.key), 'lane': lane, 'g': record.g,
                     'h': record.h, 'f': record.f,
                     'route_depth': record.proof.depth,
                     'min_step_depth': selected_depth,
                     'depth_penalty': penalty,
                     'depth_penalty_enabled': self.depth_penalty.active,
                     'penalized_priority': adjusted_score,
                     'reroot_log1p_priority': log1p_r(selected_label.log_a),
                     'changed_by_depth_penalty': changed_by_penalty,
                     'counterfactual_unpenalized_reroot_state': (raw_reroot.id if raw_reroot else None),
                     'counterfactual_unpenalized_reroot_depth': (raw_reroot.min_step_depth if raw_reroot else None),
                     'log_r': selected_label.log_a,
                     'reroot_cost_mode': REROOT_COST_MODE,
                     'reroot_priority_is_zero': selected_label.log_a == -math.inf,
                     'selected_anchor': selected_label.anchor,
                     'selected_anchor_log_weight': self.graph.records[selected_label.anchor].anchor_log_weight,
                     'new_anchor_log_weight': record.anchor_log_weight,
                     'community': record.community,
                     'born_call': record.born_call, 'selected_call': before,
                     'wait_calls': before - record.born_call,
                     'counterfactual_min_f_state': baseline.id,
                     'open_size': self.graph.open_size,
                     'real_model_call': record.key[0] not in self.policies}
            # Emit before the call too, preserving the last attempted selection
            # in a traceback/timeout. Completion is a separate event.
            self._emit(event)
            real_call = self._materialize(record, lane)
            histogram = self.real_call_depth_histogram if real_call else self.cache_depth_histogram
            histogram[selected_depth] += 1
            if real_call:
                self.real_call_route_depth_histogram[event['route_depth']] += 1
            if changed_by_penalty:
                kind = 'changed_real_model_calls' if real_call else 'changed_cached_expansions'
                self.depth_penalty_totals[kind] += 1
            self._propagate()
            self.counters['materialized_states'] += 1
            self._emit({'event': 'expansion_complete', 'state_id': record.id,
                        'model_calls': self.model_calls, 'real_model_call': real_call,
                        'outgoing_states': len(record.edges),
                        'success': self.goal is not None})
            if self.goal is not None:
                break
            # Geometric refresh runs after this expansion has generated its
            # observed edges; existing anchors retain their original weights.
            cluster_event = self.rerooter.maybe_update(self.graph, self.expansions)
            if cluster_event:
                self._emit(cluster_event)
        self.termination_reason = ('success' if self.goal is not None
                                   else 'budget_exhausted')
        return self.goal is not None

    def stats(self):
        return {'reroot_cost_mode': REROOT_COST_MODE,
                'depth_penalty_config': asdict(self.depth_penalty),
                'depth_penalty_totals': dict(self.depth_penalty_totals),
                'max_generated_route_depth': self.max_generated_route_depth,
                'max_selected_route_depth': self.max_selected_route_depth,
                'max_selected_min_step_depth': self.max_selected_min_step_depth,
                'selected_min_step_depth_histogram': dict(self.selected_depth_histogram),
                'real_call_min_step_depth_histogram': dict(self.real_call_depth_histogram),
                'cache_min_step_depth_histogram': dict(self.cache_depth_histogram),
                'selected_route_depth_histogram': dict(self.selected_route_depth_histogram),
                'real_call_route_depth_histogram': dict(self.real_call_route_depth_histogram),
                'model_calls': self.model_calls, 'state_expansions': self.expansions,
                'effective_expansions': self.effective_expansions,
                'generated_outcomes': self.generated,
                'graph_nodes': len(self.graph.records),
                'graph_edges': len(self.graph.undirected_edges),
                'graph_self_loops': self.graph.graph_self_loops,
                'open_size': self.graph.open_size, 'max_open': self.graph.max_open,
                'max_labels_per_state': self.max_labels,
                'total_labels': self.label_count,
                'peak_total_labels': self.peak_total_labels,
                'heap_stale_entries': self.graph.stale_heap_entries,
                'selection_counts': dict(self.selection_counts),
                'real_call_counts': dict(self.call_counts),
                'counters': dict(self.counters), 'timings': dict(self.timings),
                'clustering_updates': self.rerooter.updates,
                'clustering_seconds': self.rerooter.seconds,
                'communities': self.rerooter.last_communities,
                'max_communities': self.rerooter.max_communities,
                'termination_reason': self.termination_reason}
