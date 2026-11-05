"""Spawn-safe evaluation driver. Workers own models; datasets run serially."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / '.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.cache/matplotlib'))

from structure_search import GLOBAL_LOCAL_STAGES, StructureSearch
from depth_penalty import DepthPenaltyConfig
from reroot_priority import REROOT_COST_MODE
from structural_rerooter import RerootConfig, leiden_membership

DATASETS = ['USPTO', 'Ames', 'BBBP', 'ClinTox', 'Toxicity_LD50', 'logP', 'logS']
WORKER = {}


class TargetTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    if seconds <= 0:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    def timeout_handler(signum, frame):
        raise TargetTimeout('Per-target wall-time limit exceeded')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write_json(path, value):
    with Path(path).open('w') as handle:
        json.dump(safe_json(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def resolve_assets():
    """Resolve local assets; relative overrides are rooted at the repository."""
    local = PROJECT / 'RetrosynthesisPlanning'
    candidates = {
        'policy': ('KEEA_POLICY_PATH', local / 'saved_model/retro_star_value_ours.ckpt'),
        'templates': ('KEEA_TEMPLATES_PATH', local / 'saved_model/template_rules.dat'),
        'value': ('KEEA_VALUE_PATH', local / 'saved_model/best_epoch.pt'),
        'stock': ('KEEA_STOCK_PATH', local / 'prepare_data/origin_dict.csv'),
    }
    assets = {}
    for name, (variable, default) in candidates.items():
        override = os.environ.get(variable)
        chosen = Path(override).expanduser() if override else default
        if not chosen.is_absolute():
            chosen = PROJECT / chosen
        if not chosen.is_file():
            kind = 'building-block stock CSV' if name == 'stock' else name
            raise FileNotFoundError(
                f'{kind} not found: {chosen}. Place the file at '
                f'{default.relative_to(PROJECT)} or set {variable} '
                '(relative paths are resolved from the repository root).')
        assets[name] = str(chosen.resolve())
    return assets


def configure(args):
    dataset_names = [x.strip() for x in args.datasets.split(',') if x.strip()]
    if not dataset_names or len(set(dataset_names)) != len(dataset_names):
        raise ValueError('Dataset list must be nonempty and contain no duplicates')
    if any(name not in DATASETS for name in dataset_names):
        raise ValueError('Supported datasets: ' + ','.join(DATASETS))
    # Always finish USPTO before submitting any tasks from the other six.
    dataset_names.sort(key=DATASETS.index)
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or any(g < -1 for g in gpus) or (-1 in gpus and gpus != [-1]):
        raise ValueError('Use logical CUDA IDs, or -1 alone for CPU smoke tests')
    if args.workers < 1 or args.max_budget < 1 or args.timeout < 0 or args.cpu_threads < 1:
        raise ValueError('Invalid worker, budget, timeout or thread setting')
    if args.target_limit < 0 or args.trace_limit < 0:
        raise ValueError('Limits must be nonnegative')
    cfg = RerootConfig(alpha=args.alpha,
                       first_cluster_expansions=args.cluster_first,
                       leiden_resolution=args.leiden_resolution,
                       leiden_iterations=args.leiden_iterations)
    cfg.validate()
    depth_cfg = DepthPenaltyConfig(start_depth=args.depth_penalty_start,
                                   strong_depth=args.depth_penalty_strong_start,
                                   soft_slope=args.depth_penalty_soft_slope,
                                   strong_slope=args.depth_penalty_strong_slope)
    depth_cfg.validate()
    import torch
    if gpus != [-1]:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Full evaluation requires a working NVIDIA driver; '
                               'KEEA_GPUS=-1 is for CPU smoke tests only.')
        if max(gpus) >= torch.cuda.device_count():
            raise ValueError('KEEA_GPUS must use logical IDs AFTER CUDA_VISIBLE_DEVICES remapping')
    # No silent clustering fallback: verify that the actual Leiden backend runs.
    import igraph
    import leidenalg
    leiden_membership(2, {(0, 1)}, args.seed, cfg.leiden_resolution, cfg.leiden_iterations)
    assets = resolve_assets()
    targets = {}
    for dataset in dataset_names:
        path = PROJECT / 'RetrosynthesisPlanning/test_dataset' / (dataset + '.pkl')
        with path.open('rb') as handle:
            rows = pickle.load(handle)  # Local, user-specified trusted datasets.
        if not isinstance(rows, (list, tuple)) or not rows or not all(isinstance(s, str) for s in rows):
            raise ValueError(f'{dataset}: expected a sequence of target SMILES')
        targets[dataset] = list(rows[:args.target_limit] if args.target_limit else rows)
    asset_metadata = {key: {'path': path, 'size_bytes': Path(path).stat().st_size,
                            'mtime_ns': Path(path).stat().st_mtime_ns}
                      for key, path in assets.items()}
    hashes = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # With spawn, submitting fewer targets than workers may create fewer
    # processes. Cap tiny smoke runs to avoid waiting for nonexistent workers.
    workers = min(args.workers, len(targets[dataset_names[0]]))
    config = {'revision': 'shared_state_reroot_neglog_global_depth_v1', 'datasets': dataset_names,
              'reroot_cost_mode': REROOT_COST_MODE,
              'reroot_priority': 'log1p(min_anchor(local_cost / anchor_weight)) + global_depth_penalty',
              'depth_penalty_config': asdict(depth_cfg),
              'depth_penalty_scope': 'reroot_only; min-f lane and mixture unchanged',
              'depth_penalty_definition': 'soft_slope * min(max(d-start, 0), strong_start-start) + strong_slope * max(d-strong_start, 0)',
              'depth_measure': 'shortest unit-edge distance from root in observed directed state graph; independent of best-g proof',
              'reroot_add_state_h': False,
              'reroot_label_definition': {'A': 'local_cost / anchor_weight',
                                          'B': '1 / anchor_weight'},
              'dataset_sizes': {key: len(rows) for key, rows in targets.items()},
              'workers': workers, 'requested_workers': args.workers, 'logical_gpus': gpus,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'worker_gpu_assignment': [gpus[i % len(gpus)] for i in range(workers)],
              'max_budget': args.max_budget, 'timeout_seconds': args.timeout,
              'base_seed': args.seed, 'target_seed_rule': 'base_seed + global_target_id',
              'schedule': [(upper, probability) for _, upper, probability in GLOBAL_LOCAL_STAGES],
              'reroot_config': asdict(cfg), 'cpu_threads': args.cpu_threads,
              'assets': assets, 'asset_metadata': asset_metadata,
              'code_sha256': hashes, 'target_limit': args.target_limit,
              'trace_enabled': args.trace, 'trace_limit': args.trace_limit,
              'dependencies': {'torch': torch.__version__, 'igraph': igraph.__version__,
                               'leidenalg': importlib.metadata.version('leidenalg')},
              'stock_semantics': 'exact SMILES membership; full unresolved set',
              'topk_templates': 50,
              'route_edge_cost': '-log(clip(individual_outcome_score, 1e-3, 1))',
              'search_edge_probability': 'sum scores with identical full child state; retained as metadata, not used by additive labels',
              'budget_unit': 'attempted real single-step model calls; cache hits are free',
              'paper': 'https://proceedings.mlr.press/v306/tuero26a.html'}
    return config, targets


def initialize_worker(config, counter, load_gate, ready_queue, start_event):
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import RDLogger
    from policyNet import MLPModel
    from valueNet import ValueMLP
    from fingerprints import batch_smiles_to_fp
    RDLogger.DisableLog('rdApp.warning')
    torch.set_num_threads(config['cpu_threads'])
    with counter.get_lock():
        worker_id = counter.value
        counter.value += 1
    gpu = config['logical_gpus'][worker_id % len(config['logical_gpus'])]
    device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
    started = time.monotonic()
    print(json.dumps({'event': 'worker_loading', 'worker': worker_id,
                      'device': str(device)}), flush=True)
    # Limit concurrent checkpoint/stock loading to four workers. Do not pickle
    # large CUDA models from the parent, or build 20 pandas frames at once.
    with load_gate:
        policy = MLPModel(config['assets']['policy'], config['assets']['templates'],
                          device=str(device))
        value = ValueMLP(1, 2048, 128, .1, device).to(device)
        value.load_state_dict(torch.load(config['assets']['value'], map_location=device))
        value.eval()
        stock = set()
        for chunk in pd.read_csv(config['assets']['stock'], usecols=['mol'],
                                 chunksize=200000):
            stock.update(chunk['mol'].dropna().tolist())
    def values(molecules):
        fps = batch_smiles_to_fp(molecules)
        with torch.no_grad():
            inputs = torch.as_tensor(np.asarray(fps), dtype=torch.float32, device=device)
            return value(inputs).reshape(-1).cpu().numpy().tolist()
    def expand(molecule):
        with torch.no_grad():
            return policy.run(molecule, topk=50)
    WORKER.update(worker=worker_id, config=config, stock=stock,
                  value_fn=values, policy_fn=expand)
    print(json.dumps({'event': 'worker_ready', 'worker': worker_id,
                      'device': str(device), 'stock_size': len(stock),
                      'loading_seconds': round(time.monotonic() - started, 2)}), flush=True)
    ready_queue.put(worker_id)
    # Model/stock startup must not compete with the 600-second target timers.
    start_event.wait()


def evaluate_target(task):
    import torch
    dataset, target_id, target = task
    cfg = WORKER['config']
    seed = cfg['base_seed'] + target_id
    torch.manual_seed(seed)
    started = time.monotonic()
    search = None
    error = None
    reason = None
    emitted = dropped = 0
    trace_path = Path(cfg['run_dir']) / 'traces' / dataset / f'target_{target_id:05d}.jsonl'
    trace = trace_path.open('w') if cfg['trace_enabled'] else None
    def emit(event):
        nonlocal emitted, dropped
        if trace is None:
            return
        if cfg['trace_limit'] and emitted >= cfg['trace_limit']:
            dropped += 1
            return
        event = dict(event, dataset=dataset, target_id=target_id, target_seed=seed)
        trace.write(json.dumps(safe_json(event), ensure_ascii=False, allow_nan=False) + '\n')
        emitted += 1
    try:
        with time_limit(cfg['timeout_seconds']):
            search = StructureSearch(target, WORKER['stock'], WORKER['value_fn'],
                                     WORKER['policy_fn'], seed=seed,
                                     max_budget=cfg['max_budget'],
                                     reroot_config=RerootConfig(**cfg['reroot_config']),
                                     trace_fn=emit,
                                     depth_penalty_config=DepthPenaltyConfig(**cfg['depth_penalty_config']))
            search.search()
            reason = search.termination_reason
    except TargetTimeout as exc:
        reason, error = 'timeout', str(exc)
    except Exception as exc:
        reason, error = 'error:' + type(exc).__name__, traceback.format_exc()
        print(json.dumps({'event': 'target_error', 'dataset': dataset,
                          'target_id': target_id, 'error': error}), flush=True)
    finally:
        if trace:
            trace.close()
    elapsed = time.monotonic() - started
    success = reason == 'success'
    if search is not None:
        search.termination_reason = reason
    stats = search.stats() if search is not None else {}
    route = search.goal.route() if success else []
    route_states = []
    if success:
        keys = [search.graph.records[0].key] + [tuple(row['state']) for row in route]
        for key in keys:
            record = search.graph.records[search.graph.by_key[key]]
            route_states.append({'state_id': record.id, 'state': list(key),
                                 'min_step_depth': record.min_step_depth,
                                 'born_call': record.born_call,
                                 'selected_call': record.selected_call,
                                 'wait_calls': (None if record.selected_call is None
                                                else record.selected_call - record.born_call)})
    return {'dataset': dataset, 'target_id': target_id, 'target': target,
            'target_seed': seed, 'worker': WORKER['worker'], 'success': success,
            'termination_reason': reason, 'model_calls': stats.get('model_calls', 0),
            'elapsed_seconds': elapsed, 'route_length': len(route) if success else None,
            'route': route, 'success_route_states': route_states,
            'stats': stats, 'error': error,
            'trace_file': str(trace_path) if trace else None,
            'trace_events_written': emitted, 'trace_events_dropped': dropped}


def summarize(records, dataset, config):
    records = sorted(records, key=lambda r: r['target_id'])
    out = Path(config['run_dir']) / 'test'
    tag = Path(config['run_dir']).name
    result_path = out / f'stat_Structure_{tag}_{dataset}.pkl'
    # Preserve the usual result fields for existing success-overlap scripts.
    data = {'target_id': [r['target_id'] for r in records],
            'target': [r['target'] for r in records],
            'success': [r['success'] for r in records],
            'counts': [r['model_calls'] if r['success'] else -1 for r in records],
            'depth': [r['route_length'] if r['success'] else 32 for r in records],
            'route': [[x['reaction'] for x in r['route']] for r in records],
            'template': [[x['template'] for x in r['route']] for r in records],
            'play_times': [r['elapsed_seconds'] for r in records],
            'expanded': [r['stats'].get('effective_expansions', 0) for r in records],
            'generated': [r['stats'].get('generated_outcomes', 0) for r in records],
            'selection_stats': [r['stats'].get('selection_counts', {}) for r in records],
            'search_stats': [dict(r['stats'], target_id=r['target_id']) for r in records],
            'targets': records}
    with result_path.open('wb') as handle:
        pickle.dump(data, handle, protocol=4)
    budgets = [b for b in [50, 100, 200, 300, 400, 500] if b <= config['max_budget']]
    counts = {str(b): sum(r['success'] and r['model_calls'] <= b for r in records)
              for b in budgets}
    successes = [r for r in records if r['success']]
    n = len(records)
    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else None
    stats_totals = {}
    for kind in ['selection_counts', 'real_call_counts', 'counters', 'timings',
                 'depth_penalty_totals', 'selected_min_step_depth_histogram',
                 'real_call_min_step_depth_histogram', 'cache_min_step_depth_histogram',
                 'selected_route_depth_histogram', 'real_call_route_depth_histogram']:
        total = Counter()
        for row in records:
            total.update(row['stats'].get(kind, {}))
        stats_totals[kind] = dict(total)
    summary = {'dataset': dataset, 'complete': True, 'target_count': n,
               'revision': config['revision'],
               'reroot_cost_mode': config['reroot_cost_mode'],
               'reroot_priority': config['reroot_priority'],
               'reroot_add_state_h': config['reroot_add_state_h'],
               'depth_penalty_config': config['depth_penalty_config'],
               'depth_penalty_scope': config['depth_penalty_scope'],
               'depth_penalty_definition': config['depth_penalty_definition'],
               'depth_measure': config['depth_measure'],
               'success_count': len(successes), 'success_rate': len(successes) / n if n else 0.,
               'success_at_budget': counts,
               'success_rate_at_budget': {b: value / n if n else 0. for b, value in counts.items()},
               'mean_success_model_calls': mean(successes, 'model_calls'),
               'mean_all_model_calls': mean(records, 'model_calls'),
               'mean_success_route_length': mean(successes, 'route_length'),
               'mean_all_elapsed_seconds': mean(records, 'elapsed_seconds'),
               'termination_reasons': dict(Counter(r['termination_reason'] for r in records)),
               'failed_target_ids': [r['target_id'] for r in records if not r['success']],
               'max_budget': config['max_budget'], 'timeout_seconds': config['timeout_seconds'],
               'base_seed': config['base_seed'], 'hybrid_schedule': config['schedule'],
               'reroot_config': config['reroot_config'], 'stats_totals': stats_totals,
               'graph_statistics': {
                   'sum_state_expansions': sum(r['stats'].get('state_expansions', 0) for r in records),
                   'sum_unique_states': sum(r['stats'].get('graph_nodes', 0) for r in records),
                   'max_open': max((r['stats'].get('max_open', 0) for r in records), default=0),
                   'max_selected_route_depth': max((r['stats'].get('max_selected_route_depth', 0) for r in records), default=0),
                   'max_selected_min_step_depth': max((r['stats'].get('max_selected_min_step_depth', 0) for r in records), default=0),
                   'max_labels_per_state': max((r['stats'].get('max_labels_per_state', 0) for r in records), default=0),
                   'max_communities': max((r['stats'].get('max_communities', 0) for r in records), default=0),
                   'clustering_updates': sum(r['stats'].get('clustering_updates', 0) for r in records),
                   'clustering_seconds': sum(r['stats'].get('clustering_seconds', 0.) for r in records)},
               'result_file': str(result_path),
               'targets_file': str(out / f'targets_{tag}_{dataset}.jsonl'),
               'run_config': str(Path(config['run_dir']) / 'run_config.json')}
    write_json(out / f'summary_{tag}_{dataset}.json', summary)
    print(json.dumps({'event': 'dataset_complete', **summary}), flush=True)
    return summary


def parser():
    get = os.environ.get
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--datasets', default=get('KEEA_DATASETS', ','.join(DATASETS)))
    p.add_argument('--gpus', default=get('KEEA_GPUS', '0,1,2,3'))
    p.add_argument('--workers', type=int, default=int(get('KEEA_NUM_WORKERS', '20')))
    p.add_argument('--max-budget', type=int, default=int(get('KEEA_MAX_BUDGET', '500')))
    p.add_argument('--timeout', type=float, default=float(get('KEEA_TIMEOUT', '600')))
    p.add_argument('--seed', type=int, default=int(get('KEEA_SEED', '20260920')))
    p.add_argument('--cpu-threads', type=int, default=int(get('KEEA_CPU_THREADS', '1')))
    p.add_argument('--target-limit', type=int, default=int(get('KEEA_TARGET_LIMIT', '0')))
    p.add_argument('--alpha', type=float, default=float(get('KEEA_REROOT_ALPHA', '10')))
    p.add_argument('--cluster-first', type=int, default=int(get('KEEA_REROOT_CLUSTER_FIRST', '32')))
    p.add_argument('--leiden-resolution', type=float, default=float(get('KEEA_LEIDEN_RESOLUTION', '1')))
    p.add_argument('--leiden-iterations', type=int, default=int(get('KEEA_LEIDEN_ITERATIONS', '2')))
    p.add_argument('--trace', type=int, choices=[0, 1], default=int(get('KEEA_TRACE_ENABLED', '1')))
    p.add_argument('--trace-limit', type=int, default=int(get('KEEA_TRACE_LIMIT', '2000')))
    p.add_argument('--depth-penalty-start', type=int,
                   default=int(get('KEEA_DEPTH_PENALTY_START', '20')))
    p.add_argument('--depth-penalty-strong-start', type=int,
                   default=int(get('KEEA_DEPTH_PENALTY_STRONG_START', '40')))
    p.add_argument('--depth-penalty-soft-slope', type=float,
                   default=float(get('KEEA_DEPTH_PENALTY_SOFT_SLOPE', '0.05')))
    p.add_argument('--depth-penalty-strong-slope', type=float,
                   default=float(get('KEEA_DEPTH_PENALTY_STRONG_SLOPE', '1.0')))
    return p


def main():
    args = parser().parse_args()
    config, targets = configure(args)
    print(json.dumps({'event': 'preflight_ok',
                      'datasets': config['dataset_sizes'], 'workers': config['workers'],
                      'logical_gpus': config['logical_gpus'], 'max_budget': config['max_budget'],
                      'timeout_seconds': config['timeout_seconds'], 'seed': config['base_seed'],
                      'reroot_cost_mode': config['reroot_cost_mode'],
                      'depth_penalty_config': config['depth_penalty_config'],
                      'depth_penalty_scope': config['depth_penalty_scope'],
                      'schedule': config['schedule'], 'assets': config['assets']}), flush=True)
    if args.preflight:
        return
    if not args.run_dir:
        raise ValueError('--run-dir is required; use the project run_evaluation.sh')
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'run_config.json').exists():
        raise FileExistsError('Refusing to overwrite an existing run: ' + str(run_dir))
    config['run_dir'] = str(run_dir)
    (run_dir / 'test').mkdir(exist_ok=True)
    for dataset in config['datasets']:
        (run_dir / 'traces' / dataset).mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'run_config.json', config)
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    load_gate = ctx.Semaphore(min(4, config['workers']))
    ready_queue = ctx.Queue()
    start_event = ctx.Event()
    summaries = []
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=ctx,
                             initializer=initialize_worker,
                             initargs=(config, counter, load_gate, ready_queue, start_event)) as pool:
        for dataset in config['datasets']:
            records, ok = [], 0
            rows = targets[dataset]
            print(json.dumps({'event': 'dataset_start', 'dataset': dataset,
                              'total': len(rows)}), flush=True)
            futures = {pool.submit(evaluate_target, (dataset, i, target)): i
                       for i, target in enumerate(rows)}
            if not start_event.is_set():
                ready = set()
                try:
                    while len(ready) < config['workers']:
                        try:
                            ready.add(ready_queue.get(timeout=10))
                        except queue.Empty:
                            # An initializer/worker failure breaks the pool.
                            # Surface it instead of waiting forever.
                            for future in futures:
                                if future.done() and future.exception() is not None:
                                    future.result()
                        print(json.dumps({'event': 'initialization_progress',
                                          'ready_workers': len(ready),
                                          'total_workers': config['workers']}), flush=True)
                finally:
                    start_event.set()
            path = run_dir / 'test' / f'targets_{run_dir.name}_{dataset}.jsonl'
            with path.open('w') as writer:
                for future in as_completed(futures):
                    # A dead process raises BrokenProcessPool. Stop explicitly;
                    # never manufacture a complete dataset with missing targets.
                    result = future.result()
                    records.append(result)
                    ok += result['success']
                    writer.write(json.dumps(safe_json(result), ensure_ascii=False,
                                             allow_nan=False) + '\n')
                    writer.flush()
                    print(json.dumps({'event': 'target_done', 'dataset': dataset,
                                      'target_id': result['target_id'], 'done': len(records),
                                      'total': len(rows), 'success': result['success'],
                                      'success_so_far': ok, 'model_calls': result['model_calls'],
                                      'termination_reason': result['termination_reason'],
                                      'elapsed_seconds': round(result['elapsed_seconds'], 2)}), flush=True)
            summaries.append(summarize(records, dataset, config))
    write_json(run_dir / 'summary_all.json', {'complete': True, 'datasets': summaries})
    print(json.dumps({'event': 'all_datasets_complete', 'datasets': config['datasets'],
                      'run_dir': str(run_dir)}), flush=True)


if __name__ == '__main__':
    main()
