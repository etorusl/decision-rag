"""Reproducible evaluation with no answer caching and explicit timing scope."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from pathlib import Path

from .backends.base import sum_known
from .metrics import aggregate, latency_summary, retrieval_metrics
from .observability import RunReporter
from .text_recall import TextRecall, group_relevance, is_multidoc2dial, text_groups


def synchronize():
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def environment():
    import importlib.metadata
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    result = {"python": sys.version, "platform": platform.platform(), "git_commit": git('rev-parse', 'HEAD'),
              "git_dirty": bool(git('status', '--porcelain')), "packages": {}}
    for name in ['decision-rag', 'numpy', 'torch', 'transformers', 'sentence-transformers', 'peft', 'vllm']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if 'torch' in sys.modules:
        import torch
        result['cuda'] = torch.version.cuda
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name()
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _conversation_groups(queries):
    groups = {}
    for index, query in enumerate(queries, 1):
        key = ('conversation', query.conversation_id) if query.conversation_id else ('query', query.id)
        groups.setdefault(key, []).append((index, query))
    return list(groups.values())


def _run_concurrent(workers, groups, collect, query_started, reporter):
    """One persistent task per worker; only the collector touches logs and results."""
    if not workers:
        return 0.0, 0
    events = queue.Queue(maxsize=2 * len(workers))
    stopped, start_gate = threading.Event(), threading.Event()
    scheduling, activity = threading.Lock(), threading.Lock()
    pending = iter(groups)
    active = observed = 0

    def run(worker):
        nonlocal active, observed
        start_gate.wait()
        try:
            while not stopped.is_set():
                with scheduling:
                    group = next(pending, None) if not stopped.is_set() else None
                if group is None:
                    break
                worker.reset()
                for index, query in group:
                    if stopped.is_set():
                        break
                    events.put(('start', index, query))
                    if stopped.is_set():
                        break
                    with activity:
                        active += 1
                        observed = max(observed, active)
                    started = time.perf_counter()
                    try:
                        if hasattr(worker, 'prepare_query'):
                            worker.prepare_query(query)
                        result = worker.retrieve(query)
                    except BaseException as exc:  # noqa: BLE001 -- propagate to collector after draining workers
                        stopped.set()
                        events.put(('error', index, query, exc))
                        break
                    finally:
                        elapsed = time.perf_counter() - started
                        with activity:
                            active -= 1
                    events.put(('result', index, query, result, elapsed))
        except BaseException as exc:  # noqa: BLE001 -- even interrupted workers must notify the collector
            stopped.set()
            events.put(('error', None, None, exc))
        finally:
            events.put(('done',))

    failure = None
    finished = completed = failed = 0
    with ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix='retrieval') as pool:
        try:
            for worker in workers:
                pool.submit(run, worker)
        except BaseException:
            # A partial executor startup must not strand workers at the gate.
            stopped.set()
            start_gate.set()
            raise
        started = time.perf_counter()
        start_gate.set()
        while finished < len(workers):
            try:
                event, *payload = events.get()
                if event == 'done':
                    finished += 1
                elif event == 'error':
                    index, query, exc = payload
                    failed += int(query is not None)
                    if failure is None:
                        failure = exc
                    stopped.set()
                    reporter.event('query_failed', console=False, query_index=index,
                                   query_id=query.id if query is not None else None, error=str(exc))
                elif event == 'result':
                    completed += 1
                    if failure is None:
                        collect(*payload)
                elif failure is None:
                    query_started(*payload)
            except BaseException as exc:  # noqa: BLE001 -- stop and drain before reraising the original failure
                # Drain bounded completion queues even if the collector/log disk fails.
                if failure is None:
                    failure = exc
                stopped.set()
    wall_seconds = time.perf_counter() - started
    if failure is not None:
        # Logging failure must not replace the original query/collector exception.
        with suppress(Exception):
            reporter.event('benchmark_failed', console=False, completed_queries=completed,
                           failed_queries=failed, wall_seconds=wall_seconds,
                           observed_query_concurrency=observed)
        raise failure
    return wall_seconds, observed


def _restore_query_order(output, offsets):
    """Sort the on-disk completion log using offsets, without retaining result payloads."""
    path, ordered = output / 'queries.jsonl', output / '.queries.ordered.jsonl'
    try:
        with path.open(encoding='utf-8') as source, ordered.open('w', encoding='utf-8') as target:
            for index in sorted(offsets):
                source.seek(offsets[index])
                target.write(source.readline())
        ordered.replace(path)
    finally:
        ordered.unlink(missing_ok=True)


def _quality_for_ranking(query, ranking, canonical, text_recall):
    relevant = query.relevant_ids
    if canonical:
        ranking = [canonical[passage_id] for passage_id in ranking]
        relevant = group_relevance(relevant, canonical)
    metrics = retrieval_metrics(ranking, relevant)
    extra = {'text_recall': text_recall.add(query, ranking, metrics)} if text_recall is not None else {}
    return metrics, extra


def _ranking_limit(retriever, config):
    limit = getattr(retriever, 'top_k', None)
    if limit is None:
        limit = getattr(getattr(retriever, 'config', None), 'top_k', None)
    if limit is None and config:
        limit = config.get('args', {}).get('top_k', config.get('top_k'))
    return limit


def _evaluate_throughput(retriever, queries, output, *, warmup, config, source_paths, canonical,
                         dedup_normalization, text_recall,
                         initialization_seconds, gpu_monitor_index, reporter, query_concurrency, worker_factory):
    groups = _conversation_groups(queries)
    effective = min(query_concurrency, len(groups))
    workers, owned = [], {}
    failure = None
    try:
        construction_started = time.perf_counter()
        for _ in range(effective):
            worker = worker_factory()
            if worker is not retriever:
                owned[id(worker)] = worker
            if not callable(getattr(worker, 'retrieve', None)) or not callable(getattr(worker, 'reset', None)):
                raise TypeError('worker_factory must return a retriever with retrieve() and reset()')
            if any(worker is other for other in workers) and hasattr(worker, 'memory'):
                raise ValueError('Stateful retrievers require an independent worker instance')
            backend = getattr(worker, 'backend', None)
            if backend is not None and any(backend is getattr(other, 'backend', None) for other in workers):
                raise ValueError('Decision workers require independent backends and per-query cache salts')
            workers.append(worker)
        worker_initialization_seconds = time.perf_counter() - construction_started
        if initialization_seconds is not None:
            initialization_seconds += worker_initialization_seconds
        per_worker_warmup = min(warmup, len(queries))
        actual_warmup = per_worker_warmup * effective
        reporter.event('benchmark_start', queries=len(queries), warmup_queries=actual_warmup,
                       execution_mode='throughput', query_concurrency=query_concurrency,
                       effective_query_concurrency=effective, output=str(output))
        warmup_seconds = 0.0
        # Warm every logical worker equally, including workers sharing a stateless retriever.
        with reporter.phase('Warmup', total=actual_warmup, unit='query') as phase:
            for worker in workers:
                worker.reset()
                for query in queries[:per_worker_warmup]:
                    started = time.perf_counter()
                    if hasattr(worker, 'prepare_query'):
                        worker.prepare_query(query)
                    worker.retrieve(query)
                    warmup_seconds += time.perf_counter() - started
                    phase.update()
                worker.reset()
        summary = _measure_throughput(
            retriever, queries, workers, groups, output, config=config, source_paths=source_paths,
            canonical=canonical, dedup_normalization=dedup_normalization, text_recall=text_recall,
            initialization_seconds=initialization_seconds,
            worker_initialization_seconds=worker_initialization_seconds,
            warmup_seconds=warmup_seconds, warmup_queries=actual_warmup,
            warmup_queries_per_worker=per_worker_warmup, gpu_monitor_index=gpu_monitor_index,
            reporter=reporter, query_concurrency=query_concurrency)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        close_failure = None
        for worker in owned.values():
            try:
                if hasattr(worker, 'close'):
                    worker.close()
            except BaseException as exc:  # noqa: BLE001 -- close every owned worker before propagating
                if close_failure is None:
                    close_failure = exc
        if close_failure is not None and failure is None:
            raise close_failure
    # A failed worker/collector/cleanup must never leave a successful summary.
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    reporter.event('benchmark_complete', queries=len(queries), quality=summary['quality'],
                   latency=summary['latency'], throughput=summary['throughput'],
                   summary_path=str(output / 'summary.json'))
    return summary


def _measure_throughput(retriever, queries, workers, groups, output, *, config, source_paths, canonical,
                        dedup_normalization, text_recall,
                        initialization_seconds, worker_initialization_seconds, warmup_seconds, warmup_queries,
                        warmup_queries_per_worker, gpu_monitor_index, reporter, query_concurrency):
    metrics, latencies, component_times, offsets = {}, {}, {}, {}
    work = {key: 0 for key in ('input_tokens', 'forward_passes', 'requests')}
    total_latency, recall_sum, labeled = 0.0, 0.0, 0
    from .telemetry import GPUMonitor
    monitor = GPUMonitor(gpu_monitor_index)
    with reporter.phase('Benchmark', total=len(queries), unit='query') as phase, \
            monitor, (output / 'queries.jsonl').open('w', encoding='utf-8') as stream:
        def collect(index, query, result, elapsed):
            nonlocal total_latency, recall_sum, labeled
            measured, extra = _quality_for_ranking(query, result.ids, canonical, text_recall)
            record = {**result.to_dict(), 'query_index': index, 'conversation_id': query.conversation_id,
                      'turn_id': query.turn_id, 'latency_seconds': elapsed, 'metrics': measured,
                      'labeled': bool(measured), **extra}
            offsets[index] = stream.tell()
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            metrics[index], latencies[index] = measured, elapsed
            for stage in result.stages:
                for component in ('query_embedding', 'vector_search'):
                    value = stage.get(component + '_seconds')
                    if value is not None:
                        component_times.setdefault(component, []).append(value)
                for key in ('prompt_requests', 'completion_tokens', 'cached_tokens'):
                    if key in stage:
                        work.setdefault(key, 0)
                for key, value in work.items():
                    work[key] = sum_known([value, stage.get(key, 0)])
            total_latency += elapsed
            if measured:
                labeled += 1
                recall_sum += measured['recall@10']
            progress_metrics = {'mean_latency_s': total_latency / len(latencies), 'labeled_queries': labeled}
            if labeled:
                progress_metrics['recall_at_10'] = recall_sum / labeled
            reporter.event('query_complete', console=False, query_index=index, query_id=query.id,
                           latency_seconds=elapsed, metrics=measured)
            phase.update(**progress_metrics)

        def query_started(index, query):
            reporter.event('query_start', console=False, query_index=index, query_id=query.id)

        wall_seconds, observed = _run_concurrent(workers, groups, collect, query_started, reporter)
    _restore_query_order(output, offsets)
    external = bool(getattr(retriever, 'external_inference', False))
    local_search_device = getattr(retriever, 'local_search_device', None)
    process = 'benchmark_process'
    if external:
        process = ('external_vllm_server_and_local_gpu_search' if str(local_search_device).startswith('cuda')
                   else 'external_vllm_server')

    def concurrent_latency(values):
        stats = latency_summary(values)
        stats.pop('serial_queries_per_second', None)
        return stats

    return {
        'quality': aggregate([metrics[index] for index in sorted(metrics)]),
        **({'text_recall': text_recall.summary()} if text_recall is not None else {}),
        'labeled_queries': labeled, 'unlabeled_queries': len(queries) - labeled,
        'latency': concurrent_latency([latencies[index] for index in sorted(latencies)]),
        'throughput': {'queries_per_second': len(latencies) / wall_seconds if wall_seconds else None,
                       'wall_seconds': wall_seconds, 'successful_queries': len(latencies),
                       'completed_queries': len(latencies), 'failed_queries': 0},
        'execution': {'mode': 'throughput', 'query_concurrency': query_concurrency,
                      'effective_query_concurrency': len(workers), 'observed_query_concurrency': observed,
                      'groups': len(groups), 'scheduling': 'closed_loop_conversations',
                      'latency_scope': 'prepare_query + retrieve; excludes waiting for conversation worker',
                      'result_order': 'original_input_order', 'completion_buffer_size': 2 * len(workers)},
        'warmup_queries': warmup_queries, 'warmup_queries_per_worker': warmup_queries_per_worker,
        'warmup_seconds': warmup_seconds, 'initialization_seconds': initialization_seconds,
        'worker_initialization_seconds': worker_initialization_seconds,
        'timing_scope': 'Closed-loop conversation workers after initialization and per-worker warmup; '
                        'wall time includes scheduling, prepare_query, retrieval and collector; no result cache',
        'dedup_text': dedup_normalization is not None, 'dedup_normalization': dedup_normalization,
        'environment': environment(), 'config': config or {},
        'inputs': {str(path): file_hash(path) for path in source_paths}, 'work': work,
        'component_latency': {name: concurrent_latency(values) for name, values in component_times.items()},
        'local_search_device': local_search_device, 'inference_process': process,
        'gpu_telemetry': monitor.summary(),
    }


def evaluate(retriever, queries, output_dir, *, warmup=0, config=None, source_paths=(), dedup_text=False,
             initialization_seconds=None, gpu_monitor_index=None, reporter=None, progress='auto', log_interval=30.0,
             execution_mode='latency', query_concurrency=1, worker_factory=None):
    if execution_mode not in {'latency', 'throughput'}:
        raise ValueError('execution_mode must be latency or throughput')
    if type(query_concurrency) is not int or query_concurrency < 1:
        raise ValueError('query_concurrency must be a positive integer')
    if type(warmup) is not int or warmup < 0:
        raise ValueError('warmup must be a nonnegative integer')
    if execution_mode == 'latency' and query_concurrency != 1:
        raise ValueError('query_concurrency > 1 requires throughput execution_mode')
    if execution_mode == 'throughput' and not callable(worker_factory):
        raise ValueError('throughput requires a callable worker_factory')
    output = Path(output_dir)
    if (output / 'queries.jsonl').exists():
        raise ValueError("Output already contains a run. Choose a new --output directory.")
    reporting = nullcontext(reporter) if reporter is not None else RunReporter(
        output, progress=progress, log_interval=log_interval, name='benchmark')
    with reporting as active_reporter:
        return _evaluate(retriever, queries, output, warmup=warmup, config=config, source_paths=source_paths,
                         dedup_text=dedup_text, initialization_seconds=initialization_seconds,
                         gpu_monitor_index=gpu_monitor_index, reporter=active_reporter,
                         execution_mode=execution_mode, query_concurrency=query_concurrency,
                         worker_factory=worker_factory)


def _evaluate(retriever, queries, output, *, warmup, config, source_paths, dedup_text,
              initialization_seconds, gpu_monitor_index, reporter, execution_mode, query_concurrency,
              worker_factory):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs")
    passages = {p.id for p in retriever.corpus.passages}
    for query in queries:
        if any(not math.isfinite(float(v)) for v in query.relevant_ids.values()):
            raise ValueError("Relevance grades must be finite")
        unknown = set(query.relevant_ids) - passages
        if unknown:
            raise ValueError(f"Unknown gold passage IDs for {query.id}: {sorted(unknown)[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    # Ensure the chronological stateful protocol is unambiguous.
    last = {}
    for query in queries:
        if query.conversation_id:
            previous = last.get(query.conversation_id, -1)
            if query.turn_id <= previous:
                raise ValueError(f"Non-increasing turns in conversation {query.conversation_id}")
            last[query.conversation_id] = query.turn_id
    canonical, dedup_normalization, text_recall = {}, None, None
    if is_multidoc2dial(queries):
        text_recall = TextRecall(retriever.corpus, ranking_limit=_ranking_limit(retriever, config))
    elif dedup_text:
        lowercase = bool(queries) and all(query.metadata.get('dataset') == 'mtrag' for query in queries)
        canonical = text_groups(retriever.corpus, lowercase=lowercase)
        dedup_normalization = 'whitespace+lowercase' if lowercase else 'whitespace'
    if execution_mode == 'throughput':
        return _evaluate_throughput(
            retriever, queries, output, warmup=warmup, config=config, source_paths=source_paths,
            canonical=canonical, dedup_normalization=dedup_normalization, text_recall=text_recall,
            initialization_seconds=initialization_seconds,
            gpu_monitor_index=gpu_monitor_index, reporter=reporter, query_concurrency=query_concurrency,
            worker_factory=worker_factory)
    external = bool(getattr(retriever, "external_inference", False))
    reporter.event('benchmark_start', queries=len(queries), warmup_queries=min(warmup, len(queries)),
                   external_inference=external, output=str(output))
    def sync():
        if not external:
            synchronize()
    def prepare(query):
        if hasattr(retriever, "prepare_query"):
            retriever.prepare_query(query)
    retriever.reset()
    sync()
    warmup_seconds = 0.0
    with reporter.phase('Warmup', total=min(warmup, len(queries)), unit='query') as phase:
        for query in queries[:warmup]:
            warmup_start = time.perf_counter()
            prepare(query)
            retriever.retrieve(query)
            sync()
            warmup_seconds += time.perf_counter() - warmup_start
            phase.update()
    retriever.reset()  # warmup decisions must never leak into measured conversations
    sync()
    if not external and 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    metrics, latencies = [], []
    component_times = {}
    total_latency, recall_sum, labeled = 0.0, 0.0, 0
    work = {key: 0 for key in ("input_tokens", "forward_passes", "requests")}
    from .telemetry import GPUMonitor
    monitor = GPUMonitor(gpu_monitor_index)
    with reporter.phase('Benchmark', total=len(queries), unit='query') as phase, \
            monitor, (output / 'queries.jsonl').open('w', encoding='utf-8') as stream:
        for query_index, query in enumerate(queries, 1):
            reporter.event('query_start', console=False, query_index=query_index, query_id=query.id)
            prepare(query)
            sync()
            start = time.perf_counter()
            result = retriever.retrieve(query)
            sync()
            elapsed = time.perf_counter() - start
            measured, extra = _quality_for_ranking(query, result.ids, canonical, text_recall)
            record = {**result.to_dict(), "conversation_id": query.conversation_id,
                      "turn_id": query.turn_id, "latency_seconds": elapsed, "metrics": measured,
                      "labeled": bool(measured), **extra}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            metrics.append(measured)
            latencies.append(elapsed)
            for stage in result.stages:
                for component in ('query_embedding', 'vector_search'):
                    value = stage.get(component + '_seconds')
                    if value is not None:
                        component_times.setdefault(component, []).append(value)
                for key in ("prompt_requests", "completion_tokens", "cached_tokens"):
                    if key in stage:
                        work.setdefault(key, 0)
                for key, value in work.items():
                    work[key] = sum_known([value, stage.get(key, 0)])
            total_latency += elapsed
            if measured:
                labeled += 1
                recall_sum += measured['recall@10']
            progress_metrics = {'mean_latency_s': total_latency / query_index, 'labeled_queries': labeled}
            if labeled:
                progress_metrics['recall_at_10'] = recall_sum / labeled
            reporter.event('query_complete', console=False, query_index=query_index, query_id=query.id,
                           latency_seconds=elapsed, metrics=measured)
            phase.update(**progress_metrics)
    local_search_device = getattr(retriever, 'local_search_device', None)
    process = 'benchmark_process'
    if external:
        process = ('external_vllm_server_and_local_gpu_search' if str(local_search_device).startswith('cuda')
                   else 'external_vllm_server')
    summary = {"quality": aggregate(metrics), "labeled_queries": sum(bool(m) for m in metrics),
               "unlabeled_queries": sum(not bool(m) for m in metrics),
               "latency": latency_summary(latencies), "warmup_queries": min(warmup, len(queries)),
               "initialization_seconds": initialization_seconds,
               "warmup_seconds": warmup_seconds,
               "timing_scope": "Complete serial retrieval after model/corpus/index initialization; no result cache",
               "execution": {"mode": "latency", "query_concurrency": 1,
                             "effective_query_concurrency": int(bool(queries)),
                             "observed_query_concurrency": int(bool(queries)),
                             "scheduling": "serial_input_order",
                             "latency_scope": "retrieve only; excludes prepare_query and collector"},
               "dedup_text": dedup_normalization is not None,
               "dedup_normalization": dedup_normalization,
               "environment": environment(), "config": config or {},
               "inputs": {str(p): file_hash(p) for p in source_paths},
               "work": work,
               "component_latency": {name: latency_summary(values) for name, values in component_times.items()},
               "local_search_device": local_search_device,
               "inference_process": process,
               "gpu_telemetry": monitor.summary()}
    if text_recall is not None:
        summary['text_recall'] = text_recall.summary()
    if not external and 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            summary['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    reporter.event('benchmark_complete', queries=len(queries), quality=summary['quality'],
                   latency=summary['latency'], summary_path=str(output / 'summary.json'))
    return summary
