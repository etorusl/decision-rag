"""Closed-loop scheduling, state isolation and cleanup without a GPU/server."""
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from decision_rag.evaluation import evaluate
from decision_rag.retrieval import RetrievalResult
from decision_rag.schema import Corpus, Node, Query, read_jsonl


@pytest.fixture
def corpus():
    return Corpus([Node('root', 'Corpus', 'root'), Node('page', 'Page', 'page', 'root'),
                   Node('p', 'Evidence', 'passage', 'page')])


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr('decision_rag.evaluation.environment', lambda: {'test': True})
    def forbidden():
        pytest.fail('Concurrent requests must not issue process-wide CUDA synchronization')
    monkeypatch.setattr('decision_rag.evaluation.synchronize', forbidden)


class Worker:
    external_inference = True

    def __init__(self, corpus, hook=None):
        self.corpus, self.hook = corpus, hook
        self.backend = SimpleNamespace(salt=None)
        self.memory, self.calls = {}, []
        self.prepared = self.closed = self.resets = 0

    def reset(self):
        self.resets += 1
        self.memory.clear()

    def prepare_query(self, query):
        self.prepared += 1
        self.backend.salt = f'{id(self.backend)}:{self.prepared}'

    def retrieve(self, query):
        assert not self.closed
        previous = list(self.memory.get(query.conversation_id, []))
        self.calls.append((query.id, previous, self.backend.salt))
        if self.hook:
            self.hook(query)
        self.memory.setdefault(query.conversation_id, []).append(query.id)
        return RetrievalResult(query.id, ['p'], [1.0], 0.0,
                               [{'requests': 1, 'input_tokens': 10, 'forward_passes': None,
                                 'salt': self.backend.salt, 'worker': id(self)}])

    def close(self):
        self.closed += 1


def run(original, queries, output, factory, **kwargs):
    return evaluate(original, queries, output, execution_mode='throughput',
                    worker_factory=factory, progress='off', **kwargs)


def test_conversations_overlap_but_turns_keep_memory_and_outputs_keep_input_order(corpus, tmp_path):
    rendezvous = threading.Barrier(2, timeout=5)
    def hook(query):
        if query.id in {'a1', 'b1'}:
            rendezvous.wait()
    workers = []
    def factory():
        worker = Worker(corpus, hook)
        workers.append(worker)
        return worker
    original = Worker(corpus)
    queries = [Query('a1', 'a', {'p': 1}, 'a', 1), Query('b1', 'b', {'p': 1}, 'b', 1),
               Query('a2', 'a', {'p': 1}, 'a', 2), Query('b2', 'b', {'p': 1}, 'b', 2),
               Query('independent', 'q', {'p': 1})]
    summary = run(original, queries, tmp_path, factory, query_concurrency=2)
    rows = list(read_jsonl(tmp_path / 'queries.jsonl'))
    assert [row['query_id'] for row in rows] == [query.id for query in queries]
    assert [row['query_index'] for row in rows] == [1, 2, 3, 4, 5]
    calls = {qid: (previous, salt, worker) for worker in workers for qid, previous, salt in worker.calls}
    assert calls['a1'][0] == calls['b1'][0] == calls['independent'][0] == []
    assert calls['a2'][0] == ['a1'] and calls['b2'][0] == ['b1']
    assert calls['a1'][2] is calls['a2'][2]
    assert calls['b1'][2] is calls['b2'][2]
    assert calls['a1'][2] is not calls['b1'][2]
    assert len({row['stages'][0]['salt'] for row in rows}) == len(queries)
    assert summary['execution']['observed_query_concurrency'] == 2
    assert summary['execution']['groups'] == 3
    assert summary['execution']['result_order'] == 'original_input_order'
    assert summary['quality']['recall@1'] == 1
    assert summary['throughput']['successful_queries'] == 5
    assert summary['throughput']['failed_queries'] == 0
    assert summary['throughput']['queries_per_second'] == 5 / summary['throughput']['wall_seconds']
    assert 'serial_queries_per_second' not in summary['latency']
    assert summary['work'] == {'requests': 5, 'input_tokens': 50, 'forward_passes': None}
    assert all(worker.closed == 1 for worker in workers)
    assert not original.calls and original.closed == 0


def test_warmup_is_per_logical_worker_and_never_leaks_state_or_work(corpus, tmp_path):
    workers = []
    def factory():
        worker = Worker(corpus)
        workers.append(worker)
        return worker
    queries = [Query('a1', 'a', {'p': 1}, 'a', 1), Query('a2', 'a', {'p': 1}, 'a', 2),
               Query('b1', 'b', {'p': 1}, 'b', 1)]
    summary = run(Worker(corpus), queries, tmp_path, factory, warmup=2, query_concurrency=2)
    assert summary['warmup_queries'] == 4 and summary['warmup_queries_per_worker'] == 2
    for worker in workers:
        assert [(qid, previous) for qid, previous, _ in worker.calls[:2]] == [('a1', []), ('a2', ['a1'])]
        for qid, previous, _ in worker.calls[2:]:
            assert previous == (['a1'] if qid == 'a2' else [])
    assert sum(len(worker.calls) for worker in workers) == 7
    assert summary['work']['requests'] == 3
    assert summary['latency']['queries'] == 3


def test_shared_stateless_original_warms_each_logical_worker_and_stays_owned_by_caller(corpus, tmp_path):
    class Stateless:
        external_inference = True
        def __init__(self):
            self.corpus, self.calls, self.closed = corpus, [], 0
        def reset(self):
            pass
        def retrieve(self, query):
            self.calls.append(query.id)
            return RetrievalResult(query.id, ['p'], [1.0], 0, [])
        def close(self):
            self.closed += 1
    original = Stateless()
    queries = [Query('q1', 'q'), Query('q2', 'q')]
    summary = run(original, queries, tmp_path, lambda: original, warmup=1, query_concurrency=2)
    assert summary['warmup_queries'] == 2
    assert original.calls[:2] == ['q1', 'q1']
    assert sorted(original.calls[2:]) == ['q1', 'q2']
    assert original.closed == 0


def test_worker_count_is_capped_by_conversations_not_turns(corpus, tmp_path):
    workers = []
    def factory():
        workers.append(Worker(corpus))
        return workers[-1]
    queries = [Query(f'q{i}', 'q', conversation_id='one', turn_id=i) for i in range(4)]
    summary = run(Worker(corpus), queries, tmp_path, factory, query_concurrency=8)
    assert len(workers) == 1
    assert summary['execution']['query_concurrency'] == 8
    assert summary['execution']['effective_query_concurrency'] == 1
    assert summary['execution']['observed_query_concurrency'] == 1


def test_concurrent_timing_separates_construction_warmup_and_loaded_latency(corpus, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('decision_rag.evaluation.time.perf_counter', lambda: clock[0])
    class Timed(Worker):
        def prepare_query(self, query):
            clock[0] += 2
            super().prepare_query(query)
        def retrieve(self, query):
            clock[0] += 3
            return super().retrieve(query)
    class Reporter:
        def event(self, event, **kwargs):
            if event == 'query_complete':
                clock[0] += 7
        @contextmanager
        def phase(self, *args, **kwargs):
            yield SimpleNamespace(update=lambda **metrics: None)
    def factory():
        clock[0] += 11
        return Timed(corpus)
    summary = run(Worker(corpus), [Query('q', 'q')], tmp_path, factory, warmup=1,
                  initialization_seconds=13, reporter=Reporter())
    assert summary['worker_initialization_seconds'] == 11
    assert summary['initialization_seconds'] == 24
    assert summary['warmup_seconds'] == 5
    assert summary['latency']['mean_seconds'] == 5  # prepare and retrieve
    assert summary['throughput']['wall_seconds'] == 12  # includes collector
    assert summary['throughput']['queries_per_second'] == 1 / 12


def test_failure_stops_new_turns_and_groups_drains_workers_and_closes_once(corpus, tmp_path):
    rendezvous = threading.Barrier(2, timeout=5)
    failure_started = threading.Event()
    workers = []
    def hook(query):
        if query.id in {'a1', 'b1'}:
            rendezvous.wait()
        if query.id == 'a1':
            failure_started.set()
            raise RuntimeError('inference disconnected')
        if query.id == 'b1':
            assert failure_started.wait(5)
    def factory():
        workers.append(Worker(corpus, hook))
        return workers[-1]
    queries = [Query('a1', 'q', conversation_id='a', turn_id=1),
               Query('b1', 'q', conversation_id='b', turn_id=1),
               Query('a2', 'q', conversation_id='a', turn_id=2)]
    # Make new groups wait for a worker; they must not all be pre-submitted.
    queries += [Query(f'later{i}', 'q') for i in range(100)]
    with pytest.raises(RuntimeError, match='disconnected'):
        run(Worker(corpus), queries, tmp_path, factory, query_concurrency=2)
    assert all(worker.closed == 1 for worker in workers)
    assert 'a2' not in [qid for worker in workers for qid, *_ in worker.calls]
    assert len([qid for worker in workers for qid, *_ in worker.calls]) < len(queries)
    assert not (tmp_path / 'summary.json').exists()
    events = list(read_jsonl(tmp_path / 'events.jsonl'))
    failed = next(row for row in events if row['event'] == 'benchmark_failed')
    assert failed['failed_queries'] == 1


def test_collector_failure_cancels_and_drains_bounded_event_queue(corpus, tmp_path):
    workers = []
    def factory():
        workers.append(Worker(corpus))
        return workers[-1]
    class FailingReporter:
        def event(self, event, **kwargs):
            if event == 'query_complete':
                raise OSError('log disk full')
        @contextmanager
        def phase(self, *args, **kwargs):
            yield SimpleNamespace(update=lambda **metrics: None)
    with pytest.raises(OSError, match='disk full'):
        run(Worker(corpus), [Query(f'q{i}', 'q') for i in range(100)], tmp_path,
            factory, query_concurrency=3, reporter=FailingReporter())
    assert all(worker.closed == 1 for worker in workers)
    assert not (tmp_path / 'summary.json').exists()


def test_factory_failure_closes_previously_created_workers_but_not_original(corpus, tmp_path):
    original, extra = Worker(corpus), Worker(corpus)
    values = iter([original, extra])
    def factory():
        try:
            return next(values)
        except StopIteration:
            raise RuntimeError('worker setup failed') from None
    with pytest.raises(RuntimeError, match='setup failed'):
        run(original, [Query(f'q{i}', 'q') for i in range(3)], tmp_path, factory, query_concurrency=3)
    assert extra.closed == 1 and original.closed == 0
    assert not original.calls and not extra.calls
    assert not (tmp_path / 'summary.json').exists()


def test_partial_executor_startup_releases_waiting_workers_and_closes_resources(corpus, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    class PartialExecutor:
        def __init__(self, **kwargs):
            self.pool, self.calls = ThreadPoolExecutor(**kwargs), 0
        def __enter__(self):
            self.pool.__enter__()
            return self
        def submit(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('thread creation failed')
            return self.pool.submit(*args)
        def __exit__(self, *args):
            return self.pool.__exit__(*args)
    monkeypatch.setattr('decision_rag.evaluation.ThreadPoolExecutor', PartialExecutor)
    workers = [Worker(corpus), Worker(corpus)]
    factory = iter(workers)
    with pytest.raises(RuntimeError, match='thread creation failed'):
        run(Worker(corpus), [Query('q1', 'q'), Query('q2', 'q')], tmp_path,
            lambda: next(factory), query_concurrency=2)
    assert all(worker.closed == 1 and not worker.calls for worker in workers)
    assert not (tmp_path / 'summary.json').exists()


def test_warmup_failure_closes_all_constructed_workers_without_summary(corpus, tmp_path):
    def fail(query):
        raise RuntimeError('warmup failed')
    workers = [Worker(corpus, fail), Worker(corpus)]
    factory = iter(workers)
    with pytest.raises(RuntimeError, match='warmup failed'):
        run(Worker(corpus), [Query('q1', 'q'), Query('q2', 'q')], tmp_path,
            lambda: next(factory), query_concurrency=2, warmup=1)
    assert all(worker.closed == 1 for worker in workers)
    assert not (tmp_path / 'queries.jsonl').exists()
    assert not (tmp_path / 'summary.json').exists()


def test_cleanup_failure_still_closes_remaining_workers_and_never_writes_summary(corpus, tmp_path):
    class BadClose(Worker):
        def close(self):
            super().close()
            raise OSError('closing client failed')
    workers = [BadClose(corpus), Worker(corpus)]
    factory = iter(workers)
    with pytest.raises(OSError, match='closing client failed'):
        run(Worker(corpus), [Query('q1', 'q'), Query('q2', 'q')], tmp_path,
            lambda: next(factory), query_concurrency=2)
    assert all(worker.closed == 1 for worker in workers)
    assert not (tmp_path / 'summary.json').exists()


@pytest.mark.parametrize('shared', ['retriever', 'backend'])
def test_stateful_worker_or_backend_sharing_is_rejected_before_requests(corpus, tmp_path, shared):
    original = Worker(corpus)
    first, second = Worker(corpus), Worker(corpus)
    if shared == 'backend':
        second.backend = first.backend
    values = iter([first, first if shared == 'retriever' else second])
    with pytest.raises(ValueError, match='independent'):
        run(original, [Query('q1', 'q'), Query('q2', 'q')], tmp_path,
            lambda: next(values), query_concurrency=2)
    assert not first.calls and not second.calls
    assert first.closed == 1
    assert second.closed == (1 if shared == 'backend' else 0)


@pytest.mark.parametrize('kwargs', [
    {'execution_mode': 'invalid'}, {'query_concurrency': 0}, {'query_concurrency': -1},
    {'query_concurrency': 1.5}, {'query_concurrency': True}, {'warmup': -1}, {'warmup': .5},
    {'execution_mode': 'latency', 'query_concurrency': 2}, {'worker_factory': None},
])
def test_malformed_execution_args_rejected_before_factory_or_output(corpus, tmp_path, kwargs):
    def forbidden():
        pytest.fail('Invalid arguments must not construct workers')
    arguments = {'execution_mode': 'throughput', 'worker_factory': forbidden, **kwargs}
    with pytest.raises(ValueError):
        evaluate(Worker(corpus), [Query('q', 'q')], tmp_path / 'run', **arguments)
    assert not (tmp_path / 'run').exists()


def test_empty_workload_has_no_workers_or_invented_throughput(corpus, tmp_path):
    def forbidden():
        pytest.fail('Empty workload needs no workers')
    summary = run(Worker(corpus), [], tmp_path, forbidden, query_concurrency=8, warmup=2)
    assert summary['execution']['effective_query_concurrency'] == 0
    assert summary['execution']['observed_query_concurrency'] == 0
    assert summary['throughput']['queries_per_second'] is None
    assert summary['warmup_queries'] == 0
    assert list(read_jsonl(tmp_path / 'queries.jsonl')) == []
    assert json.loads((tmp_path / 'summary.json').read_text()) == summary
