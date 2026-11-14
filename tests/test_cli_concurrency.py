"""CLI ownership and isolation for concurrent conversations, without model weights."""
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import httpx
import pytest

from decision_rag import cli
from decision_rag.backends.vllm import VLLMDecisionBackend
from decision_rag.demo import SmokeBackend, write_demo
from decision_rag.retrieval import DecisionRetriever
from decision_rag.schema import Query, write_jsonl


def command(tmp_path, *extra, conversations=2):
    data = tmp_path / 'data'
    write_demo(data)
    queries = [Query(f'{conversation}-{turn}', 'Who is the protagonist of Memento?', {'p1': 1},
                     conversation_id=str(conversation), turn_id=turn)
               for turn in (1, 2) for conversation in range(conversations)]
    write_jsonl(data / 'queries.jsonl', [asdict(query) for query in queries])
    return ['benchmark', '--corpus', str(data / 'corpus.jsonl'), '--queries', str(data / 'queries.jsonl'),
            '--output', str(tmp_path / 'output'), '--execution-mode', 'throughput',
            '--query-concurrency', str(conversations), '--warmup', '0', '--progress', 'off',
            '--top-k', '1', '--pages', '1', *extra]


def test_cli_smoke_conversations_use_private_backends_and_memory(monkeypatch, tmp_path):
    backends, retrievers = [], []
    overlap = threading.Barrier(2)

    class TrackedSmoke(SmokeBackend):
        def __init__(self):
            self.first_call, self.closed = True, 0
            backends.append(self)

        def binary(self, *args):
            if self.first_call:
                self.first_call = False
                overlap.wait(timeout=5)
            return super().binary(*args)

        def close(self):
            self.closed += 1

    class TrackedRetriever(DecisionRetriever):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            retrievers.append(self)

    monkeypatch.setattr('decision_rag.demo.SmokeBackend', TrackedSmoke)
    monkeypatch.setattr(cli, 'DecisionRetriever', TrackedRetriever)
    cli.main(command(tmp_path, '--backend', 'smoke'))
    assert len(retrievers) == len(backends) == 2
    assert len({id(worker.backend) for worker in retrievers}) == 2
    assert len({id(worker.memory) for worker in retrievers}) == 2
    assert len({id(worker.corpus) for worker in retrievers}) == 1
    assert sorted([list(worker.memory) for worker in retrievers]) == [['0'], ['1']]
    assert [backend.closed for backend in backends] == [1, 1]
    summary = json.loads((tmp_path / 'output' / 'summary.json').read_text())
    assert summary['execution']['observed_query_concurrency'] == 2
    rows = [json.loads(line) for line in (tmp_path / 'output' / 'queries.jsonl').read_text().splitlines()]
    assert [row['query_id'] for row in rows] == ['0-1', '1-1', '0-2', '1-2']


@pytest.mark.parametrize('method,target', [
    ('bm25', 'decision_rag.baselines.BM25Retriever'),
    ('dense', 'decision_rag.vllm_retrieval.VLLMDenseRetriever'),
    ('dense-rerank', 'decision_rag.vllm_retrieval.VLLMDenseRerankRetriever'),
])
def test_cli_stateless_workers_share_one_index_and_close_once(monkeypatch, tmp_path, method, target):
    created = []

    class IndexedRetriever:
        def __init__(self, corpus, *args, **kwargs):
            assert kwargs.get('search_device', kwargs.get('device')) == 'cuda'
            self.corpus, self.closed = corpus, 0
            self.gpu_index = object()
            self.runtime_metadata = {'search_runtime': {'device': 'cuda:0'}}
            created.append(self)

        def close(self):
            self.closed += 1

    def inspect(original, queries, output, *, worker_factory, execution_mode, query_concurrency, **kwargs):
        assert execution_mode == 'throughput' and query_concurrency == 2
        workers = [worker_factory() for _ in range(query_concurrency)]
        assert all(worker is original for worker in workers)
        assert all(worker.gpu_index is original.gpu_index for worker in workers)
        # Exercise CLI cleanup when the evaluator fails after workers were created.
        raise RuntimeError('evaluation failed')

    monkeypatch.setattr(target, IndexedRetriever)
    monkeypatch.setattr('decision_rag.evaluation.evaluate', inspect)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        cli.main(command(tmp_path, '--method', method))
    assert len(created) == 1
    assert created[0].closed == 1


@pytest.mark.parametrize('flags', [
    ['--backend', 'hf'], ['--backend', 'head'],
    ['--method', 'dense', '--engine', 'hf'], ['--method', 'dense-rerank', '--engine', 'hf'],
])
def test_cli_rejects_unsafe_local_throughput_before_loading(monkeypatch, tmp_path, flags):
    def unexpected_load(*args, **kwargs):
        pytest.fail('Unsupported concurrent backend must fail before loading corpus or model')
    monkeypatch.setattr('decision_rag.cli.Corpus.load', unexpected_load)
    monkeypatch.setattr(cli, 'backend_from_args', unexpected_load)
    with pytest.raises(ValueError, match='Concurrent .* evaluation requires'):
        cli.main(command(tmp_path, *flags))


@pytest.mark.parametrize('failure_point', ['factory', 'inference'])
def test_cli_closes_all_decision_workers_on_failure(monkeypatch, tmp_path, failure_point):
    created = []
    overlap = threading.Barrier(2)

    class FailingSmoke(SmokeBackend):
        def __init__(self):
            if failure_point == 'factory' and len(created) == 2:
                raise RuntimeError('worker initialization failed')
            self.number, self.closed, self.active, self.first_call = len(created), 0, False, True
            created.append(self)

        def binary(self, *args):
            self.active = True
            try:
                if self.first_call:
                    self.first_call = False
                    overlap.wait(timeout=5)
                if self.number == 0:
                    raise RuntimeError('inference failed')
                return super().binary(*args)
            finally:
                self.active = False

        def close(self):
            assert not any(backend.active for backend in created)
            self.closed += 1

    monkeypatch.setattr('decision_rag.demo.SmokeBackend', FailingSmoke)
    count = 3 if failure_point == 'factory' else 2
    with pytest.raises(RuntimeError, match='worker initialization failed|inference failed'):
        cli.main(command(tmp_path, '--backend', 'smoke', conversations=count))
    assert len(created) == 2
    assert [backend.closed for backend in created] == [1, 1]
    assert not (tmp_path / 'output' / 'summary.json').exists()


@pytest.mark.parametrize('evaluation_fails', [False, True])
def test_primary_close_failure_removes_success_summary_and_preserves_evaluation_error(
        monkeypatch, tmp_path, evaluation_fails):
    closed = []
    summary = tmp_path / 'output' / 'summary.json'

    class BrokenCleanupSmoke(SmokeBackend):
        def binary(self, *args):
            if evaluation_fails:
                raise ValueError('primary evaluation failed')
            return super().binary(*args)

        def close(self):
            closed.append(self)
            assert summary.exists() is not evaluation_fails
            raise RuntimeError('primary cleanup failed')

    monkeypatch.setattr('decision_rag.demo.SmokeBackend', BrokenCleanupSmoke)
    error = ValueError if evaluation_fails else RuntimeError
    message = 'primary evaluation failed' if evaluation_fails else 'primary cleanup failed'
    with pytest.raises(error, match=message):
        cli.main(command(tmp_path, '--backend', 'smoke', conversations=1))
    assert len(closed) == 1
    assert not summary.exists()


def test_cli_vllm_workers_keep_query_salts_stable_across_interleaved_stages(monkeypatch, tmp_path):
    clients, payloads = [], []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [{'no': 10, 'yes': 11}.get(text, 100)]

    def reply(request):
        body = json.loads(request.content)
        assert request.url.path == '/v1/completions'
        payloads.append(body)
        rows = [{'index': index, 'logprobs': {'top_logprobs': [
            {f'token_id:{token}': math.log(.5) for token in body['allowed_token_ids']}]}}
                for index in range(len(body['prompt']))]
        return httpx.Response(200, json={'choices': rows, 'usage': {
            'prompt_tokens': sum(map(len, body['prompt'])), 'completion_tokens': len(body['prompt'])}})

    def backend_factory(args):
        client = httpx.Client(transport=httpx.MockTransport(reply))
        clients.append(client)
        backend = VLLMDecisionBackend(tokenizer=Tokenizer(), client=client, verify_server=False)
        backend._owns_client = True
        return backend

    def inspect(original, queries, output, *, worker_factory, **kwargs):
        first, second = worker_factory(), worker_factory()
        try:
            assert first is original and second.backend is not first.backend
            first_stage_done, other_prepared = threading.Event(), threading.Event()

            def first_query():
                first.prepare_query(queries[0])
                first.backend.score_token_prompts([[1]], [10, 11])
                first_stage_done.set()
                assert other_prepared.wait(timeout=5)
                first.backend.score_token_prompts([[2]], [10, 11])

            def second_query():
                assert first_stage_done.wait(timeout=5)
                second.prepare_query(queries[1])
                second.backend.score_token_prompts([[3]], [10, 11])
                other_prepared.set()

            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = [pool.submit(first_query), pool.submit(second_query)]
                for future in pending:
                    future.result(timeout=5)
            salts = {body['prompt'][0][0]: body['cache_salt'] for body in payloads}
            assert salts[1] == salts[2] != salts[3]
            return {'ok': True}
        finally:
            second.close()  # evaluator owns additional workers; CLI owns original

    monkeypatch.setattr(cli, 'backend_from_args', backend_factory)
    monkeypatch.setattr('decision_rag.evaluation.evaluate', inspect)
    try:
        cli.main(command(tmp_path, '--backend', 'vllm'))
        assert len(clients) == 2 and all(client.is_closed for client in clients)
    finally:
        for client in clients:
            client.close()
