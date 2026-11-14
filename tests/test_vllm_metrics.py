import json

import pytest

from decision_rag.backends.base import DecisionOutput
from decision_rag.demo import SmokeBackend, write_demo
from decision_rag.evaluation import evaluate
from decision_rag.retrieval import DecisionRetriever, RetrievalConfig
from decision_rag.schema import Corpus, load_queries
from decision_rag.telemetry import GPUMonitor


class ExternalBackend(SmokeBackend):
    external_inference = True

    def __init__(self):
        self.prepared = 0

    def prepare_query(self, query):
        self.prepared += 1

    def binary(self, context, questions):
        output = super().binary(context, questions)
        return DecisionOutput(output.probabilities, 100, None,
                              {'requests': 2, 'prompt_requests': len(questions),
                               'completion_tokens': len(questions), 'cached_tokens': None})


def test_external_inference_preserves_unknown_work_and_prepares_each_query(tmp_path, monkeypatch):
    write_demo(tmp_path / 'data')
    backend = ExternalBackend()
    retriever = DecisionRetriever(Corpus.load(tmp_path / 'data/corpus.jsonl'), backend,
                                  RetrievalConfig(budgets=[1]))
    def forbidden():
        pytest.fail('Client CUDA synchronization is unrelated to external vLLM work')
    monkeypatch.setattr('decision_rag.evaluation.synchronize', forbidden)
    summary = evaluate(retriever, load_queries(tmp_path / 'data/queries.jsonl'), tmp_path / 'run', warmup=1)
    assert backend.prepared == 2  # one warmup + one measured query; independent cache salts
    assert summary['work']['forward_passes'] is None
    assert summary['work']['cached_tokens'] is None
    assert summary['work']['requests'] == 4
    assert summary['work']['input_tokens'] == 200
    assert summary['inference_process'] == 'external_vllm_server'
    assert 'gpu_peak_allocated_bytes' not in summary
    assert json.loads((tmp_path / 'run/queries.jsonl').read_text())['stages'][0]['forward_passes'] is None


def test_device_memory_sampling_includes_begin_and_end(monkeypatch):
    values = iter(['32000, 20\n', '34000, 80\n'])
    monkeypatch.setattr('decision_rag.telemetry.subprocess.check_output', lambda *a, **k: next(values))
    with GPUMonitor('0', interval=60) as monitor:
        pass
    summary = monitor.summary()
    assert summary['samples'] == 2
    assert summary['peak_device_used_bytes'] == 34000 * 2**20
    assert summary['mean_utilization_percent'] == 50
    assert 'entire GPU' in summary['scope']


def test_device_memory_sampling_failure_is_explicit(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError('nvidia-smi unavailable')
    monkeypatch.setattr('decision_rag.telemetry.subprocess.check_output', missing)
    with GPUMonitor('0') as monitor:
        pass
    assert monitor.summary()['peak_device_used_bytes'] is None
    assert 'unavailable' in monitor.summary()['error']
