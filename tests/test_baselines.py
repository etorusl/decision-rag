import math
from types import SimpleNamespace

import numpy as np
import pytest

from decision_rag.baselines import BM25Retriever, DenseRetriever
from decision_rag.schema import Corpus, Node, Query


def small_corpus():
    return Corpus([Node('root','','root'), Node('page','','page','root'),
                   Node('a','apple apple','passage','page',order=0),
                   Node('b','banana','passage','page',order=1)])


def test_bm25_precomputed_postings_match_formula():
    result = BM25Retriever(small_corpus(), device='cpu').retrieve(Query('q','apple'))
    expected = math.log(1 + 1.5/1.5) * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 2/1.5))
    assert result.ids == ['a','b']
    assert result.scores == pytest.approx([expected,0])


def test_dense_precision_cache_and_query_instruction(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    st = pytest.importorskip('sentence_transformers')
    calls = []
    class Encoder:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
        def __getitem__(self, index):
            return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash='revision')))
        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return np.array([[1.,0.] if 'apple' in text else [0.,1.] for text in texts])
    monkeypatch.setattr(st, 'SentenceTransformer', Encoder)
    dense = DenseRetriever(small_corpus(), device='cuda', index_dir=tmp_path, search_device='cpu')
    assert calls[0]['model_kwargs']['torch_dtype'] == torch.bfloat16
    result = dense.retrieve(Query('q','apple'))
    assert result.ids[0] == 'a'
    assert dense.runtime_metadata['search_runtime']['device'] == 'cpu'
    assert result.stages[0]['query_embedding_seconds'] >= 0
    assert result.stages[0]['vector_search_seconds'] >= 0
    assert result.stages[0]['search_device'] == 'cpu'
    assert calls[-1]['prompt'].startswith('Instruct:')
    with pytest.raises(ValueError, match='mismatch'):
        DenseRetriever(small_corpus(),device='cuda',dtype='float32',index_dir=tmp_path, search_device='cpu')
    dense.close()
    assert dense.model is None
    with pytest.raises(RuntimeError, match='closed'):
        dense.search_index.search(np.array([1, 0]), 1)


def test_bm25_ties_and_close_are_explicit():
    retriever = BM25Retriever(small_corpus(), device='cpu')
    result = retriever.retrieve(Query('q', 'unknown'))
    assert result.ids == ['a', 'b']
    assert result.scores == [0, 0]
    assert retriever.runtime_metadata['search_runtime']['device'] == 'cpu'
    retriever.close()
    retriever.close()
    with pytest.raises(RuntimeError, match='closed'):
        retriever.retrieve(Query('q', 'apple'))


@pytest.mark.parametrize('reported_device,requested_device,expected_device', [
    ('cuda:1', 'cuda:0', 'cuda:1'), (None, 'cuda:2', 'cuda:2')])
def test_dense_times_actual_encoder_device_without_global_cuda_sync(
        monkeypatch, reported_device, requested_device, expected_device):
    torch = pytest.importorskip('torch')
    st = pytest.importorskip('sentence_transformers')
    clock, pending, calls = [0.0], [0.0], []

    class Encoder:
        def __init__(self, *args, **kwargs):
            if reported_device is not None:
                self.device = torch.device(reported_device)

        def __getitem__(self, index):
            return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash='revision')))

        def encode(self, texts, **kwargs):
            calls.append('encode')
            pending[0] += .25
            return np.array([[1., 0.] for text in texts], dtype=np.float32)

    def synchronize(device=None):
        calls.append(str(device))
        if str(device) == expected_device:
            clock[0] += pending[0]
            pending[0] = 0.0

    monkeypatch.setattr(st, 'SentenceTransformer', Encoder)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    monkeypatch.setattr('decision_rag.baselines.time.perf_counter', lambda: clock[0])
    retriever = DenseRetriever(small_corpus(), device=requested_device, search_device='cpu')
    pending[0] = 10.0  # Outstanding initialization work must stay outside query timing.
    calls.clear()
    result = retriever.retrieve(Query('q', 'apple'))
    assert calls == [expected_device, 'encode', expected_device]
    assert result.stages[0]['query_embedding_seconds'] == .25
    assert result.latency_seconds == .25
    assert retriever.runtime_metadata['embedding_device'] == expected_device
    retriever.close()


def test_gpu_search_defaults_never_fall_back_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        BM25Retriever(small_corpus())
    pytest.importorskip('sentence_transformers')
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        DenseRetriever(small_corpus())


def test_bm25_cuda_matches_cpu_scores_and_stable_ties():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for GPU BM25 parity')
    gpu = BM25Retriever(small_corpus())
    cpu = BM25Retriever(small_corpus(), device='cpu')
    try:
        for question in ('apple', 'banana', 'apple banana', 'unknown'):
            query = Query('q', question)
            actual, expected = gpu.retrieve(query), cpu.retrieve(query)
            assert actual.ids == expected.ids
            assert actual.scores == pytest.approx(expected.scores, rel=1e-6, abs=1e-7)
        assert gpu.runtime_metadata['search_runtime']['device'].startswith('cuda:')
    finally:
        gpu.close()
        cpu.close()
