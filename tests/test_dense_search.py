"""Exact top-k parity and explicit placement, with CUDA coverage when available."""
import builtins
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from decision_rag.dense_search import ExactDenseIndex, _ieee_matmul, stable_numpy_top_k, stable_torch_top_k


@pytest.mark.parametrize('k', [1, 2, 5, 100])
def test_cpu_partial_selection_matches_stable_full_sort_with_boundary_ties(k):
    vectors = np.array([[1, 0], [0, 1], [1, 0], [.5, .5], [1, 0]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    expected_scores = vectors @ query
    expected_ids = np.argsort(-expected_scores, kind='stable')[:k]
    index = ExactDenseIndex(vectors, device='cpu')
    indices, scores = index.search(query, k)
    assert indices.tolist() == expected_ids.tolist()
    np.testing.assert_allclose(scores, expected_scores[expected_ids])
    assert index.metadata['device'] == 'cpu'
    assert index.metadata['precision'] == 'float32'
    assert index.metadata['index_bytes'] == vectors.nbytes
    index.close()
    index.close()
    with pytest.raises(RuntimeError, match='closed'):
        index.search(query, k)


def test_cpu_search_does_not_import_torch(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            pytest.fail('Explicit CPU search must not import torch')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    index = ExactDenseIndex(np.eye(2, dtype=np.float32), device='cpu')
    assert index.search(np.array([1, 0], dtype=np.float32), 1)[0].tolist() == [0]
    index.close()


@pytest.mark.parametrize('scores,k', [([1., 1., 1., 1.], 2), ([.4, .8, .6, .8, .6, .1], 3),
                                    ([-1., -3., -1.], 3), ([0., 1., 2.], 10)])
def test_torch_topk_tie_policy_matches_numpy_without_cuda(scores, k):
    torch = pytest.importorskip('torch')
    array = np.array(scores, dtype=np.float32)
    assert stable_torch_top_k(torch.from_numpy(array), k).tolist() == stable_numpy_top_k(array, k).tolist()


def test_precision_policy_is_restored_after_failure():
    torch = pytest.importorskip('torch')
    matmul = torch.backends.cuda.matmul
    field = 'fp32_precision' if hasattr(matmul, 'fp32_precision') else 'allow_tf32'
    original = getattr(matmul, field)
    try:
        setattr(matmul, field, 'tf32' if field == 'fp32_precision' else True)
        with pytest.raises(RuntimeError, match='failed'), _ieee_matmul(torch):
            assert getattr(matmul, field) == ('ieee' if field == 'fp32_precision' else False)
            raise RuntimeError('failed')
        assert getattr(matmul, field) == ('tf32' if field == 'fp32_precision' else True)
    finally:
        setattr(matmul, field, original)


@pytest.mark.parametrize('field,original,expected', [('allow_tf32', True, False),
                                                   ('fp32_precision', 'tf32', 'ieee')])
@pytest.mark.parametrize('fail_body', [False, True])
def test_precision_lock_covers_completed_cuda_work_across_threads(field, original, expected, fail_body):
    matmul = SimpleNamespace(**{field: original})
    first_sync, finish_sync, second_attempted, second_entered = (threading.Event() for _ in range(4))
    synced = []

    def synchronize(device):
        assert getattr(matmul, field) == expected
        synced.append(device)
        if len(synced) == 1:
            first_sync.set()
            assert finish_sync.wait(timeout=5)

    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
                            cuda=SimpleNamespace(synchronize=synchronize),
                            autocast=lambda **kwargs: nullcontext())

    def first():
        with _ieee_matmul(torch, device='cuda:0'):
            assert getattr(matmul, field) == expected
            if fail_body:
                raise RuntimeError('search failed after submitting CUDA work')

    def second():
        second_attempted.set()
        with _ieee_matmul(torch, device='cuda:0'):
            second_entered.set()
            assert getattr(matmul, field) == expected

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first)
        try:
            assert first_sync.wait(timeout=5)
            second_result = pool.submit(second)
            assert second_attempted.wait(timeout=5)
            assert not second_entered.wait(timeout=.05)
        finally:
            finish_sync.set()
        if fail_body:
            with pytest.raises(RuntimeError, match='search failed'):
                first_result.result(timeout=5)
        else:
            first_result.result(timeout=5)
        second_result.result(timeout=5)
    assert synced == ['cuda:0', 'cuda:0']
    assert getattr(matmul, field) == original


def test_precision_policy_restored_when_cuda_synchronization_fails():
    matmul = SimpleNamespace(allow_tf32=True)
    def fail(device):
        raise RuntimeError('CUDA execution failed')
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
                            cuda=SimpleNamespace(synchronize=fail),
                            autocast=lambda **kwargs: nullcontext())
    with pytest.raises(RuntimeError, match='CUDA execution failed'), _ieee_matmul(torch, device='cuda:0'):
        assert matmul.allow_tf32 is False
    assert matmul.allow_tf32 is True


@pytest.mark.parametrize('vectors', [np.empty((0, 2)), np.ones(3), np.array([[np.nan, 1]]),
                                    np.array([[np.inf, 1]])])
def test_index_rejects_invalid_vectors(vectors):
    with pytest.raises(ValueError):
        ExactDenseIndex(vectors, device='cpu')


@pytest.mark.parametrize('query', [np.ones(3), np.array([np.nan, 1]), np.array([np.inf, 1])])
def test_search_rejects_invalid_query(query):
    with pytest.raises(ValueError, match='finite vector'):
        ExactDenseIndex(np.eye(2), device='cpu').search(query, 1)


def test_cuda_request_never_silently_falls_back(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        ExactDenseIndex(np.eye(2))


def test_cuda_resident_index_matches_cpu_with_stable_ties():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    # Well separated scores and exact duplicate rows avoid cross-BLAS near-tie ambiguity.
    vectors = np.array([[1, 0], [0, 1], [1, 0], [.5, .5], [1, 0]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    cpu, gpu = ExactDenseIndex(vectors, 'cpu'), ExactDenseIndex(vectors, 'cuda')
    assert gpu._index.device.type == 'cuda'
    assert gpu._index.dtype == torch.float32
    assert gpu.metadata['tf32'] is False
    for k in (1, 2, 3, 5, 100):
        cpu_ids, cpu_scores = cpu.search(query, k)
        gpu_ids, gpu_scores = gpu.search(query, k)
        assert gpu_ids.tolist() == cpu_ids.tolist()
        np.testing.assert_allclose(gpu_scores, cpu_scores, rtol=1e-6, atol=1e-7)
    tensor_ids, tensor_scores = gpu.search(torch.from_numpy(query).to(gpu.device), 2)
    assert tensor_ids.tolist() == [0, 2]
    np.testing.assert_allclose(tensor_scores, [1, 1])
    queries = [query, np.array([0, 1], dtype=np.float32)] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda vector: gpu.search(vector, 3), queries))
    for vector, (indices, scores) in zip(queries, concurrent, strict=True):
        expected_ids, expected_scores = cpu.search(vector, 3)
        assert indices.tolist() == expected_ids.tolist()
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-6, atol=1e-7)
    gpu.close()
    assert gpu._index is None
