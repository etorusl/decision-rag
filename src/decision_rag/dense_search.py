"""Exact resident-vector search with explicit CPU/CUDA placement and FP32 scoring."""
from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager

import numpy as np


def resolve_search_device(device):
    """Resolve logical CUDA numbering after CUDA_VISIBLE_DEVICES; never fall back."""
    device = str(device)
    if device == 'cpu':
        return device
    if not re.fullmatch(r'cuda(?::[0-9]+)?', device):
        raise ValueError('Search device must be cpu, cuda, or cuda:N')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('CUDA search needs PyTorch; use the vLLM environment or explicitly --search-device cpu') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA search requested but CUDA is unavailable; use a CUDA environment or explicitly --search-device cpu')
    index = int(device.split(':')[1]) if ':' in device else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(f'Search device {device} is outside the visible CUDA devices')
    return f'cuda:{index}'


def stable_numpy_top_k(scores, k):
    """Select only k candidates, resolving exact score ties by corpus position."""
    k = min(k, len(scores))
    if k <= 0:
        raise ValueError('top_k must be positive and scores nonempty')
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k - len(higher)]
    selected = np.concatenate((higher, tied))
    return selected[np.argsort(-scores[selected], kind='stable')]


def stable_torch_top_k(scores, k):
    """GPU top-k plus deterministic boundary ties; no full score-vector download."""
    import torch

    k = min(k, scores.numel())
    if k <= 0:
        raise ValueError('top_k must be positive and scores nonempty')
    threshold = torch.topk(scores, k, sorted=True).values[-1]
    # torch.topk itself does not guarantee which tied indices are selected.
    # nonzero yields ascending positions, so a stable sort of <=k values has
    # the same score-descending / position-ascending ordering as the CPU path.
    higher = torch.nonzero(scores > threshold, as_tuple=True)[0]
    tied = torch.nonzero(scores == threshold, as_tuple=True)[0][:k - higher.numel()]
    selected = torch.cat((higher, tied))
    return selected[torch.argsort(scores[selected], descending=True, stable=True)]


@contextmanager
def _ieee_matmul(torch):
    """Use the version-appropriate precision API without changing caller policy."""
    matmul = torch.backends.cuda.matmul
    # New 2.9 controls must not be mixed with the legacy 2.7 allow_tf32 flag.
    field = 'fp32_precision' if hasattr(matmul, 'fp32_precision') else 'allow_tf32'
    previous = getattr(matmul, field)
    setattr(matmul, field, 'ieee' if field == 'fp32_precision' else False)
    try:
        with torch.autocast(device_type='cuda', enabled=False):
            yield
    finally:
        setattr(matmul, field, previous)


class ExactDenseIndex:
    """FP32 exact dot products over pre-normalized vectors, resident on one device.

    Cached embeddings remain portable FP32 NumPy data. Only CUDA construction
    imports PyTorch; initialization copies the index once, before query timing.
    """

    def __init__(self, embeddings, device='cuda'):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or min(embeddings.shape) < 1:
            raise ValueError('Embedding index must be a nonempty two-dimensional array')
        # Bound validation scratch memory for large memory-mapped corpora.
        for start in range(0, len(embeddings), 4096):
            if not np.isfinite(embeddings[start:start + 4096]).all():
                raise ValueError('Embedding index must contain finite vectors')
        self.device = resolve_search_device(device)
        self.vectors, self.dimensions = embeddings.shape
        self._index = None
        self.metadata = {
            'algorithm': 'exact_inner_product', 'selection': 'topk_then_stable_ties',
            'device': self.device, 'backend': 'numpy' if self.device == 'cpu' else 'torch',
            'precision': 'float32', 'tf32': None if self.device == 'cpu' else False,
            'index_bytes': int(embeddings.nbytes), 'vectors': self.vectors, 'dimensions': self.dimensions,
            'index_transfer_seconds': 0.0, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        }
        if self.device == 'cpu':
            self._index = embeddings
            return
        import torch

        self.metadata['gpu_name'] = torch.cuda.get_device_name(self.device)
        started = time.perf_counter()
        try:
            self._index = torch.empty(embeddings.shape, device=self.device, dtype=torch.float32)
            # A bounded writable host copy avoids both a second full host index
            # and torch.from_numpy's warning for a read-only mmap.
            for start in range(0, self.vectors, 4096):
                host = torch.from_numpy(np.array(embeddings[start:start + 4096], copy=True))
                self._index[start:start + len(host)].copy_(host)
            torch.cuda.synchronize(self.device)
        except torch.OutOfMemoryError as exc:
            self._index = None
            raise RuntimeError(self._memory_error('load the index')) from exc
        self.metadata['index_transfer_seconds'] = time.perf_counter() - started

    def _memory_error(self, operation):
        gib = self.metadata['index_bytes'] / 1024**3
        return (f'Insufficient GPU memory to {operation} on {self.device}; FP32 index alone needs {gib:.2f} GiB. '
                'Reduce --pool-memory to leave room beside vLLM, use a smaller corpus, or explicitly '
                'select --search-device cpu for a separately reported CPU run. No CPU fallback was performed.')

    def search(self, query, top_k):
        if self._index is None:
            raise RuntimeError('Search index is closed')
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError('top_k must be a positive integer')
        if self.device == 'cpu':
            query = np.asarray(query, dtype=np.float32)
            if query.shape != (self.dimensions,) or not np.isfinite(query).all():
                raise ValueError('Query must be a finite vector matching index dimensions')
            scores = self._index @ query
            if not np.isfinite(scores).all():
                raise ValueError('Dense scores must be finite')
            ranking = stable_numpy_top_k(scores, top_k)
            return ranking, scores[ranking]
        import torch

        try:
            with torch.inference_mode(), _ieee_matmul(torch):
                if isinstance(query, torch.Tensor):
                    if query.device != self._index.device:
                        raise ValueError('Tensor query must be on the search index device')
                    vector = query.to(dtype=torch.float32)
                else:
                    query = np.asarray(query, dtype=np.float32)
                    if query.shape != (self.dimensions,) or not np.isfinite(query).all():
                        raise ValueError('Query must be a finite vector matching index dimensions')
                    vector = torch.from_numpy(query).to(device=self.device)
                if vector.shape != (self.dimensions,) or not bool(torch.isfinite(vector).all()):
                    raise ValueError('Query must be a finite vector matching index dimensions')
                scores = torch.mv(self._index, vector)
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError('Dense scores must be finite')
                ranking = stable_torch_top_k(scores, top_k)
                # Only the selected k IDs/scores cross back to CPU. Both copies
                # complete before return, so the outer wall timer includes CUDA work.
                selected_scores = scores[ranking].cpu().numpy()
                selected_ids = ranking.cpu().numpy()
                torch.cuda.synchronize(self.device)
                return selected_ids, selected_scores
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(self._memory_error('score a query')) from exc

    def close(self):
        had_index = self._index is not None
        self._index = None
        if had_index and self.device != 'cpu':
            import torch

            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
