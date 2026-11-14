"""GPU-first BM25 and exact dense retrieval, sharing evaluation and corpus IDs."""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .dense_search import ExactDenseIndex, resolve_search_device, stable_numpy_top_k, stable_torch_top_k
from .retrieval import RetrievalResult


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, corpus, top_k=20, k1=1.2, b=0.75, *, device='cuda'):
        if top_k <= 0 or not corpus.passages:
            raise ValueError('BM25 requires passages and positive top_k')
        self.local_search_device = resolve_search_device(device)
        self._closed = False
        self.corpus, self.top_k, self.k1, self.b = corpus, top_k, k1, b
        self.passages = corpus.passages
        self.postings = defaultdict(list)
        lengths = []
        for i, p in enumerate(self.passages):
            page = corpus.page_id(p.id)
            title = corpus.nodes[page].title if page else p.title
            counts = Counter(tokenize(title + "\n" + p.text))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.lengths = np.asarray(lengths)
        self.avgdl = float(self.lengths.mean()) if lengths else 1
        self.avgdl = self.avgdl or 1

        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        n = len(self.passages)
        self.weights = {}
        for token, rows in self.postings.items():
            indices = np.array([i for i, _ in rows], dtype=np.int32)
            frequencies = np.array([tf for _, tf in rows], dtype=np.float32)
            idf = math.log(1 + (n - len(rows) + .5) / (len(rows) + .5))
            self.weights[token] = (indices, idf * frequencies * (self.k1 + 1) / (frequencies + norm[indices]))
        del self.postings
        self._ids = self._weights = None
        self.offsets = {}
        index_bytes = sum(ids.nbytes + values.nbytes for ids, values in self.weights.values())
        transfer_seconds = 0.0
        if self.local_search_device != 'cpu':
            import torch
            offset = 0
            for term, (ids, values) in self.weights.items():
                self.offsets[term] = (offset, offset + len(ids))
                offset += len(ids)
            ids = np.concatenate([ids for ids, _ in self.weights.values()]) if self.weights else np.empty(0)
            values = np.concatenate([values for _, values in self.weights.values()]) if self.weights else np.empty(0)
            transfer_start = time.perf_counter()
            try:
                self._ids = torch.tensor(ids, dtype=torch.int64, device=self.local_search_device)
                self._weights = torch.tensor(values, dtype=torch.float32, device=self.local_search_device)
                torch.cuda.synchronize(self.local_search_device)
            except torch.cuda.OutOfMemoryError as exc:
                self._ids = self._weights = None
                raise RuntimeError('BM25 index does not fit on the selected GPU; free GPU memory before retrying') from exc
            index_bytes = self._ids.numel() * self._ids.element_size() + self._weights.numel() * self._weights.element_size()
            transfer_seconds = time.perf_counter() - transfer_start
            self.weights.clear()
        self.runtime_metadata = {
            'engine': 'bm25',
            'search_runtime': {'device': self.local_search_device, 'algorithm': 'precomputed BM25 postings',
                               'precision': 'float64' if self.local_search_device == 'cpu' else 'float32',
                               'backend': 'numpy' if self.local_search_device == 'cpu' else 'torch',
                               'index_bytes': index_bytes, 'index_transfer_seconds': transfer_seconds},
        }

    def reset(self):
        pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._ids = self._weights = None
        self.weights.clear()
        self.offsets.clear()
        if self.local_search_device != 'cpu':
            import torch
            with torch.cuda.device(self.local_search_device):
                torch.cuda.empty_cache()

    def retrieve(self, query):
        if self._closed:
            raise RuntimeError('BM25 index is closed')
        start = time.perf_counter()
        n = len(self.passages)
        tokens = sorted(set(tokenize(query.context())))
        if self.local_search_device == 'cpu':
            scores = np.zeros(n)
            for token in tokens:
                posting = self.weights.get(token)
                if posting is not None:
                    ids, weights = posting
                    scores[ids] += weights
            indices = stable_numpy_top_k(scores, self.top_k)
            selected_scores = scores[indices].tolist()
        else:
            import torch
            with torch.inference_mode():
                scores = torch.zeros(n, device=self.local_search_device, dtype=torch.float32)
                for token in tokens:
                    bounds = self.offsets.get(token)
                    if bounds:
                        begin, end = bounds
                        scores.index_add_(0, self._ids[begin:end], self._weights[begin:end])
                ranking = stable_torch_top_k(scores, self.top_k)
                indices = ranking.cpu().numpy()
                selected_scores = scores[ranking].cpu().tolist()
        elapsed = time.perf_counter() - start
        return RetrievalResult(query.id, [self.passages[i].id for i in indices],
                               selected_scores, elapsed,
                               [{'stage': 'bm25', 'seconds': elapsed, 'candidates': n,
                                 'search_device': self.local_search_device}])


class DenseRetriever:
    def __init__(self, corpus, model_name="Qwen/Qwen3-Embedding-0.6B", top_k=20,
                 device="cuda", batch_size=16, index_dir=None, dtype="bfloat16", *, search_device="cuda"):
        import hashlib
        import json
        from pathlib import Path

        from sentence_transformers import SentenceTransformer
        if min(top_k, batch_size) <= 0 or not corpus.passages:
            raise ValueError('Dense retrieval requires passages and positive top_k/batch_size')
        self.local_search_device = resolve_search_device(search_device)
        self.corpus, self.top_k, self.passages = corpus, top_k, corpus.passages
        self.model_name, self.device = model_name, device
        import torch

        from .backends.decision_head import resolve_dtype
        precision = resolve_dtype(dtype) if str(device).startswith("cuda") else torch.float32
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=False,
                                        model_kwargs={"torch_dtype": precision, "attn_implementation": "sdpa"})
        self.dtype = str(precision)
        self.revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        digest = hashlib.sha256()
        for passage in self.passages:
            for value in (passage.id, passage.text):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        self.fingerprint = digest.hexdigest()
        expected_meta = {"model": model_name, "revision": self.revision, "dtype": self.dtype,
                         "corpus": self.fingerprint}
        self.embeddings = None
        if index_dir:
            index_dir = Path(index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
            meta = index_dir / 'meta.json'
            if meta.exists():
                actual = json.loads(meta.read_text())
                if actual != expected_meta:
                    raise ValueError("Dense index model/corpus mismatch; use a new index directory")
                self.embeddings = np.load(index_dir / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
                if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.passages):
                    raise ValueError('Dense cached index shape does not match corpus')
        if self.embeddings is None:
            texts = [p.text for p in self.passages]
            if "e5" in model_name.lower():
                texts = ["passage: " + t for t in texts]
            self.embeddings = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=True)
            if index_dir:
                np.save(index_dir / 'embeddings.npy', self.embeddings)
                meta.write_text(json.dumps(expected_meta))
        try:
            self.search_index = ExactDenseIndex(self.embeddings, device=self.local_search_device)
        except BaseException:
            self.model = None
            raise
        self.runtime_metadata = {
            'engine': 'sentence-transformers', 'embedding_model': self.model_name,
            'embedding_revision': self.revision, 'embedding_dtype': self.dtype,
            'embedding_device': str(getattr(self.model, 'device', self.device)),
            'search_runtime': self.search_index.metadata,
        }

    def reset(self):
        pass

    def close(self):
        self.model = None
        self.search_index.close()

    def _synchronize_embedding(self):
        import torch

        device = getattr(self.model, 'device', self.device)
        if str(device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def retrieve(self, query):
        self._synchronize_embedding()
        start = time.perf_counter()
        text = query.context()
        kwargs = {}
        if "qwen3-embedding" in self.model_name.lower():
            kwargs['prompt'] = "Instruct: Given a question, retrieve passages that answer it\nQuery: "
        elif "e5" in self.model_name.lower():
            text = "query: " + text
        gpu_search = self.local_search_device != 'cpu'
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=not gpu_search,
                                   convert_to_tensor=gpu_search, **kwargs)[0]
        self._synchronize_embedding()
        embedding_seconds = time.perf_counter() - start
        search_start = time.perf_counter()
        if gpu_search:
            vector = vector.to(self.local_search_device)
        indices, scores = self.search_index.search(vector, self.top_k)
        search_seconds = time.perf_counter() - search_start
        # ExactDenseIndex synchronizes its own search device before returning.
        elapsed = time.perf_counter() - start
        return RetrievalResult(query.id, [self.passages[i].id for i in indices], scores.tolist(), elapsed,
                               [{'stage': 'dense', 'seconds': elapsed, 'latency_seconds': elapsed,
                                 'query_embedding_seconds': embedding_seconds, 'vector_search_seconds': search_seconds,
                                 'candidates': len(self.passages), 'search_device': self.local_search_device,
                                 'forward_passes': None, 'input_tokens': None}])
