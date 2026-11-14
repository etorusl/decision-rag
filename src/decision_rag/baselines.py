"""CPU BM25 and optional exact dense retrieval, sharing evaluation and corpus IDs."""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .retrieval import RetrievalResult


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, corpus, top_k=20, k1=1.2, b=0.75):
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

    def reset(self):
        pass

    def retrieve(self, query):
        start = time.perf_counter()
        n = len(self.passages)
        scores = np.zeros(n)
        for token in set(tokenize(query.context())):
            postings = self.postings.get(token, [])
            idf = math.log(1 + (n - len(postings) + .5) / (len(postings) + .5))
            for idx, tf in postings:
                scores[idx] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avgdl))
        indices = np.argsort(-scores, kind="stable")[:self.top_k]
        return RetrievalResult(query.id, [self.passages[i].id for i in indices],
                               scores[indices].tolist(), time.perf_counter() - start, [])


class DenseRetriever:
    def __init__(self, corpus, model_name="Qwen/Qwen3-Embedding-0.6B", top_k=20,
                 device="cuda", batch_size=16, index_dir=None):
        import hashlib
        import json
        from pathlib import Path

        from sentence_transformers import SentenceTransformer
        self.corpus, self.top_k, self.passages = corpus, top_k, corpus.passages
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=False)
        self.fingerprint = hashlib.sha256(json.dumps([(p.id, p.text) for p in self.passages],
                                                    ensure_ascii=False).encode()).hexdigest()
        self.embeddings = None
        if index_dir:
            index_dir = Path(index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
            meta = index_dir / 'meta.json'
            if meta.exists():
                actual = json.loads(meta.read_text())
                if actual != {"model": model_name, "corpus": self.fingerprint}:
                    raise ValueError("Dense index model/corpus mismatch; use a new index directory")
                self.embeddings = np.load(index_dir / 'embeddings.npy', mmap_mode='r')
        if self.embeddings is None:
            texts = [p.text for p in self.passages]
            if "e5" in model_name.lower():
                texts = ["passage: " + t for t in texts]
            self.embeddings = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=True)
            if index_dir:
                np.save(index_dir / 'embeddings.npy', self.embeddings)
                meta.write_text(json.dumps({"model": model_name, "corpus": self.fingerprint}))

    def reset(self):
        pass

    def retrieve(self, query):
        from .evaluation import synchronize
        synchronize()
        start = time.perf_counter()
        text = query.context()
        kwargs = {}
        if "qwen3-embedding" in self.model_name.lower():
            kwargs['prompt'] = "Instruct: Given a question, retrieve passages that answer it\nQuery: "
        elif "e5" in self.model_name.lower():
            text = "query: " + text
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True, **kwargs)[0]
        # CPU exact search is included in timings; no unmeasured saved candidates.
        scores = self.embeddings @ vector
        indices = np.argsort(-scores, kind="stable")[:self.top_k]
        synchronize()
        return RetrievalResult(query.id, [self.passages[i].id for i in indices], scores[indices].tolist(),
                               time.perf_counter() - start, [])
