"""Ready-model retrieval through local vLLM pooling servers; no training imports.

Verified against vLLM 0.11.2 /v1/embeddings and /score. HTTP timings include the
server's completed work; GPU forward-pass counts are intentionally unavailable.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np

from .retrieval import RetrievalResult

EMBED_QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer it\nQuery: "
RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
# Official vLLM example: examples/offline_inference/pooling/qwen3_reranker.py (v0.11.2).
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _total(values):
    return None if any(value is None for value in values) else sum(values)


def _usage(data: dict, prompt_requests: int) -> dict:
    usage = data.get("usage") or {}
    tokens = usage.get("prompt_tokens")
    if tokens is not None and (not isinstance(tokens, int) or tokens < 0):
        raise ValueError("vLLM returned invalid prompt token usage")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None and (not isinstance(cached, int) or cached < 0):
        raise ValueError("vLLM returned invalid cached token usage")
    return {"requests": 1, "prompt_requests": prompt_requests, "input_tokens": tokens,
            "cached_tokens": cached, "completion_tokens": 0, "forward_passes": None}


def _merge_usage(rows):
    keys = ("requests", "prompt_requests", "input_tokens", "cached_tokens", "completion_tokens")
    return {**{key: _total([row[key] for row in rows]) for key in keys}, "forward_passes": None}


class _PoolingServer:
    def __init__(self, base_url: str, model_name: str, *, revision: str | None = None,
                 timeout: float = 120, client: httpx.Client | None = None):
        base_url = base_url.rstrip("/")
        base_url = base_url.removesuffix("/v1")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("Expected an HTTP server base URL without query/fragment")
        if timeout <= 0:
            raise ValueError("HTTP timeout must be positive")
        self.base_url, self.model_name, self.revision = base_url, model_name, revision
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)
        models = self._request("GET", "/v1/models").get("data")
        if not isinstance(models, list):
            raise ValueError("Invalid /v1/models response")  # noqa: TRY004
        matches = [row for row in models if row.get("id") == model_name]
        if len(matches) != 1:
            raise ValueError(f"Server {base_url} does not advertise requested model {model_name}")
        self.model_card = matches[0]
        version = self._request("GET", "/version").get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("Invalid vLLM /version response")
        self.metadata = {"base_url": base_url, "model": model_name,
                         "model_root": self.model_card.get("root"), "revision": revision,
                         "revision_source": "explicit_server_launch_argument" if revision else "unspecified",
                         "server_version": version,
                         "max_model_len": self.model_card.get("max_model_len")}

    def _request(self, method: str, path: str, payload=None) -> dict:
        response = self.client.request(method, self.base_url + path, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"vLLM {method} {path} failed ({response.status_code}): {response.text[:1000]}") from exc
        value = response.json()
        if not isinstance(value, dict) or "error" in value:
            raise ValueError(f"Invalid/error vLLM response from {path}: {value}")
        returned_model = value.get("model")
        if returned_model is not None and returned_model != self.model_name:
            raise ValueError(f"vLLM returned model {returned_model!r}, expected {self.model_name!r}")
        return value

    def close(self):
        if self._owns_client:
            self.client.close()


def _ordered_data(data: dict, count: int) -> list[dict]:
    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("vLLM returned an unexpected result count")
    indices = [row.get("index") for row in rows]
    if any(type(i) is not int for i in indices) or set(indices) != set(range(count)):
        raise ValueError("vLLM result indices are missing, duplicated or out of range")
    return sorted(rows, key=lambda row: row["index"])


class VLLMDenseRetriever:
    """Normalized exact dense retrieval, with live query encoding on every call."""

    external_inference = True

    def __init__(self, corpus, model_name="Qwen/Qwen3-Embedding-8B", *,
                 base_url="http://127.0.0.1:8001", top_k=20, batch_size=16,
                 index_dir=None, revision=None, timeout=120, client=None,
                 dtype="bfloat16", quantization=None):
        if min(top_k, batch_size) <= 0:
            raise ValueError("top_k and batch_size must be positive")
        if not corpus.passages:
            raise ValueError("Dense retrieval requires passages")
        # Immutable revisions prevent index reuse after same-name model changes.
        if index_dir and (revision is None or not re.fullmatch(r"[0-9a-fA-F]{40}|sha256:[0-9a-fA-F]{64}", revision)):
            raise ValueError("Persistent vLLM index requires an immutable revision: HF commit SHA or sha256:...")
        self.corpus, self.passages = corpus, corpus.passages
        self.top_k, self.batch_size = top_k, batch_size
        self.server = _PoolingServer(base_url, model_name, revision=revision, timeout=timeout, client=client)
        self.model_name = model_name
        self.runtime_metadata = {"engine": "vllm-http", "embedding": self.server.metadata,
                                 "search": "CPU exact dot product of normalized embeddings",
                                 "forward_passes": "not observable through HTTP",
                                 "truncation": "none; overlength requests fail",
                                 "declared_dtype": dtype, "declared_quantization": quantization,
                                 "server_prefix_caching": "not reported by HTTP API"}
        digest = hashlib.sha256()
        for passage in self.passages:
            for value in (passage.id, passage.text):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        self.fingerprint = digest.hexdigest()
        expected = {"format": 1, "engine": "vllm-http", "model": model_name,
                    "revision": revision, "server_version": self.server.metadata["server_version"],
                    "declared_dtype": dtype, "declared_quantization": quantization,
                    "corpus": self.fingerprint, "normalize": True,
                    "query_prefix": self._query_text("")}
        self.indexing_usage = []
        self.embeddings = None
        directory = Path(index_dir) if index_dir else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            metadata_path = directory / "meta.json"
            if metadata_path.exists():
                actual = json.loads(metadata_path.read_text())
                if actual != expected:
                    raise ValueError("vLLM index model/revision/corpus mismatch; use another index directory")
                self.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
                if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.passages):
                    raise ValueError("Cached embedding index shape does not match corpus")
        if self.embeddings is None:
            chunks = []
            partial = directory / "embeddings.partial.npy" if directory else None
            for start in range(0, len(self.passages), batch_size):
                texts = [self._document_text(p.text) for p in self.passages[start:start + batch_size]]
                array, usage = self._embed(texts)
                self.indexing_usage.append(usage)
                if start == 0 and partial:
                    self.embeddings = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                                               shape=(len(self.passages), array.shape[1]))
                if self.embeddings is not None:
                    if array.shape[1] != self.embeddings.shape[1]:
                        raise ValueError("Embedding dimensions changed while building index")
                    self.embeddings[start:start + len(array)] = array
                else:
                    if chunks and array.shape[1] != chunks[0].shape[1]:
                        raise ValueError("Embedding dimensions changed while building index")
                    chunks.append(array)
            if partial:
                self.embeddings.flush()
                del self.embeddings
                partial.replace(directory / "embeddings.npy")
                metadata_path.write_text(json.dumps(expected, indent=2) + "\n")
                self.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            else:
                self.embeddings = np.concatenate(chunks)
        self.runtime_metadata["index_cache_reused"] = not bool(self.indexing_usage)
        self.runtime_metadata["indexing_usage"] = _merge_usage(self.indexing_usage)

    def _document_text(self, text):
        return "passage: " + text if "e5" in self.model_name.lower() else text

    def _query_text(self, text):
        if "qwen3-embedding" in self.model_name.lower():
            return EMBED_QUERY_PREFIX + text
        return "query: " + text if "e5" in self.model_name.lower() else text

    def _embed(self, texts: Sequence[str]):
        data = self.server._request("POST", "/v1/embeddings", {
            "model": self.model_name, "input": list(texts), "encoding_format": "float"})
        rows = _ordered_data(data, len(texts))
        try:
            with np.errstate(over="ignore"):
                array = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid embedding vectors from vLLM") from exc
        if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
            raise ValueError("vLLM embeddings must be finite vectors of a consistent dimension")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("vLLM returned a zero/invalid embedding vector")
        return array / norms, _usage(data, len(texts))

    def reset(self):
        # No answer cache or conversational state exists in this retriever.
        pass

    def close(self):
        self.server.close()

    def retrieve(self, query):
        start = time.perf_counter()
        vectors, usage = self._embed([self._query_text(query.context())])
        if vectors.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension differs from the corpus index")
        scores = self.embeddings @ vectors[0]
        ranking = np.argsort(-scores, kind="stable")[:self.top_k]
        elapsed = time.perf_counter() - start
        stage = {"stage": "dense", "seconds": elapsed, "candidates": len(self.passages), **usage}
        return RetrievalResult(query.id, [self.passages[i].id for i in ranking], scores[ranking].tolist(),
                               elapsed, [stage])


class VLLMQwenReranker:
    """Official Qwen3 pair prompt + /score on a converted pooling server."""

    def __init__(self, model_name="Qwen/Qwen3-Reranker-8B", *,
                 base_url="http://127.0.0.1:8002", batch_size=16, revision=None,
                 timeout=120, client=None, instruction=RERANK_INSTRUCTION):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.server = _PoolingServer(base_url, model_name, revision=revision, timeout=timeout, client=client)
        self.batch_size, self.model_name, self.instruction = batch_size, model_name, instruction

    def score(self, query, documents):
        rows, usages = [], []
        first = f"{RERANK_PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n"
        for start in range(0, len(documents), self.batch_size):
            docs = documents[start:start + self.batch_size]
            data = self.server._request("POST", "/score", {
                "model": self.model_name, "text_1": first,
                "text_2": [f"<Document>: {doc}{RERANK_SUFFIX}" for doc in docs],
                "use_activation": True})
            result = _ordered_data(data, len(docs))
            values = [row.get("score") for row in result]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                   for value in values):
                raise ValueError("vLLM Qwen3 /score must return finite probabilities in [0,1]")
            rows.extend(values)
            usages.append(_usage(data, len(docs)))
        return rows, {**_merge_usage(usages), "truncation_policy": "error (server context limit)",
                      "truncated_candidates": 0}

    def close(self):
        self.server.close()


class VLLMDenseRerankRetriever:
    external_inference = True

    def __init__(self, corpus, embedding_model="Qwen/Qwen3-Embedding-8B",
                 reranker_model="Qwen/Qwen3-Reranker-8B", *,
                 embedding_base_url="http://127.0.0.1:8001", reranker_base_url="http://127.0.0.1:8002",
                 top_k=20, candidates=100, batch_size=16, index_dir=None,
                 embedding_revision=None, reranker_revision=None, timeout=120,
                 embedding_client=None, reranker_client=None, dtype="bfloat16", quantization=None):
        if top_k <= 0 or candidates < top_k:
            raise ValueError("Require 0 < top_k <= candidates")
        self.corpus, self.top_k = corpus, top_k
        self.dense = VLLMDenseRetriever(corpus, embedding_model, base_url=embedding_base_url,
                                       top_k=candidates, batch_size=batch_size, index_dir=index_dir,
                                       revision=embedding_revision, timeout=timeout, client=embedding_client,
                                       dtype=dtype, quantization=quantization)
        self.reranker = VLLMQwenReranker(reranker_model, base_url=reranker_base_url,
                                        batch_size=batch_size, revision=reranker_revision,
                                        timeout=timeout, client=reranker_client)
        self.runtime_metadata = {**self.dense.runtime_metadata, "reranker": self.reranker.server.metadata,
                                 "timing_scope": "live query embedding + exact search + live reranking"}

    def reset(self):
        self.dense.reset()

    def close(self):
        self.dense.close()
        self.reranker.close()

    def retrieve(self, query):
        start = time.perf_counter()
        candidates = self.dense.retrieve(query)
        rerank_start = time.perf_counter()
        scores, usage = self.reranker.score(query.context(), [self.corpus.nodes[pid].text for pid in candidates.ids])
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:self.top_k]
        stages = candidates.stages + [{"stage": "rerank", "seconds": time.perf_counter() - rerank_start,
                                      "candidates": len(candidates.ids), **usage}]
        return RetrievalResult(query.id, [candidates.ids[i] for i in ranking], [scores[i] for i in ranking],
                               time.perf_counter() - start, stages)
