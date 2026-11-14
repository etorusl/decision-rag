"""Exact dense search followed by the official Qwen3 pointwise reranker prompt.

Timing includes query embedding, corpus search, tokenization and all reranker batches.
No full-vocabulary logits or autoregressive generation are needed for yes/no scoring.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .backends.decision_head import resolve_dtype
from .baselines import DenseRetriever
from .evaluation import synchronize
from .retrieval import RetrievalResult

# Published prompt: https://huggingface.co/Qwen/Qwen3-Reranker-8B#using-transformers
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class QwenReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 16,
        max_length: int = 8192,
        max_batch_tokens: int = 32768,
        truncation: str = "document_tail",
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if min(batch_size, max_length, max_batch_tokens) <= 0:
            raise ValueError("Reranker token/batch limits must be positive")
        if max_length > max_batch_tokens:
            raise ValueError("max_batch_tokens must accommodate one max_length sequence")
        if truncation not in {"document_tail", "error"}:
            raise ValueError("truncation must be document_tail or error")
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.max_batch_tokens, self.truncation, self.instruction = max_batch_tokens, truncation, instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=resolve_dtype(dtype),
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        if self.model.config.model_type != "qwen3":
            raise ValueError("QwenReranker requires a Qwen3 model")
        if max_length > self.model.config.max_position_embeddings:
            raise ValueError("Reranker max_length exceeds the model's native context")
        self.prefix_ids = self.tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
        self.label_ids = [self.tokenizer.convert_tokens_to_ids(label) for label in ("no", "yes")]
        if (
            None in self.label_ids
            or len(set(self.label_ids)) != 2
            or self.tokenizer.unk_token_id in self.label_ids
        ):
            raise ValueError("Reranker requires distinct vocabulary tokens for no/yes")
        if len(self.prefix_ids) + len(self.suffix_ids) >= max_length:
            raise ValueError("max_length is too small for the official reranker framing")
        self.name = model_name

    def close(self):
        self.model = None
        if str(self.device).startswith('cuda'):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _encode(self, query: str, documents: Sequence[str]):
        header = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: "
        available = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        # Never discard part of the query/instruction while silently keeping no document.
        if len(self.tokenizer.encode(header, add_special_tokens=False)) >= available:
            raise ValueError("Query/instruction leaves no reranker document budget; increase max_length")
        rows, truncated, removed = [], 0, 0
        for document in documents:
            body = self.tokenizer.encode(header + document, add_special_tokens=False)
            if len(body) > available:
                if self.truncation == "error":
                    raise ValueError("Reranker pair exceeds max_length with truncation=error")
                truncated += 1
                removed += len(body) - available
                body = body[:available]
            rows.append(self.prefix_ids + body + self.suffix_ids)
        return rows, truncated, removed

    @torch.inference_mode()
    def score(self, query: str, documents: Sequence[str]) -> tuple[list[float], dict]:
        if not documents:
            return [], {
                "input_tokens": 0,
                "padded_tokens": 0,
                "forward_passes": 0,
                "truncated_candidates": 0,
                "truncated_tokens": 0,
                "requests": 0,
            }
        rows, truncated, removed = self._encode(query, documents)
        # Sorting only changes batching. Returned scores retain original dense-candidate order.
        order = sorted(range(len(rows)), key=lambda i: (len(rows[i]), i))
        batches, current = [], []
        for index in order:
            padded = len(rows[index]) * (len(current) + 1)
            if current and (len(current) >= self.batch_size or padded > self.max_batch_tokens):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        scores = [0.0] * len(rows)
        padded_tokens = 0
        head = self.model.get_output_embeddings()
        label_ids = torch.tensor(self.label_ids, device=head.weight.device)
        weights = head.weight.index_select(0, label_ids)
        bias = head.bias.index_select(0, label_ids) if getattr(head, "bias", None) is not None else None
        for batch in batches:
            inputs = self.tokenizer.pad(
                {"input_ids": [rows[i] for i in batch]},
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            }
            inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            padded_tokens += inputs["input_ids"].numel()
            hidden = self.model.base_model(**inputs, use_cache=False).last_hidden_state[:, -1]
            probabilities = F.linear(hidden, weights, bias).float().softmax(-1)[:, 1].tolist()
            for index, probability in zip(batch, probabilities, strict=True):
                scores[index] = probability
        return scores, {
            "input_tokens": sum(map(len, rows)),
            "padded_tokens": padded_tokens,
            "forward_passes": len(batches),
            "requests": len(batches),
            "truncated_candidates": truncated,
            "truncated_tokens": removed,
            "max_length": self.max_length,
            "truncation_policy": self.truncation,
        }


class DenseRerankRetriever:
    def __init__(
        self,
        corpus,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
        top_k: int = 20,
        candidates: int = 100,
        device: str = "cuda",
        batch_size: int = 16,
        index_dir=None,
        *,
        dtype: str = "bfloat16",
        max_length: int = 8192,
        max_batch_tokens: int = 32768,
        truncation: str = "document_tail",
        instruction: str = DEFAULT_INSTRUCTION,
        search_device: str = 'cuda',
    ):
        if top_k <= 0 or candidates < top_k:
            raise ValueError("Require 0 < top_k <= dense candidate count")
        self.corpus, self.top_k = corpus, top_k
        self.dense = DenseRetriever(corpus, embedding_model, candidates, device, batch_size, index_dir,
                                    dtype=dtype, search_device=search_device)
        try:
            self.reranker = QwenReranker(
                reranker_model,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                max_length=max_length,
                max_batch_tokens=max_batch_tokens,
                truncation=truncation,
                instruction=instruction,
            )
        except BaseException:
            self.dense.close()
            raise
        self.local_search_device = self.dense.local_search_device
        self.runtime_metadata = {**self.dense.runtime_metadata, 'reranker_model': reranker_model,
                                 'reranker_device': device, 'reranker_dtype': dtype}

    def reset(self):
        self.dense.reset()

    def close(self):
        try:
            self.dense.close()
        finally:
            self.reranker.close()

    def retrieve(self, query) -> RetrievalResult:
        synchronize()
        start = time.perf_counter()
        dense = self.dense.retrieve(query)
        synchronize()
        dense_elapsed = time.perf_counter() - start
        rerank_start = time.perf_counter()
        scores, stats = self.reranker.score(
            query.context(), [self.corpus.nodes[pid].text for pid in dense.ids]
        )
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[: self.top_k]
        synchronize()
        elapsed = time.perf_counter() - start
        stages = (dense.stages or [
            {"stage": "dense", "candidates": len(dense.ids), "latency_seconds": dense_elapsed}
        ]) + [
            {
                "stage": "pointwise_rerank",
                "candidates": len(dense.ids),
                "latency_seconds": time.perf_counter() - rerank_start,
                **stats,
            },
        ]
        return RetrievalResult(
            query.id, [dense.ids[i] for i in order], [scores[i] for i in order], elapsed, stages
        )
