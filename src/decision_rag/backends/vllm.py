"""One-token decisions through a local vLLM 0.11.2-compatible HTTP server.

The server batches independent requests and may reuse their common KV prefix. This
is inference of an existing causal LM, not the custom packed decision-head runtime.
"""
from __future__ import annotations

import math
import string
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .base import DecisionOutput


def server_root(base_url: str) -> str:
    """Accept either the server root or its /v1 OpenAI-compatible base URL."""
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("vLLM base_url must be an HTTP(S) server URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Pass authentication separately; base_url cannot contain credentials/query/fragment")
    path = parts.path.rstrip("/")
    path = path.removesuffix("/v1")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class VLLMDecisionBackend:
    """Score yes/no or choice labels on an already running local model server.

    At most one answer token is produced per candidate; only its label logprobs
    determine scores. Physical GPU forwards are not observable over this API.
    """

    thread_safe = True
    external_inference = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        *,
        base_url: str = "http://127.0.0.1:8000",
        served_model_name: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        max_context_tokens: int = 32768,
        batch_size: int = 64,
        concurrency: int = 1,
        timeout: float = 120,
        api_key: str | None = None,
        cache_policy: str = "per-query",
        prime_prefix: bool = True,
        verify_server: bool = True,
        tokenizer: Any = None,
        client: httpx.Client | None = None,
    ):
        if min(max_context_tokens, batch_size, concurrency) < 1 or timeout <= 0:
            raise ValueError("vLLM token, batch, concurrency and timeout limits must be positive")
        if cache_policy not in {"per-query", "warm"}:
            raise ValueError("cache_policy must be per-query or warm")
        self.base_url = server_root(base_url)
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.max_context_tokens = max_context_tokens
        self.batch_size, self.concurrency = batch_size, concurrency
        self.cache_policy, self.prime_prefix = cache_policy, prime_prefix
        self._cache_salt = uuid.uuid4().hex if cache_policy == "per-query" else None
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout, trust_env=False,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            limits=httpx.Limits(max_connections=max(concurrency, 4), max_keepalive_connections=max(concurrency, 4)),
        )
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name or model_name, revision=tokenizer_revision, trust_remote_code=False,
            )
        self.tokenizer = tokenizer
        self.label_ids = self._single_token_ids(["no", "yes"])
        self.name = f"vllm-logits:{self.served_model_name}"
        self.runtime_metadata = {
            "runtime": "vllm-http", "base_url": self.base_url,
            "model": self.served_model_name, "tokenizer": tokenizer_name or model_name,
            "tokenizer_revision": tokenizer_revision, "cache_policy": cache_policy,
            "prefix_priming": prime_prefix, "http_batch_size": batch_size,
            "http_concurrency": concurrency, "scoring": "one-token-restricted-logprobs",
            "physical_forward_passes": "not observable from HTTP responses",
            "server_verified": False,
        }
        try:
            if verify_server:
                self._verify_server()
        except Exception:
            self.close()
            raise

    def _request(self, method: str, path: str, *, payload=None, params=None) -> dict:
        try:
            response = self.client.request(method, self.base_url + path, json=payload, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Local vLLM {method} {path} failed: {exc}. Check the server and model; no HF fallback is used."
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise ValueError(f"vLLM {path} returned invalid JSON") from exc
        if not isinstance(body, dict) or body.get("error"):
            raise ValueError(f"vLLM {path} returned an error or invalid response object")
        return body

    def _verify_server(self):
        models = self._request("GET", "/v1/models").get("data", [])
        if not any(isinstance(row, dict) and row.get("id") == self.served_model_name for row in models):
            raise ValueError(f"vLLM does not serve model {self.served_model_name!r}")
        version = self._request("GET", "/version").get("version")
        # Some server builds expose this only in development mode. Model/version
        # and label-token validation remain mandatory; never enable dev mode merely
        # for metadata. Probability responses still validate the scoring contract.
        response = self.client.get(self.base_url + "/server_info", params={"config_format": "json"})
        config = {}
        if response.status_code not in {404, 405}:
            response.raise_for_status()
            raw = response.json()
            candidate_config = raw.get("vllm_config") if isinstance(raw, dict) else None
            if isinstance(candidate_config, dict):
                config = candidate_config
        model, cache = config.get("model_config", {}), config.get("cache_config", {})
        if model and model.get("logprobs_mode") != "processed_logprobs":
            raise ValueError("Start vLLM with --logprobs-mode processed_logprobs to score all allowed labels")
        max_logprobs = model.get("max_logprobs", 26)
        if not isinstance(max_logprobs, int) or max_logprobs != -1 and max_logprobs < 26:
            raise ValueError("Start vLLM with --max-logprobs 26 (binary and up to 26-way choice)")
        if cache and not cache.get("enable_prefix_caching", False):
            raise ValueError("Start vLLM with --enable-prefix-caching for shared-context inference")
        model_card = next(row for row in models if row.get("id") == self.served_model_name)
        server_limit = model.get("max_model_len", model_card.get("max_model_len"))
        if isinstance(server_limit, int) and server_limit > 0:
            self.max_context_tokens = min(self.max_context_tokens, server_limit)
        sample = "no yes A Z\n<|im_start|>user\nDecision retrieval<|im_end|>"
        expected = self.tokenizer.encode(sample, add_special_tokens=False)
        actual = self._request("POST", "/tokenize", payload={
            "model": self.served_model_name, "prompt": sample, "add_special_tokens": False,
        }).get("tokens")
        if actual != expected:
            raise ValueError("Local and served tokenizers differ; set the matching tokenizer name/revision")
        # Never persist the complete server config: it may contain credentials.
        self.runtime_metadata.update({
            "server_verified": True, "server_version": version, "server_config_available": bool(config),
            "model_config": {key: model.get(key) for key in (
                "model", "dtype", "revision", "tokenizer_revision", "quantization", "max_model_len",
                "logprobs_mode", "max_logprobs",
            )},
            "cache_config": {key: cache.get(key) for key in (
                "enable_prefix_caching", "cache_dtype", "block_size", "gpu_memory_utilization",
            )},
            "scheduler_config": {key: config.get("scheduler_config", {}).get(key) for key in (
                "max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill",
            )},
        })

    def prepare_query(self, query=None):
        """Keep APC inside one query while isolating it from warmup/previous queries."""
        if self.cache_policy == "per-query":
            self._cache_salt = uuid.uuid4().hex

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _single_token_ids(self, labels: Sequence[str]) -> list[int]:
        encoded = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(encoded):
            raise ValueError("vLLM decisions need distinct single-token answer labels")
        return [ids[0] for ids in encoded]

    def _chat(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def _binary_prompts(self, context: str, questions: Sequence[str]) -> list[list[int]]:
        # Match the HF reference's BPE boundary: tokenize shared prefix separately.
        marker = "__DECISION_RAG_QUESTION_BOUNDARY_4fc71e__"
        if marker in context or any(marker in question for question in questions):
            raise ValueError("Reserved prompt delimiter found in input")
        prompt = self._chat(
            "Judge relevance using the context and candidates below.\n\n" + context + "\n\n" + marker
            + "\nAnswer with exactly one lowercase word: yes or no."
        )
        prefix, suffix = prompt.split(marker)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        return [prefix_ids + self.tokenizer.encode(q + suffix, add_special_tokens=False) for q in questions]

    @staticmethod
    def _parse_choices(body: dict, count: int, labels: Sequence[int]) -> list[list[float]]:
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != count:
            raise ValueError("vLLM returned the wrong number of candidate decisions")
        result: list[list[float] | None] = [None] * count
        for choice in choices:
            index = choice.get("index")
            if not isinstance(index, int) or not 0 <= index < count or result[index] is not None:
                raise ValueError("vLLM completion indexes are missing, repeated or outside the batch")
            logprobs = choice.get("logprobs") or {}
            top = logprobs.get("top_logprobs")
            if not isinstance(top, list) or len(top) != 1 or not isinstance(top[0], dict):
                raise ValueError("vLLM must return one token's top_logprobs for every decision")
            keys = [f"token_id:{token_id}" for token_id in labels]
            if any(key not in top[0] for key in keys):
                raise ValueError("Missing label logprobs; require processed_logprobs and all allowed token IDs")
            values = [top[0][key] for key in keys]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v > 1e-5 for v in values):
                raise ValueError("vLLM returned invalid label log probabilities")
            # A raw full-vocabulary top-k response is a protocol mismatch. Never
            # silently score only whichever yes/no label happened to make top-k.
            total = sum(math.exp(v) for v in values)
            if not math.isclose(total, 1.0, rel_tol=2e-4, abs_tol=2e-4):
                raise ValueError("Allowed-label logprobs are not normalized; check processed_logprobs mode")
            largest = max(values)
            weights = [math.exp(v - largest) for v in values]
            denominator = sum(weights)
            result[index] = [weight / denominator for weight in weights]
        return result

    def _score_batch(self, prompts: list[list[int]], labels: Sequence[int], salt: str | None):
        payload = {
            "model": self.served_model_name, "prompt": prompts,
            "max_tokens": 1, "min_tokens": 1, "n": 1, "stream": False,
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "seed": 0, "allowed_token_ids": list(labels), "logprobs": len(labels),
            "return_tokens_as_token_ids": True, "add_special_tokens": False,
            "ignore_eos": True, "stop": [], "stop_token_ids": [],
        }
        if salt is not None:
            payload["cache_salt"] = salt
        body = self._request("POST", "/v1/completions", payload=payload)
        rows = self._parse_choices(body, len(prompts), labels)
        usage = body.get("usage") or {}
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        expected_tokens = sum(len(prompt) for prompt in prompts)
        if prompt_tokens != expected_tokens or completion_tokens != len(prompts):
            raise ValueError("vLLM token usage does not match the submitted one-token prompt batch")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is not None and (not isinstance(cached, int) or not 0 <= cached <= prompt_tokens):
            raise ValueError("Invalid cached-token count from vLLM")
        return rows, {
            "requests": 1, "prompt_requests": len(prompts), "input_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "cached_tokens": cached,
        }

    def score_token_prompts(self, prompts: list[list[int]], labels: Sequence[int], *, prime_prefix=False):
        """Reusable exact label scorer; inputs and label IDs must match the server tokenizer."""
        if not prompts:
            return [], {"requests": 0, "prompt_requests": 0, "input_tokens": 0,
                        "completion_tokens": 0, "cached_tokens": 0}
        if not labels or len(set(labels)) != len(labels) or len(labels) > 26:
            raise ValueError("Provide 1–26 distinct label token IDs")
        if any(not prompt or len(prompt) + 1 > self.max_context_tokens for prompt in prompts):
            raise ValueError("Prompt plus one scoring token exceeds vLLM max_context_tokens; no truncation applied")
        salt = self._cache_salt  # stable through every batch within a query
        results = []
        if prime_prefix and len(prompts) > 1:
            # Complete one real judgment to populate the common prefix before the
            # remaining candidates enter the server scheduler. Its work is counted.
            results.append(self._score_batch(prompts[:1], labels, salt))
            prompts = prompts[1:]
        batches = [prompts[start:start + self.batch_size] for start in range(0, len(prompts), self.batch_size)]
        if self.concurrency > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                results.extend(pool.map(lambda batch: self._score_batch(batch, labels, salt), batches))
        else:
            results.extend(self._score_batch(batch, labels, salt) for batch in batches)
        rows = [row for part, _ in results for row in part]
        stats = {key: sum(stat[key] for _, stat in results)
                 for key in ("requests", "prompt_requests", "input_tokens", "completion_tokens")}
        cached = [stat["cached_tokens"] for _, stat in results]
        stats["cached_tokens"] = sum(cached) if all(value is not None for value in cached) else None
        return rows, stats

    def binary(self, context: str, questions: Sequence[str]) -> DecisionOutput:
        rows, stats = self.score_token_prompts(
            self._binary_prompts(context, questions), self.label_ids, prime_prefix=self.prime_prefix,
        )
        return DecisionOutput([row[1] for row in rows], stats["input_tokens"], None if rows else 0, stats)

    def choice(self, context: str, question: str, options: Sequence[str]) -> DecisionOutput:
        if len(options) < 2:
            return DecisionOutput([1.0] if options else [], 0, 0,
                                  {"requests": 0, "prompt_requests": 0, "completion_tokens": 0, "cached_tokens": 0})
        if len(options) > 26:
            raise ValueError("vLLM choice supports at most 26 single-token letter options")
        labels = list(string.ascii_uppercase[:len(options)])
        ids = self._single_token_ids(labels)
        option_text = "\n".join(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
        prompt = self._chat(context + "\n\n" + question + "\n" + option_text
                            + "\nAnswer with only the letter of the best option.")
        rows, stats = self.score_token_prompts([self.tokenizer.encode(prompt, add_special_tokens=False)], ids)
        return DecisionOutput(rows[0], stats["input_tokens"], None, stats)

    def close(self):
        if self._owns_client:
            self.client.close()
