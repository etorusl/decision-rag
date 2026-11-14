"""Hosted Jev adapter using the documented TypeSafe System One wire format.

This module performs no paid calls until binary() or choice() is invoked. There
is deliberately no answer cache: benchmark timings must measure real requests.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Sequence
from typing import ClassVar, Self

import httpx

from .base import DecisionOutput


class JevBackend:
    """Jev 1.13 over TypeSafe directly or the paper's OpenRouter endpoint.

    ``count_tokens`` is a conservative UTF-8-byte bound because neither provider
    publishes Jev's tokenizer. Actual input usage comes from API responses.
    """

    _PROVIDERS: ClassVar = {
        "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-1.13.0"),
        "openrouter": ("https://openrouter.ai/api/v1/systemone", "OPENROUTER_API_KEY", "typesafe/jev-1.13"),
    }
    _RETRYABLE: ClassVar = {429, 500, 502, 503, 504, 529}
    thread_safe = True  # httpx.Client supports sharing a connection pool across threads.

    def __init__(
        self,
        provider: str = "typesafe",
        api_key: str | None = None,
        model: str | None = None,
        expected_snapshot: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        long_context: bool = False,
        client: httpx.Client | None = None,
    ):
        if provider not in self._PROVIDERS:
            raise ValueError(f"Unknown Jev provider: {provider!r}")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries nonnegative")
        self.url, env_name, default_model = self._PROVIDERS[provider]
        key = api_key or os.environ.get(env_name)
        if not key:
            raise ValueError(f"Set {env_name} to use the hosted Jev backend")
        self.provider = provider
        self.model = model or default_model
        self.expected_snapshot = expected_snapshot
        if expected_snapshot is None and model is None:
            self.expected_snapshot = (
                "typesafe/jev-1.13-20260917" if provider == "openrouter" else "jev-1.13.0"
            )
        self.name = f"jev:{provider}:{self.model}"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if long_context:
            if provider != "typesafe":
                raise ValueError("long_context header is only verified for the direct TypeSafe API")
            self.headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or httpx.Client()
        self._owns_client = client is None
        self.last_model: str | None = None
        self.total_cost_usd = 0.0
        self._usage_lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))

    @staticmethod
    def _probability(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Jev returned a nonnumeric probability")  # noqa: TRY004 - invalid server data
        probability = float(value)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Jev returned a probability outside [0, 1]")
        return probability

    @staticmethod
    def _input_tokens(data: dict) -> int:
        value = data.get("usage", {}).get("input_tokens", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Jev returned invalid input token usage")
        return value

    def _post(self, context: str, questions: dict) -> dict:
        body = {"model": self.model, "state": context, "questions": questions}
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code in self._RETRYABLE and attempt < self.max_retries:
                retry_after = response.headers.get("retry-after", "")
                try:
                    delay = max(0.0, min(float(retry_after), 60.0))
                except ValueError:
                    delay = min(2**attempt, 8)
                time.sleep(delay)
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                raise ValueError("Malformed Jev response: expected an answers mapping")  # noqa: TRY004
            response_model = data.get("model")
            if self.expected_snapshot and response_model != self.expected_snapshot:
                raise ValueError(
                    f"Jev model changed: got {response_model!r}, expected {self.expected_snapshot!r}"
                )
            cost = data.get("usage", {}).get("cost", 0)
            with self._usage_lock:
                self.last_model = response_model
                if isinstance(cost, (float, int)) and math.isfinite(cost) and cost >= 0:
                    self.total_cost_usd += cost
            return data
        raise RuntimeError("Jev request exhausted retries")  # defensive; every branch returns or raises

    def binary(self, context: str, questions: Sequence[str]) -> DecisionOutput:
        if not questions:
            return DecisionOutput([], forward_passes=0)
        named = {f"q{i}": {"type": "noul", "instructions": question} for i, question in enumerate(questions)}
        data = self._post(context, named)
        try:
            probabilities = [self._probability(data["answers"][key]["noul"]) for key in named]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed Jev binary response: missing answer") from exc
        return DecisionOutput(probabilities, input_tokens=self._input_tokens(data))

    def choice(self, context: str, question: str, options: Sequence[str]) -> DecisionOutput:
        if not 1 <= len(options) <= 255:
            raise ValueError("Jev choice requires between 1 and 255 options")
        criteria = {f"o{i}": option for i, option in enumerate(options)}
        data = self._post(context, {"choice": {"type": "choice", "instructions": question, "criteria": criteria}})
        try:
            raw = data["answers"]["choice"]["probabilities"]
            if set(raw) != set(criteria):
                raise ValueError("Jev choice response has missing or unexpected options")
            probabilities = [self._probability(raw[key]) for key in criteria]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed Jev choice response") from exc
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Jev choice probabilities do not sum to one")
        return DecisionOutput(probabilities, input_tokens=self._input_tokens(data))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
