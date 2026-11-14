"""Naive zero-shot local judgments from restricted next-token logits, no generation.

All binary questions share one candidate context and run in isolated branches of one
packed forward pass. This is a local baseline, not the paper's trained decision model.
"""

from __future__ import annotations

import string
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .base import DecisionOutput
from .decision_head import check_layout, check_model, pack_branches, resolve_attention, resolve_dtype


class HFLogitBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_context_tokens: int = 32768,
        max_packed_tokens: int = 65536,
        attention: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.attention = resolve_attention(device, attention)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=resolve_dtype(dtype),
                attn_implementation=self.attention,
            )
            .to(device)
            .eval()
        )
        check_model(self.model.base_model)
        self.device = device
        self.max_context_tokens = min(max_context_tokens, self.model.config.max_position_embeddings)
        self.max_packed_tokens = max_packed_tokens
        self.label_ids = self._single_token_ids(["no", "yes"])
        self.name = f"hf-logits:{model_name}"

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _single_token_ids(self, labels: Sequence[str]) -> list[int]:
        ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != len(ids):
            raise ValueError(f"Expected distinct single-token answer labels: {labels}")
        return [row[0] for row in ids]

    def _chat(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _binary_layout(self, context: str, questions: Sequence[str]):
        marker = "__DECISION_RAG_QUESTION_BOUNDARY_4fc71e__"
        if marker in context or any(marker in question for question in questions):
            raise ValueError("Reserved prompt delimiter found in input")
        prompt = self._chat(
            "Judge relevance using the context and candidates below.\n\n"
            + context
            + "\n\n"
            + marker
            + "\nAnswer with exactly one lowercase word: yes or no."
        )
        prefix, suffix = prompt.split(marker)
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        return pack_branches(encode(prefix), [encode(question + suffix) for question in questions])

    def _restricted_logits(self, hidden, label_ids):
        head = self.model.get_output_embeddings()
        indexes = torch.tensor(label_ids, device=head.weight.device)
        weights = head.weight.index_select(0, indexes)
        bias = head.bias.index_select(0, indexes) if getattr(head, "bias", None) is not None else None
        return F.linear(hidden, weights, bias).float()

    @torch.inference_mode()
    def binary(self, context: str, questions: Sequence[str]) -> DecisionOutput:
        if not questions:
            return DecisionOutput([], 0, 0)
        layout = self._binary_layout(context, questions)
        check_layout(layout, self.max_context_tokens, self.max_packed_tokens)
        parameter = next(self.model.parameters())
        hidden = self.model.base_model(
            **layout.model_inputs(parameter.device, parameter.dtype, self.attention)
        ).last_hidden_state[0, layout.question_ends]
        probabilities = self._restricted_logits(hidden, self.label_ids).softmax(-1)[:, 1]
        return DecisionOutput(probabilities.tolist(), layout.length, 1)

    @torch.inference_mode()
    def choice(self, context: str, question: str, options: Sequence[str]) -> DecisionOutput:
        if len(options) < 2:
            return DecisionOutput([1.0] if options else [], 0, 0)
        if len(options) > 26:
            raise ValueError("HF choice baseline supports at most 26 single-token letter options")
        labels = list(string.ascii_uppercase[: len(options)])
        label_ids = self._single_token_ids(labels)
        option_text = "\n".join(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
        prompt = self._chat(
            context
            + "\n\n"
            + question
            + "\n"
            + option_text
            + "\nAnswer with only the letter of the best option."
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > min(self.max_context_tokens, self.max_packed_tokens):
            raise ValueError("Choice context exceeds token budget; reduce shortlist size")
        hidden = self.model.base_model(
            input_ids=torch.tensor([ids], device=self.device),
            use_cache=False,
        ).last_hidden_state[:, -1]
        probabilities = self._restricted_logits(hidden, label_ids).softmax(-1)[0]
        return DecisionOutput(probabilities.tolist(), len(ids), 1)
