"""Appendix-G-style decision head with shared-context, isolated question branches.

The mask describes a tree: context -> question -> answer option. Questions start at
identical rotary positions and cannot read each other. CUDA uses a sparse FlexAttention
BlockMask; a bounded dense mask exists solely for CPU correctness tests/small runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .base import DecisionOutput


@dataclass
class BranchLayout:
    input_ids: list[int]
    position_ids: list[int]
    owners: list[int]
    branches: list[int]
    option_end: list[bool]
    question_ends: list[int]
    option_ends: list[list[int]]
    context_length: int
    option_links: bool = False

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def mask_function(self, device: str | torch.device):
        # BlockMask construction may inspect the rounded-up final block.
        padded = math.ceil(self.length / 128) * 128
        pad = padded - self.length
        owners = torch.tensor(self.owners + [-2] * pad, device=device)
        branches = torch.tensor(self.branches + [-2] * pad, device=device)
        positions = torch.tensor(self.position_ids + [0] * pad, device=device)
        ends = torch.tensor(self.option_end + [False] * pad, device=device)
        length, links = self.length, self.option_links

        def allowed(batch, head, q, k):
            causal = positions[k] <= positions[q]
            context_key = (owners[k] == -1) & (k <= q)
            same_question = (owners[q] >= 0) & (owners[q] == owners[k])
            question_key = branches[k] == 0
            own_branch = branches[q] == branches[k]
            branch_key = same_question & (question_key | own_branch) & causal
            linked_ends = same_question & ends[q] & ends[k] & links
            return (q < length) & (k < length) & (context_key | branch_key | linked_ends)

        return allowed

    def attention_mask(self, device, dtype, implementation: str):
        allowed = self.mask_function(device)
        if implementation == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            return create_block_mask(
                allowed,
                B=1,
                H=None,
                Q_LEN=self.length,
                KV_LEN=self.length,
                device=str(device),
                _compile=True,
            )
        if self.length > 4096:
            raise ValueError(
                "Dense branch mask is limited to 4096 tokens. Use CUDA --attention flex_attention "
                "for long contexts; refusing to allocate a quadratic mask."
            )
        indexes = torch.arange(self.length, device=device)
        visible = allowed(0, 0, indexes[:, None], indexes[None, :])
        mask = torch.zeros((self.length, self.length), dtype=dtype, device=device)
        return mask.masked_fill(~visible, torch.finfo(dtype).min)[None, None]

    def model_inputs(self, device, dtype, implementation: str) -> dict:
        mask = self.attention_mask(device, dtype, implementation)
        return {
            "input_ids": torch.tensor([self.input_ids], device=device),
            "position_ids": torch.tensor([self.position_ids], device=device),
            # A dictionary bypasses Transformers' extra causal masking. Necessary for
            # the optional non-causal links between answer-option end tokens.
            "attention_mask": {"full_attention": mask},
            "use_cache": False,
        }


def pack_branches(
    prefix: Sequence[int],
    questions: Sequence[Sequence[int]],
    options: Sequence[Sequence[Sequence[int]]] | None = None,
    *,
    option_links: bool = False,
) -> BranchLayout:
    """Pack independent branches without duplicating context tokens or rotary positions."""
    if not prefix:
        raise ValueError("Shared context must contain at least one token")
    if options is not None and len(options) != len(questions):
        raise ValueError("Each question needs its own option list")
    ids = list(prefix)
    positions = list(range(len(ids)))
    owners, branches, end_mask = [-1] * len(ids), [-1] * len(ids), [False] * len(ids)
    q_ends, all_option_ends = [], []
    for i, question in enumerate(questions):
        if not question:
            raise ValueError("Empty question token sequence")
        ids.extend(question)
        positions.extend(range(len(prefix), len(prefix) + len(question)))
        owners.extend([i] * len(question))
        branches.extend([0] * len(question))
        end_mask.extend([False] * len(question))
        q_ends.append(len(ids) - 1)
        option_ends = []
        for j, option in enumerate(options[i] if options is not None else []):
            if not option:
                raise ValueError("Empty answer option token sequence")
            ids.extend(option)
            start = len(prefix) + len(question)
            positions.extend(range(start, start + len(option)))
            owners.extend([i] * len(option))
            branches.extend([j + 1] * len(option))
            end_mask.extend([False] * (len(option) - 1) + [True])
            option_ends.append(len(ids) - 1)
        all_option_ends.append(option_ends)
    return BranchLayout(
        ids, positions, owners, branches, end_mask, q_ends, all_option_ends, len(prefix), option_links
    )


def encode_record(
    tokenizer,
    context: str,
    questions: Sequence[str],
    options: Sequence[Sequence[str]],
    *,
    option_links: bool = False,
) -> BranchLayout:
    if len(questions) != len(options) or any(len(row) < 2 for row in options):
        raise ValueError("Each question requires at least two options")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Decision-head tokenizer must define eos_token_id")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = encode(context + "\n\n")
    if not prefix:
        prefix = [eos]
    question_tokens = [encode("Question: " + question + "\n") + [eos] for question in questions]
    option_tokens = [[encode("Option: " + option + "\n") + [eos] for option in row] for row in options]
    return pack_branches(prefix, question_tokens, option_tokens, option_links=option_links)


class CosineDecisionHead(nn.Module):
    """Question/option projections and learned inverse temperature (Appendix G)."""

    def __init__(self, hidden_size: int, projection_dim: int = 1024):
        super().__init__()
        self.question_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.option_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.log_inverse_temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, question: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        # Keep the small head/softmax in FP32 even with a BF16 backbone.
        q = F.normalize(self.question_projection(question.float()), dim=-1)
        o = F.normalize(self.option_projection(options.float()), dim=-1)
        scale = self.log_inverse_temperature.clamp(max=math.log(100.0)).exp()
        return (q.unsqueeze(-2) * o).sum(dim=-1) * scale

    def forward_ragged(
        self, questions: torch.Tensor, options: torch.Tensor, option_counts: Sequence[int]
    ) -> list[torch.Tensor]:
        """Project every question/option in two GEMMs, even with unequal option counts."""
        if len(option_counts) != len(questions) or sum(option_counts) != len(options):
            raise ValueError("Question/option states do not match option_counts")
        if any(count < 1 for count in option_counts):
            raise ValueError("Every question requires answer options")
        if not option_counts:
            return []
        q = F.normalize(self.question_projection(questions.float()), dim=-1)
        o = F.normalize(self.option_projection(options.float()), dim=-1)
        owners = torch.tensor(
            [i for i, count in enumerate(option_counts) for _ in range(count)], device=q.device
        )
        scale = self.log_inverse_temperature.clamp(max=math.log(100.0)).exp()
        scores = (q.index_select(0, owners) * o).sum(dim=-1) * scale
        return list(scores.split(tuple(option_counts)))


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, projection_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.head = CosineDecisionHead(backbone.config.hidden_size, projection_dim)

    def forward(self, layout: BranchLayout, implementation: str) -> list[torch.Tensor]:
        parameter = next(self.backbone.parameters())
        inputs = layout.model_inputs(parameter.device, parameter.dtype, implementation)
        # PEFT's input-gradient hook can remain installed during the head-only phase.
        # Explicit no_grad avoids retaining an unnecessary full backbone graph then.
        train_backbone = torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(train_backbone):
            hidden = self.backbone(**inputs)
        states = hidden.last_hidden_state[0]
        if not layout.question_ends:
            return []
        question_indexes = torch.tensor(layout.question_ends, device=states.device)
        option_indexes = torch.tensor(
            [end for ends in layout.option_ends for end in ends], device=states.device
        )
        return self.head.forward_ragged(
            states.index_select(0, question_indexes),
            states.index_select(0, option_indexes),
            [len(ends) for ends in layout.option_ends],
        )


def resolve_attention(device: str, attention: str) -> str:
    if attention == "auto":
        return "flex_attention" if device.startswith("cuda") else "sdpa"
    if attention not in {"flex_attention", "sdpa", "eager"}:
        raise ValueError(f"Unknown attention implementation: {attention}")
    if attention == "flex_attention" and not device.startswith("cuda"):
        raise ValueError("FlexAttention backend requires CUDA; use sdpa for small CPU tests")
    return attention


def resolve_dtype(dtype: str):
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in values:
        raise ValueError(f"Unsupported dtype {dtype}; choose {list(values)}")
    return values[dtype]


def check_model(backbone) -> None:
    # Dict masks/reset rotary positions have been verified against this implementation.
    if backbone.config.model_type != "qwen3":
        raise ValueError("Packed decision backends currently support Qwen3 dense models only")
    if getattr(backbone.config, "use_sliding_window", False):
        raise ValueError("Sliding-window attention is not supported by the branch mask")


def check_layout(layout: BranchLayout, max_context_tokens: int, max_packed_tokens: int):
    if max(layout.position_ids) >= max_context_tokens:
        raise ValueError(
            f"Context plus longest question/option exceeds {max_context_tokens} rotary positions; "
            "reduce retrieval request size. Text is never silently truncated."
        )
    if layout.length > max_packed_tokens:
        raise ValueError(f"Packed request has {layout.length} tokens, limit is {max_packed_tokens}")


class DecisionHeadBackend:
    name = "trained-decision-head"

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_context_tokens: int = 32768,
        max_packed_tokens: int = 65536,
        attention: str = "auto",
    ):
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        path = Path(checkpoint)
        manifest_path = path / "decision_config.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing trained decision checkpoint: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("trained_steps", 0):
            raise ValueError("Refusing to score with an untrained/random decision head")
        self.attention = resolve_attention(device, attention)
        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        model_path = self.manifest["base_model"]
        if self.manifest["mode"] == "full":
            model_path = str(path / "backbone")
        backbone = AutoModel.from_pretrained(
            model_path,
            torch_dtype=resolve_dtype(dtype),
            attn_implementation=self.attention,
            revision=self.manifest.get("base_revision") if self.manifest["mode"] != "full" else None,
        )
        if self.manifest["mode"] == "lora":
            from peft import PeftModel

            backbone = PeftModel.from_pretrained(backbone, path / "adapter")
        check_model(backbone)
        self.model = DecisionModel(backbone, self.manifest["projection_dim"])
        self.model.head.load_state_dict(load_file(str(path / "head.safetensors")))
        self.model.to(device).eval()
        self.max_context_tokens = min(max_context_tokens, backbone.config.max_position_embeddings)
        self.max_packed_tokens = max_packed_tokens
        self.option_links = self.manifest.get("option_links", False)
        self.name = f"trained-decision-head:{path.name}"

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.inference_mode()
    def _score(
        self, context: str, questions: Sequence[str], options: Sequence[Sequence[str]]
    ) -> tuple[list[list[float]], int]:
        if not questions:
            return [], 0
        layout = encode_record(self.tokenizer, context, questions, options, option_links=self.option_links)
        check_layout(layout, self.max_context_tokens, self.max_packed_tokens)
        logits = self.model(layout, self.attention)
        if len({row.numel() for row in logits}) == 1:
            # Binary retrieval is the common case: one softmax and one host transfer.
            probabilities = torch.stack(logits).softmax(-1).tolist()
        else:
            probabilities = [row.softmax(-1).tolist() for row in logits]
        return probabilities, layout.length

    def binary(self, context: str, questions: Sequence[str]) -> DecisionOutput:
        probabilities, tokens = self._score(context, questions, [["no", "yes"] for _ in questions])
        return DecisionOutput([p[1] for p in probabilities], tokens, int(bool(questions)))

    def choice(self, context: str, question: str, options: Sequence[str]) -> DecisionOutput:
        if len(options) < 2:
            return DecisionOutput([1.0] if options else [], 0, 0)
        probabilities, tokens = self._score(context, [question], [options])
        return DecisionOutput(probabilities[0], tokens, 1)
