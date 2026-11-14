"""Numerical invariants, no pretrained weights or network access required."""

from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from decision_rag.backends.decision_head import (
    CosineDecisionHead,
    DecisionHeadBackend,
    DecisionModel,
    check_layout,
    pack_branches,
)
from decision_rag.backends.hf import HFLogitBackend
from decision_rag.training import train_records


def tiny_config():
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        eos_token_id=1,
        pad_token_id=0,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )


def tiny_tokenizer():
    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "context",
        "Question",
        "Option",
        ":",
        "A",
        "B",
        "yes",
        "no",
        "one",
        "two",
        "q1",
        "q2",
        "candidate",
        "user",
        "assistant",
    ]
    tokenizer = Tokenizer(WordLevel({s: i for i, s in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    result.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return result


def dense_visible(layout):
    indexes = torch.arange(layout.length)
    return layout.mask_function("cpu")(0, 0, indexes[:, None], indexes[None, :])


def test_branch_attention_and_reset_positions():
    layout = pack_branches([2, 3], [[4, 5], [6]], [[[7], [8]], [[9], [10]]])
    visible = dense_visible(layout)
    assert layout.position_ids == [0, 1, 2, 3, 4, 4, 2, 3, 3]
    assert visible[7, :2].all()  # Every option sees shared context.
    assert visible[7, 6]  # Option sees own question.
    assert not visible[7, 3]  # Another question is invisible.
    assert not visible[4, 5] and not visible[5, 4]  # Sibling options isolated.
    assert not visible[0, 2]  # Context never sees appended questions.
    linked = pack_branches([2, 3], [[4, 5]], [[[7], [8]]], option_links=True)
    assert dense_visible(linked)[4, 5] and dense_visible(linked)[5, 4]


def test_packed_hidden_matches_independent_branches():
    torch.manual_seed(2)
    model = Qwen3Model(tiny_config()).eval()
    layout = pack_branches([2, 3, 4], [[5, 6], [7, 8, 9]])
    with torch.inference_mode():
        packed = model(**layout.model_inputs("cpu", torch.float32, "sdpa")).last_hidden_state
        for i, branch in enumerate(([5, 6], [7, 8, 9])):
            independent = model(torch.tensor([[2, 3, 4, *branch]]), use_cache=False).last_hidden_state
            torch.testing.assert_close(
                packed[0, layout.question_ends[i]], independent[0, -1], rtol=1e-5, atol=1e-6
            )


def test_option_order_does_not_change_scores_and_gradients_flow():
    torch.manual_seed(4)
    model = DecisionModel(Qwen3Model(tiny_config()), projection_dim=8)
    original = pack_branches([2, 3], [[4, 5]], [[[6, 7], [8, 9]]])
    reversed_options = pack_branches([2, 3], [[4, 5]], [[[8, 9], [6, 7]]])
    logits = model(original, "sdpa")[0]
    reversed_logits = model(reversed_options, "sdpa")[0]
    torch.testing.assert_close(logits, reversed_logits.flip(0), rtol=1e-5, atol=1e-6)
    torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([1])).backward()
    assert model.head.question_projection.weight.grad.abs().sum() > 0
    assert model.backbone.embed_tokens.weight.grad.abs().sum() > 0


def test_temperature_is_capped_and_zero_vectors_are_finite():
    head = CosineDecisionHead(4, 4)
    with torch.no_grad():
        head.log_inverse_temperature.fill_(1000)
    logits = head(torch.zeros(4), torch.zeros(2, 4))
    assert torch.isfinite(logits).all()
    assert logits.abs().max() <= 100


@pytest.mark.parametrize("counts", [(2, 2, 2), (2, 3, 4)])
def test_batched_head_matches_individual_logits_and_gradients(counts):
    torch.manual_seed(12)
    head = CosineDecisionHead(32, 8)
    reference = copy.deepcopy(head)
    questions = torch.randn(len(counts), 32, requires_grad=True)
    options = torch.randn(sum(counts), 32, requires_grad=True)
    calls = {"question": 0, "option": 0}

    def record(name):
        def hook(module, args, output):
            calls[name] += 1

        return hook

    head.question_projection.register_forward_hook(record("question"))
    head.option_projection.register_forward_hook(record("option"))
    actual = head.forward_ragged(questions, options, counts)
    assert calls == {"question": 1, "option": 1}
    assert [len(row) for row in actual] == list(counts)
    expected = [reference(question, row) for question, row in zip(questions, options.split(counts))]
    torch.testing.assert_close(torch.cat(actual), torch.cat(expected), rtol=1e-5, atol=1e-6)
    torch.cat(actual).sum().backward(retain_graph=True)
    torch.cat(expected).sum().backward()
    for actual_parameter, expected_parameter in zip(head.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-6)


def test_long_dense_masks_fail_before_quadratic_allocation():
    layout = pack_branches([2] * 4096, [[3]])
    with pytest.raises(ValueError, match="quadratic mask"):
        layout.attention_mask("cpu", torch.float32, "sdpa")
    with pytest.raises(ValueError, match="rotary positions"):
        check_layout(layout, 4096, 65536)


def test_hf_scores_are_order_invariant_and_do_not_generate():
    torch.manual_seed(3)
    backend = HFLogitBackend.__new__(HFLogitBackend)
    backend.model = Qwen3ForCausalLM(tiny_config()).eval()
    backend.tokenizer = tiny_tokenizer()
    backend.device = "cpu"
    backend.attention = "sdpa"
    backend.max_context_tokens = 1024
    backend.max_packed_tokens = 2048
    backend.label_ids = backend._single_token_ids(["no", "yes"])
    backend.model.generate = lambda *a, **kw: pytest.fail("Generation must never be called")
    backend.model.forward = lambda *a, **kw: pytest.fail("Full vocabulary LM forward must never be called")
    original = backend.binary("context one two", ["q1", "q2"])
    swapped = backend.binary("context one two", ["q2", "q1"])
    assert original.forward_passes == 1
    assert original.probabilities == pytest.approx(swapped.probabilities[::-1], abs=1e-6)
    choice = backend.choice("context one two", "q1", ["one", "two"])
    assert sum(choice.probabilities) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at most 26"):
        backend.choice("context", "q1", ["one"] * 27)


@pytest.mark.parametrize("mode", ["head", "lora"])
def test_tiny_training_and_checkpoint_reload(tmp_path, mode):
    if mode == "lora":
        pytest.importorskip("peft")
    torch.manual_seed(1)
    source = tmp_path / "base"
    model = Qwen3Model(tiny_config())
    model.save_pretrained(source)
    tokenizer = tiny_tokenizer()
    tokenizer.save_pretrained(source)
    train = tmp_path / "train.jsonl"
    row = {
        "context": "context one two",
        "questions": ["q1", "q2"],
        "options": [["no", "yes"], ["no", "yes"]],
        "labels": [1, 0],
    }
    train.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    output = tmp_path / "checkpoint"
    summary = train_records(
        train,
        output,
        model_name=str(source),
        mode=mode,
        device="cpu",
        dtype="float32",
        projection_dim=8,
        lora_rank=2,
        lora_alpha=4,
        max_steps=1,
        gradient_accumulation=2,
        max_context_tokens=1024,
        max_packed_tokens=2048,
        warmup_steps=0,
        save_every=0,
        dev_path=train,
    )
    assert summary["trained_steps"] == 1
    assert summary["validation"]["questions"] == 4
    backend = DecisionHeadBackend(str(output), device="cpu", dtype="float32")
    result = backend.binary(row["context"], row["questions"])
    assert len(result.probabilities) == 2
    assert all(0 <= value <= 1 for value in result.probabilities)
    assert result.forward_passes == 1
    manifest = output / "decision_config.json"
    data = json.loads(manifest.read_text())
    data["trained_steps"] = 0
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="untrained/random"):
        DecisionHeadBackend(str(output), device="cpu", dtype="float32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA FlexAttention")
def test_cuda_flex_matches_dense_forward_and_backward():
    torch.manual_seed(8)
    config = tiny_config()
    config._attn_implementation = "sdpa"
    dense = DecisionModel(Qwen3Model(config), projection_dim=8).cuda()
    flex_config = tiny_config()
    flex_config._attn_implementation = "flex_attention"
    flex = DecisionModel(Qwen3Model(flex_config), projection_dim=8).cuda()
    flex.load_state_dict(dense.state_dict())
    layout = pack_branches([2, 3, 4], [[5, 6], [7]], [[[8], [9]], [[10], [11]]], option_links=True)
    dense_logits = torch.stack(dense(layout, "sdpa"))
    flex_logits = torch.stack(flex(layout, "flex_attention"))
    torch.testing.assert_close(flex_logits, dense_logits, rtol=5e-4, atol=5e-5)
    labels = torch.tensor([0, 1], device="cuda")
    torch.nn.functional.cross_entropy(dense_logits, labels).backward()
    torch.nn.functional.cross_entropy(flex_logits, labels).backward()
    torch.testing.assert_close(
        flex.head.question_projection.weight.grad,
        dense.head.question_projection.weight.grad,
        rtol=1e-3,
        atol=1e-4,
    )
