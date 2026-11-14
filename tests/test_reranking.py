from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from decision_rag.reranking import RERANK_SUFFIX, DenseRerankRetriever, QwenReranker
from decision_rag.retrieval import RetrievalResult
from decision_rag.schema import Corpus, Node, Query


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 1
    eos_token = "[EOS]"

    def encode(self, text, add_special_tokens=False):
        # One character per token makes truncation independently inspectable.
        return [ord(c) + 4 for c in text]

    def convert_tokens_to_ids(self, value):
        return {"no": 2, "yes": 3}[value]

    def pad(self, inputs, padding, return_tensors):
        length = max(map(len, inputs["input_ids"]))
        ids = [[0] * (length - len(row)) + row for row in inputs["input_ids"]]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(ids).ne(0).long()}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", max_position_embeddings=32768)
        self.head = torch.nn.Linear(1, 512, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[3, 0] = 1
        self.calls = []

    @property
    def base_model(self):
        return self

    def get_output_embeddings(self):
        return self.head

    def generate(self, *args, **kwargs):
        pytest.fail("Reranking must never generate")

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        assert not use_cache
        assert input_ids[:, -1].ne(0).all()
        # Last actual token has its unpadded rotary position in every row.
        assert torch.equal(position_ids[:, -1], attention_mask.sum(-1) - 1)
        self.calls.append(input_ids.numel())
        positive_markers = input_ids.eq(ord("Z") + 4).sum(-1).float()
        state = positive_markers[:, None, None].expand(-1, input_ids.shape[-1], 1)
        return SimpleNamespace(last_hidden_state=state)


@pytest.fixture
def fake_hf(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeModel())


def test_restricted_logits_probabilities_batching_and_alignment(fake_hf):
    scorer = QwenReranker(device="cpu", dtype="float32", batch_size=2, max_length=1024, max_batch_tokens=1100)
    scores, stats = scorer.score("question", ["x" * 100, "ZZ", "Z" * 3 + "x" * 300])
    expected = torch.tensor([0.0, 2.0, 3.0]).sigmoid().tolist()
    assert scores == pytest.approx(expected)
    assert stats["forward_passes"] >= 2
    assert max(scorer.model.calls) <= 1100
    assert stats["padded_tokens"] >= stats["input_tokens"]
    assert stats["truncated_candidates"] == 0


def test_document_truncation_preserves_suffix_and_records_counts(fake_hf):
    scorer = QwenReranker(device="cpu", dtype="float32", max_length=512, max_batch_tokens=1024)
    rows, count, removed = scorer._encode("query", ["x" * 2000, "short"])
    assert count == 1 and removed > 0
    assert len(rows[0]) == 512
    suffix = scorer.tokenizer.encode(RERANK_SUFFIX)
    assert rows[0][-len(suffix) :] == suffix
    assert len(rows[1]) < 512
    with pytest.raises(ValueError, match="no reranker document budget"):
        scorer._encode("query " * 1000, ["short"])
    scorer.truncation = "error"
    with pytest.raises(ValueError, match="truncation=error"):
        scorer.score("query", ["x" * 2000])


def test_dense_reranker_uses_fresh_candidates_and_stable_ties(fake_hf, monkeypatch):
    corpus = Corpus(
        [
            Node("root", "", "root"),
            Node("page", "Page", "page", "root"),
            Node("p1", "ZZ", "passage", "page", order=1),
            Node("p2", "ZZ", "passage", "page", order=2),
            Node("p3", "x", "passage", "page", order=3),
        ]
    )

    class FakeDense:
        def __init__(self, *args, **kwargs):
            assert kwargs["dtype"] == "float32"
            self.calls, self.resets = 0, 0

        def reset(self):
            self.resets += 1

        def retrieve(self, query):
            self.calls += 1
            return RetrievalResult(query.id, ["p3", "p2", "p1"], [3.0, 2.0, 1.0], 0, [])

    monkeypatch.setattr("decision_rag.reranking.DenseRetriever", FakeDense)
    retriever = DenseRerankRetriever(corpus, top_k=2, candidates=3, device="cpu", dtype="float32")
    result = retriever.retrieve(Query("q", "question"))
    assert result.ids == ["p2", "p1"]  # Equal reranker scores retain the dense order.
    assert retriever.dense.calls == 1
    assert result.stages[1]["candidates"] == 3
    assert result.latency_seconds >= result.stages[0]["latency_seconds"]
    retriever.reset()
    assert retriever.dense.resets == 1
    with pytest.raises(ValueError, match="candidate count"):
        DenseRerankRetriever(corpus, top_k=20, candidates=10)
