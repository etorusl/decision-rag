import json

import httpx
import numpy as np
import pytest

from decision_rag.schema import Corpus, Node, Query
from decision_rag.vllm_retrieval import (
    EMBED_QUERY_PREFIX,
    RERANK_PREFIX,
    RERANK_SUFFIX,
    VLLMDenseRerankRetriever,
    VLLMDenseRetriever,
    VLLMQwenReranker,
)

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"
REVISION = "a" * 40


def corpus():
    return Corpus([Node("root", "", "root"), Node("page", "", "page", "root"),
                   Node("p1", "apple", "passage", "page", order=0),
                   Node("p2", "pear", "passage", "page", order=1),
                   Node("p3", "banana", "passage", "page", order=2)])


def pooling_client(model, callback):
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model, "root": model, "max_model_len": 8192}]})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.2"})
        return callback(request)
    return httpx.Client(transport=httpx.MockTransport(handler))


def embeddings_reply(request, calls):
    assert request.url.path == "/v1/embeddings"
    body = json.loads(request.content)
    calls.append(body)
    values = {"apple": [3, 0], "pear": [1, 1], "banana": [0, 2]}
    rows = [{"index": i, "embedding": values.get(text, [1, 0])} for i, text in enumerate(body["input"])]
    return httpx.Response(200, json={"model": EMBED_MODEL, "data": rows[::-1],
                                    "usage": {"prompt_tokens": len(rows) * 3}})


def test_live_query_encoding_normalized_order_and_usage():
    calls = []
    client = pooling_client(EMBED_MODEL, lambda request: embeddings_reply(request, calls))
    dense = VLLMDenseRetriever(corpus(), client=client, batch_size=2)
    assert len(calls) == 2
    np.testing.assert_allclose(np.linalg.norm(dense.embeddings, axis=1), 1)
    first = dense.retrieve(Query("q", "where is apple"))
    second = dense.retrieve(Query("q", "where is apple"))
    assert first.ids == second.ids == ["p1", "p2", "p3"]
    assert len(calls) == 4  # query embeddings are never cached
    assert calls[-1]["input"] == [EMBED_QUERY_PREFIX + "where is apple"]
    assert calls[-1]["encoding_format"] == "float"
    assert first.stages[0]["requests"] == 1
    assert first.stages[0]["prompt_requests"] == 1
    assert first.stages[0]["input_tokens"] == 3
    assert first.stages[0]["forward_passes"] is None
    assert dense.external_inference is True
    assert dense.runtime_metadata["embedding"]["server_version"] == "0.11.2"


def test_cache_requires_revision_and_rejects_changed_identity(tmp_path):
    with pytest.raises(ValueError, match="immutable revision"):
        VLLMDenseRetriever(corpus(), index_dir=tmp_path)
    calls = []
    client = pooling_client(EMBED_MODEL, lambda request: embeddings_reply(request, calls))
    first = VLLMDenseRetriever(corpus(), index_dir=tmp_path, revision=REVISION, client=client)
    assert len(calls) == 1
    second = VLLMDenseRetriever(corpus(), index_dir=tmp_path, revision=REVISION, client=client)
    assert len(calls) == 1
    assert second.runtime_metadata["index_cache_reused"] is True
    np.testing.assert_allclose(first.embeddings, second.embeddings)
    with pytest.raises(ValueError, match="mismatch"):
        VLLMDenseRetriever(corpus(), index_dir=tmp_path, revision="b" * 40, client=client)
    with pytest.raises(ValueError, match="mismatch"):
        VLLMDenseRetriever(corpus(), index_dir=tmp_path, revision=REVISION, dtype="float16", client=client)
    different = Corpus([Node("root", "", "root"), Node("page", "", "page", "root"),
                        Node("p1", "changed text", "passage", "page")])
    with pytest.raises(ValueError, match="mismatch"):
        VLLMDenseRetriever(different, index_dir=tmp_path, revision=REVISION, client=client)


def test_index_progress_counts_actual_passages_and_reports_cache_reuse(tmp_path):
    from decision_rag.observability import RunReporter

    calls = []
    client = pooling_client(EMBED_MODEL, lambda request: embeddings_reply(request, calls))
    with RunReporter(tmp_path / 'logs', progress='off') as reporter:
        first = VLLMDenseRetriever(corpus(), index_dir=tmp_path / 'index', revision=REVISION,
                                   client=client, batch_size=2, reporter=reporter)
        second = VLLMDenseRetriever(corpus(), index_dir=tmp_path / 'index', revision=REVISION,
                                    client=client, reporter=reporter)
    assert len(calls) == 2
    np.testing.assert_allclose(first.embeddings, second.embeddings)
    events = [json.loads(line) for line in (tmp_path / 'logs' / 'events.jsonl').read_text().splitlines()]
    built = next(row for row in events if row['event'] == 'phase_completed')
    assert built['completed'] == built['total'] == 3
    assert built['metrics']['batches'] == 2
    assert any(row['event'] == 'index_cache_loaded' for row in events)


@pytest.mark.parametrize("failure", ["indices", "nan", "zero", "dimensions", "model"])
def test_embedding_responses_fail_closed(failure):
    def bad(request):
        rows = [{"index": i, "embedding": [1, 0]} for i in range(3)]
        model = EMBED_MODEL
        if failure == "indices":
            rows[2]["index"] = 1
        elif failure == "nan":
            # A very large numeric value overflows FP32 and is rejected as nonfinite.
            rows[0]["embedding"] = [1e300, 0]
        elif failure == "zero":
            rows[0]["embedding"] = [0, 0]
        elif failure == "dimensions":
            rows[0]["embedding"] = [1, 2, 3]
        else:
            model = "wrong"
        return httpx.Response(200, json={"model": model, "data": rows})
    with pytest.raises(ValueError):
        VLLMDenseRetriever(corpus(), client=pooling_client(EMBED_MODEL, bad))


def test_server_must_advertise_requested_model():
    client = pooling_client("wrong-model", lambda request: pytest.fail("Inference must not run"))
    with pytest.raises(ValueError, match="does not advertise"):
        VLLMDenseRetriever(corpus(), client=client)


def test_score_uses_official_prompt_preserves_indices_and_reports_unknown_passes():
    calls = []
    def reply(request):
        assert request.url.path == "/score"
        body = json.loads(request.content)
        calls.append(body)
        values = [0.2, 0.8][:len(body["text_2"])]
        return httpx.Response(200, json={"model": RERANK_MODEL,
                                        "data": [{"index": i, "score": value} for i, value in enumerate(values)][::-1],
                                        "usage": {"prompt_tokens": 30}})
    scorer = VLLMQwenReranker(client=pooling_client(RERANK_MODEL, reply), batch_size=2)
    scores, usage = scorer.score("question", ["A", "B", "C"])
    assert scores == [.2, .8, .2]
    assert usage["requests"] == 2
    assert usage["prompt_requests"] == 3
    assert usage["input_tokens"] == 60
    assert usage["forward_passes"] is None
    assert calls[0]["text_1"].startswith(RERANK_PREFIX + "<Instruct>:")
    assert calls[0]["text_1"].endswith("<Query>: question\n")
    assert calls[0]["text_2"] == ["<Document>: A" + RERANK_SUFFIX, "<Document>: B" + RERANK_SUFFIX]
    assert calls[0]["use_activation"] is True
    assert "truncate_prompt_tokens" not in calls[0]


def test_score_errors_do_not_fallback_or_silently_truncate():
    def fail(request):
        return httpx.Response(400, json={"error": {"message": "context too long"}})
    scorer = VLLMQwenReranker(client=pooling_client(RERANK_MODEL, fail))
    with pytest.raises(RuntimeError, match="context too long"):
        scorer.score("query", ["long doc"])


def test_dense_rerank_includes_live_dense_and_reranker_work():
    calls = []
    embedding_client = pooling_client(EMBED_MODEL, lambda request: embeddings_reply(request, calls))
    def scores(request):
        body = json.loads(request.content)
        assert len(body["text_2"]) == 3
        return httpx.Response(200, json={"model": RERANK_MODEL,
                                        "data": [{"index": 0, "score": .1}, {"index": 1, "score": .9},
                                                 {"index": 2, "score": .9}]})
    retriever = VLLMDenseRerankRetriever(corpus(), candidates=3, top_k=2,
                                        embedding_client=embedding_client,
                                        reranker_client=pooling_client(RERANK_MODEL, scores))
    result = retriever.retrieve(Query("q", "apple"))
    assert len(calls) == 2
    assert result.ids == ["p2", "p3"]  # ties retain original dense order
    assert [row["stage"] for row in result.stages] == ["dense", "rerank"]
    assert result.stages[-1]["input_tokens"] is None
    assert result.latency_seconds >= sum(row["seconds"] for row in result.stages)
    assert retriever.runtime_metadata["reranker"]["model"] == RERANK_MODEL


def test_invalid_score_probability_is_not_accepted():
    def bad(request):
        return httpx.Response(200, json={"data": [{"index": 0, "score": 2.0}]})
    scorer = VLLMQwenReranker(client=pooling_client(RERANK_MODEL, bad))
    with pytest.raises(ValueError, match="probabilities"):
        scorer.score("q", ["d"])
