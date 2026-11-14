import pytest

from decision_rag.evaluation import evaluate
from decision_rag.retrieval import RetrievalResult
from decision_rag.schema import Corpus, Node, Query, read_jsonl


def corpus():
    return Corpus(
        [
            Node("root", "Corpus", "root"),
            Node("page", "Page", "page", "root"),
            Node("p1", "Same evidence", "passage", "page", order=0),
            Node("p2", "same   evidence", "passage", "page", order=1),
            Node("p3", "Different evidence", "passage", "page", order=2),
        ]
    )


class StatefulRetriever:
    def __init__(self, ranking=None):
        self.corpus = corpus()
        self.ranking = ranking or ["p1"]
        self.memory = {"old": ["contaminated"]}
        self.calls = []
        self.resets = 0

    def reset(self):
        self.memory.clear()
        self.resets += 1

    def retrieve(self, query):
        previous = list(self.memory.get(query.conversation_id, []))
        self.calls.append((query.id, previous))
        self.memory.setdefault(query.conversation_id, []).append(query.id)
        return RetrievalResult(
            query.id,
            list(self.ranking),
            [1.0] * len(self.ranking),
            0.0,
            [{"input_tokens": 100, "forward_passes": 1, "requests": 1}],
        )


@pytest.fixture(autouse=True)
def lightweight_environment(monkeypatch):
    monkeypatch.setattr("decision_rag.evaluation.environment", lambda: {"test": True})
    monkeypatch.setattr("decision_rag.evaluation.synchronize", lambda: None)


def test_warmup_resets_memory_and_excludes_warmup_work(tmp_path):
    queries = [
        Query("q1", "first", {"p1": 1}, "dialogue", 1),
        Query("q2", "second", {"p3": 1}, "dialogue", 2),
        Query("q3", "unlabeled", {}, "other", 1),
    ]
    retriever = StatefulRetriever()
    summary = evaluate(retriever, queries, tmp_path, warmup=1)
    assert retriever.resets == 2
    assert retriever.calls == [("q1", []), ("q1", []), ("q2", ["q1"]), ("q3", [])]
    assert summary["labeled_queries"] == 2 and summary["unlabeled_queries"] == 1
    assert summary["quality"]["recall@1"] == 0.5
    assert summary["latency"]["queries"] == 3 and summary["warmup_queries"] == 1
    assert summary["work"] == {"input_tokens": 300, "forward_passes": 3, "requests": 3}
    assert len(list(read_jsonl(tmp_path / "queries.jsonl"))) == 3


def test_rejects_out_of_order_turns_without_running(tmp_path):
    retriever = StatefulRetriever()
    queries = [
        Query("late", "q", conversation_id="c", turn_id=2),
        Query("early", "q", conversation_id="c", turn_id=1),
    ]
    with pytest.raises(ValueError, match="Non-increasing"):
        evaluate(retriever, queries, tmp_path)
    assert not retriever.calls
    assert not (tmp_path / "queries.jsonl").exists()


def test_interleaved_conversations_keep_independent_memory(tmp_path):
    retriever = StatefulRetriever()
    queries = [
        Query("a1", "q", conversation_id="a", turn_id=1),
        Query("b1", "q", conversation_id="b", turn_id=1),
        Query("a2", "q", conversation_id="a", turn_id=2),
    ]
    evaluate(retriever, queries, tmp_path)
    assert retriever.calls == [("a1", []), ("b1", []), ("a2", ["a1"])]


@pytest.mark.parametrize("invalid", ["duplicate", "unknown_gold", "negative_warmup"])
def test_rejects_invalid_evaluation_inputs_before_running(tmp_path, invalid):
    retriever = StatefulRetriever()
    queries = [Query("q", "question", {"p1": 1})]
    warmup = 0
    if invalid == "duplicate":
        queries += [Query("q", "different question", {"p3": 1})]
    elif invalid == "unknown_gold":
        queries = [Query("q", "question", {"missing": 1})]
    else:
        warmup = -1
    with pytest.raises(ValueError):
        evaluate(retriever, queries, tmp_path, warmup=warmup)
    assert not retriever.calls
    assert not (tmp_path / "queries.jsonl").exists()


def test_text_dedup_maps_rankings_and_gold_to_same_groups(tmp_path):
    retriever = StatefulRetriever(["p2", "p1", "p3"])
    query = Query("q", "question", {"p1": 2, "p2": 1, "p3": 1}, metadata={"dataset": "mtrag"})
    summary = evaluate(retriever, [query], tmp_path, dedup_text=True)
    quality = summary["quality"]
    assert quality["recall@1"] == 0.5
    assert quality["recall@5"] == 1.0
    assert quality["ndcg@1"] == 1.0
    assert quality["ndcg@5"] == 1.0


def test_refuses_overwriting_existing_run(tmp_path):
    retriever = StatefulRetriever()
    query = Query("q", "question", {"p1": 1})
    evaluate(retriever, [query], tmp_path)
    previous = (tmp_path / "queries.jsonl").read_bytes()
    with pytest.raises(ValueError, match="already contains"):
        evaluate(retriever, [query], tmp_path)
    assert (tmp_path / "queries.jsonl").read_bytes() == previous


@pytest.mark.parametrize("grade", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_grades_rejected_before_retrieval(tmp_path, grade):
    retriever = StatefulRetriever()
    with pytest.raises(ValueError, match="finite"):
        evaluate(retriever, [Query("q", "q", {"p1": grade})], tmp_path)
    assert not retriever.calls
