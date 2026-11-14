import pytest

from decision_rag.schema import Corpus, load_queries
from decision_rag.stress import generate


def test_stress_deterministic_tree_and_no_answer_leakage(tmp_path):
    kwargs = {
        "pages": 23,
        "passages_per_page": 3,
        "words_per_passage": 32,
        "queries": 10,
        "seed": 7,
        "branching": 3,
    }
    first = generate(tmp_path / "first", **kwargs)
    second = generate(tmp_path / "second", **kwargs)
    assert first == second
    assert first["passages"] == 69 and first["pages"] == 23
    corpus = Corpus.load(tmp_path / "first/corpus.jsonl")
    queries = load_queries(tmp_path / "first/queries.jsonl")
    assert all(len(p.text.split()) == 32 for p in corpus.passages)
    assert all(len(corpus.children(n.id)) <= 3 for n in corpus.nodes.values() if n.kind in {"root", "group"})
    assert len({next(iter(q.relevant_ids)) for q in queries}) == 10
    for query in queries:
        answer = query.answers[0]
        gold = next(iter(query.relevant_ids))
        assert answer in corpus.nodes[gold].text
        assert sum(answer in p.text for p in corpus.passages) == 1
        assert answer not in query.text
        assert all(answer not in n.text for n in corpus.nodes.values() if n.kind != "passage")
        assert query.metadata["purpose"] == "performance stress only"
    assert (tmp_path / "first/corpus.jsonl").read_bytes() == (tmp_path / "second/corpus.jsonl").read_bytes()


def test_stress_changes_with_seed_and_rejects_invalid_sizes(tmp_path):
    a = generate(tmp_path / "a", pages=2, passages_per_page=2, queries=2, seed=1)
    b = generate(tmp_path / "b", pages=2, passages_per_page=2, queries=2, seed=2)
    assert a["corpus_sha256"] != b["corpus_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        generate(tmp_path / "a", pages=2, passages_per_page=2, queries=2)
    with pytest.raises(ValueError, match="cannot exceed"):
        generate(tmp_path / "bad", pages=1, passages_per_page=1, queries=2)
    with pytest.raises(ValueError, match="words_per_passage"):
        generate(tmp_path / "short", words_per_passage=5)
