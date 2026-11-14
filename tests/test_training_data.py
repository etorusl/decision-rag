import pytest

from decision_rag.schema import Corpus, Node, Query, read_jsonl
from decision_rag.training_data import make_training_data


class TokenCounter:
    def count_tokens(self, text):
        return len(text.split())


def corpus():
    return Corpus(
        [
            Node("root", "Corpus", "root"),
            Node("center", "Help center", "group", "root"),
            Node("a", "Guide A", "page", "center", "Guide A", 0),
            Node("b", "Guide B", "page", "center", "Guide B", 1),
            Node("c", "Guide C", "page", "center", "Guide C", 2),
            Node("p-a", "alpha evidence", "passage", "a", order=0),
            Node("p-copy", "  ALPHA   evidence ", "passage", "a", order=1),
            Node("p-b", "beta evidence", "passage", "b"),
            Node("p-c", "gamma irrelevant", "passage", "c"),
        ]
    )


def train_query(qid, pid="p-a", conversation=None, turn=1):
    return Query(
        qid,
        "Find evidence",
        {pid: 1.0},
        conversation,
        turn,
        answers=["TARGET RESPONSE MUST NOT ENTER PROMPT"],
        metadata={"split": "train"},
    )


def records(path):
    return list(read_jsonl(path))


def test_training_and_monitoring_keep_complete_dialogues_disjoint(tmp_path):
    queries = [
        train_query("a1", conversation="dialogue-a"),
        train_query("a2", conversation="dialogue-a", turn=2),
        train_query("b1", "p-b", "dialogue-b"),
        train_query("b2", "p-b", "dialogue-b", 2),
    ]
    make_training_data(
        corpus(), queries, tmp_path / "first", TokenCounter(), search_device="cpu", monitor_fraction=0.5, gold_page_probability=1.0
    )
    train = records(tmp_path / "first/train.jsonl")
    monitor = records(tmp_path / "first/monitor.jsonl")
    train_units = {r["metadata"]["unit"] for r in train}
    monitor_units = {r["metadata"]["unit"] for r in monitor}
    assert train_units.isdisjoint(monitor_units)
    assert train_units | monitor_units == {"dialogue-a", "dialogue-b"}
    for subset in (train, monitor):
        qids = {r["metadata"]["query_id"] for r in subset}
        assert qids in ({"a1", "a2"}, {"b1", "b2"})
        assert all("TARGET RESPONSE MUST NOT ENTER PROMPT" not in r["context"] for r in subset)
        assert all(len(r["questions"]) == len(r["labels"]) == len(r["options"]) for r in subset)
    make_training_data(
        corpus(),
        queries,
        tmp_path / "second",
        TokenCounter(), search_device="cpu",
        monitor_fraction=0.5,
        gold_page_probability=1.0,
    )
    assert (tmp_path / "first/train.jsonl").read_bytes() == (tmp_path / "second/train.jsonl").read_bytes()
    assert (tmp_path / "first/monitor.jsonl").read_bytes() == (tmp_path / "second/monitor.jsonl").read_bytes()


def test_identical_gold_text_is_never_negative(tmp_path):
    queries = [train_query("a", conversation="one"), train_query("b", conversation="two")]
    make_training_data(
        corpus(), queries, tmp_path, TokenCounter(), search_device="cpu", monitor_fraction=0.5, gold_page_probability=1.0
    )
    all_records = records(tmp_path / "train.jsonl") + records(tmp_path / "monitor.jsonl")
    passage_records = [r for r in all_records if r["metadata"]["stage"] == "passages"]
    assert passage_records
    for row in passage_records:
        ids = row["metadata"]["candidate_ids"]
        assert "p-copy" not in ids
        assert row["labels"][ids.index("p-a")] == 1
        assert all(label == int(pid == "p-a") for pid, label in zip(ids, row["labels"]))


@pytest.mark.parametrize("split", ["validation", "dev", "test", None])
def test_rejects_evaluation_queries_before_writing(tmp_path, split):
    queries = [Query("q", "question", {"p-a": 1}, metadata={"split": split})]
    with pytest.raises(ValueError, match="train"):
        make_training_data(corpus(), queries, tmp_path, TokenCounter(), search_device="cpu")
    assert not (tmp_path / "train.jsonl").exists()


def test_cqa_monitor_split_uses_source_pages_including_questions_without_evidence(tmp_path):
    queries = [
        Query("a1", "alpha?", {"p-a": 1}, metadata={"split": "train", "gold_page_ids": ["a"]}),
        Query("a2", "unanswerable on alpha?", {}, metadata={"split": "train", "gold_page_ids": ["a"]}),
        Query("b1", "beta?", {"p-b": 1}, metadata={"split": "train", "gold_page_ids": ["b"]}),
        Query("b2", "unanswerable on beta?", {}, metadata={"split": "train", "gold_page_ids": ["b"]}),
    ]
    make_training_data(
        corpus(),
        queries,
        tmp_path,
        TokenCounter(), search_device="cpu",
        dataset="conditionalqa",
        monitor_fraction=0.5,
        gold_page_probability=1.0,
    )
    train, monitor = records(tmp_path / "train.jsonl"), records(tmp_path / "monitor.jsonl")
    for subset in (train, monitor):
        ids = {r["metadata"]["query_id"] for r in subset}
        assert ids in ({"a1", "a2"}, {"b1", "b2"})
    unlabeled = [r for r in train + monitor if r["metadata"]["query_id"] in {"a2", "b2"}]
    assert len(unlabeled) == 2
    assert all(r["metadata"]["stage"] == "pages" for r in unlabeled)
    assert all(sum(r["labels"]) == 1 for r in unlabeled)


@pytest.mark.parametrize("invalid", ["duplicate", "unknown"])
def test_training_rejects_duplicate_queries_and_unknown_gold(tmp_path, invalid):
    queries = [train_query("a", conversation="one"), train_query("b", "p-b", "two")]
    if invalid == "duplicate":
        queries.append(queries[0])
    else:
        queries.append(train_query("bad", "missing", "three"))
    with pytest.raises(ValueError):
        make_training_data(corpus(), queries, tmp_path, TokenCounter(), search_device="cpu", monitor_fraction=0.5)
    assert not (tmp_path / "train.jsonl").exists()


def test_training_requires_at_least_two_groups(tmp_path):
    with pytest.raises(ValueError, match="two"):
        make_training_data(corpus(), [train_query("q", conversation="one")], tmp_path, TokenCounter(), search_device="cpu")
