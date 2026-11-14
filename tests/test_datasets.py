import json
import zipfile
from pathlib import Path

import pytest

from decision_rag.datasets import _unzip, prepare
from decision_rag.schema import Corpus, load_queries


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_conditionalqa_preserves_element_ids_and_duplicate_evidence(tmp_path):
    source = tmp_path / "raw"
    documents = [
        {
            "title": "Benefits",
            "url": "https://example.test/benefits",
            "contents": [
                "<h1>Overview</h1>",
                "<h2>Eligibility</h2>",
                "<p>You qualify.</p>",
                "<h3>Exceptions</h3>",
                "<p>You qualify.</p>",
            ],
        }
    ]
    dump(source / "documents.json", documents)
    dump(
        source / "dev.json",
        [
            {
                "id": "dev-0",
                "url": documents[0]["url"],
                "scenario": "My situation.",
                "question": "Eligible?",
                "evidences": ["<p>You qualify.</p>", "<h2>Eligibility</h2>"],
                "answers": [["yes", []]],
            },
            {
                "id": "dev-1",
                "url": documents[0]["url"],
                "scenario": "Unknown",
                "question": "Other?",
                "evidences": [],
                "not_answerable": True,
            },
        ],
    )
    out = tmp_path / "out"
    info = prepare("conditionalqa", out, source, limit=1)
    assert info["passages"] == 3
    assert info["queries"] == 1 and info["queries_before_limit"] == 2
    query = load_queries(out / "queries.jsonl")[0]
    assert query.relevant_ids == {"P0:e2": 1.0, "P0:e4": 1.0}
    assert query.answers == ["yes"]
    corpus = Corpus.load(out / "corpus.jsonl")
    assert corpus.nodes["P0:e3"].text == "Exceptions"
    assert "Eligible?" not in corpus.nodes["P0"].text
    assert len(info["sources"][0]["sha256"]) == 64


def test_conditionalqa_unknown_evidence_is_error(tmp_path):
    dump(tmp_path / "documents.json", [{"title": "A", "url": "a", "contents": ["<p>x</p>"]}])
    dump(
        tmp_path / "dev.json",
        [{"id": "q", "url": "a", "scenario": "s", "question": "q", "evidences": ["<p>not present</p>"]}],
    )
    with pytest.raises(ValueError, match="Unmapped ConditionalQA"):
        prepare("conditionalqa", tmp_path / "out", tmp_path)


def test_multidoc2dial_joins_parts_and_excludes_target_from_history(tmp_path):
    def document(name, text):
        return {
            "doc_id": name,
            "title": name,
            "doc_text": text,
            "spans": {
                "1": {
                    "id_sp": "1",
                    "id_sec": "s1",
                    "title": "Section",
                    "text_sp": text[:5],
                    "parent_titles": [],
                    "start_sp": 0,
                    "end_sp": 5,
                },
                "2": {
                    "id_sp": "2",
                    "id_sec": "s2",
                    "title": "Section",
                    "text_sp": text[5:],
                    "parent_titles": [],
                    "start_sp": 5,
                    "end_sp": len(text),
                },
            },
        }

    docs = {
        "doc_data": {
            "ssa": {
                "Guide#1": document("Guide#1", "first evidence"),
                "Guide#2": document("Guide#2", "other evidence"),
            }
        }
    }
    dump(tmp_path / "multidoc2dial_doc.json", docs)
    refs = [{"doc_id": "Guide#1", "id_sp": "1"}, {"doc_id": "Guide#1", "id_sp": "2"}]
    turns = [
        {"role": "user", "turn_id": 1, "utterance": "Question", "references": refs, "da": "query"},
        {
            "role": "agent",
            "turn_id": 2,
            "utterance": "SECRET ANSWER",
            "references": refs,
            "da": "respond_solution",
        },
    ]
    dump(
        tmp_path / "multidoc2dial_dial_validation.json",
        {"dial_data": {"ssa": [{"dial_id": "dialogue", "turns": turns}]}},
    )
    out = tmp_path / "out"
    manifest = prepare("multidoc2dial", out, tmp_path)
    assert manifest["pages"] == 1 and manifest["passages"] == 2
    query = load_queries(out / "queries.jsonl")[0]
    assert query.id == "dialogue_1"
    assert query.history == [] and query.context() == "Question"
    assert query.relevant_ids == {"0": 1.0}
    assert query.answers == ["SECRET ANSWER"]
    assert "SECRET ANSWER" not in (out / "corpus.jsonl").read_text()


def test_beir_preserves_qrels_and_full_corpus(tmp_path):
    jsonl(
        tmp_path / "corpus.jsonl",
        [{"_id": "a", "title": "A", "text": "one"}, {"_id": "b", "title": "B", "text": "two"}],
    )
    jsonl(tmp_path / "queries.jsonl", [{"_id": "q", "text": "question"}, {"_id": "unlabeled", "text": "x"}])
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\tb\t2\n")
    out = tmp_path / "out"
    result = prepare("beir", out, tmp_path, limit=1)
    assert result["passages"] == 2
    assert load_queries(out / "queries.jsonl")[0].relevant_ids == {"b": 2.0}


def test_dapr_original_chunks_and_graded_judgments(tmp_path):
    jsonl(
        tmp_path / "corpus.jsonl",
        [
            {
                "id": "doc",
                "title": "Title",
                "chunks": [{"id": "doc-0", "text": "zero"}, {"id": "doc-1", "text": "one"}],
            }
        ],
    )
    jsonl(
        tmp_path / "test.jsonl",
        [
            {
                "query": {"id": "q", "text": "question"},
                "judged_chunks": [{"chunk": {"id": "doc-1"}, "judgement": 2}],
            }
        ],
    )
    out = tmp_path / "out"
    prepare("nq-dapr", out, tmp_path)
    corpus = Corpus.load(out / "corpus.jsonl")
    assert corpus.page_id("doc-1") == "dapr-page:doc"
    assert load_queries(out / "queries.jsonl")[0].relevant_ids == {"doc-1": 2.0}


def test_rejects_missing_gold_and_zip_traversal(tmp_path):
    jsonl(tmp_path / "corpus.jsonl", [{"_id": "a", "text": "one"}])
    jsonl(tmp_path / "queries.jsonl", [{"_id": "q", "text": "question"}])
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\tmissing\t1\n")
    with pytest.raises(ValueError, match="missing passages"):
        prepare("beir", tmp_path / "out", tmp_path)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped", "bad")
    with pytest.raises(ValueError, match="Unsafe archive"):
        _unzip(archive, tmp_path / "extracted")
    assert not (tmp_path / "escaped").exists()


def test_mtrag_preserves_dialogue_order_and_qrels(tmp_path):
    jsonl(
        tmp_path / "corpora/document_level/cloud.jsonl",
        [{"document_id": "ibmcld_00001", "text": "A service\n\nGetting started\n\nUse the console."}],
    )
    jsonl(
        tmp_path / "corpora/passage_level/cloud.jsonl",
        [
            {
                "id": "ibmcld_00001-0-42",
                "text": "Getting started\n\nUse the console.",
                "url": "https://cloud.ibm.com/docs/service?topic=service-start",
            }
        ],
    )
    jsonl(
        tmp_path / "mtrag-human/generation_tasks/reference.jsonl",
        [
            {
                "task_id": "q",
                "conversation_id": "c",
                "turn": 2,
                "Collection": "ibmcloud",
                "input": [
                    {"speaker": "user", "text": "First"},
                    {"speaker": "agent", "text": "Earlier"},
                    {"speaker": "user", "text": "Now?"},
                ],
                "targets": [{"text": "Use the console"}],
            }
        ],
    )
    path = tmp_path / "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv"
    path.parent.mkdir(parents=True)
    path.write_text("query-id\tcorpus-id\tscore\nq\tibmcld_00001-0-42\t1\n")
    out = tmp_path / "out"
    prepare("mtrag", out, tmp_path)
    query = load_queries(out / "queries.jsonl")[0]
    assert query.history[-1] == {"role": "assistant", "content": "Earlier"}
    assert query.text == "Now?" and query.turn_id == 2
    assert query.relevant_ids == {"ibmcld_00001-0-42": 1.0}
