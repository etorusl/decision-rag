"""Reporter tests exercise only stdlib JSON/CSV processing, without model libraries."""

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "summarize_runs", Path(__file__).parents[1] / "scripts/summarize_runs.py"
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_summary(directory, metric=0.5):
    directory.mkdir(parents=True)
    path = directory / "summary.json"
    path.write_text(
        json.dumps(
            {
                "quality": {"recall@1": metric, "recall@10": 1.0},
                "latency": {
                    "queries": 4,
                    "p50_seconds": 0.2,
                    "p95_seconds": 0.4,
                    "serial_queries_per_second": 4.0,
                },
                "labeled_queries": 3,
                "unlabeled_queries": 1,
                "warmup_queries": 2,
                "work": {"input_tokens": 400, "forward_passes": 8, "requests": 8},
                "gpu_peak_allocated_bytes": 2 * 2**30,
                "gpu_peak_reserved_bytes": 3 * 2**30,
                "environment": {"git_commit": "abcdef", "gpu": "H100", "git_dirty": False},
                "inputs": {"data/queries.jsonl": "a" * 64, "data/corpus.jsonl": "b" * 64},
                "config": {"backend": "test"},
                "timing_scope": "serial retrieval",
            }
        )
    )
    return path


def test_reporter_discovers_unique_runs_and_flattens_measurements(tmp_path):
    left = write_summary(tmp_path / "left")
    right = write_summary(tmp_path / "nested/right", 0.75)
    files = reporter.find_summaries([tmp_path, left, tmp_path / "nested"])
    assert files == sorted([left.resolve(), right.resolve()])
    rows = [reporter.flatten_summary(path) for path in files]
    assert rows[0]["gpu_peak_allocated_gib"] == 2.0
    assert rows[0]["work.input_tokens_per_query"] == 100.0
    assert rows[0]["work.forward_passes_per_query"] == 2.0
    csv_rows = list(csv.DictReader(io.StringIO(reporter.csv_text(rows))))
    assert len(csv_rows) == 2
    assert csv_rows[0]["quality.recall@1"] == "0.5"
    markdown = reporter.markdown_text(rows)
    assert "50.00%" in markdown and "75.00%" in markdown
    assert "p95 s" in markdown and "Input tokens/query" in markdown and "H100" in markdown
    assert "aaaaaaaaaaaa" in markdown


def test_reporter_cli_writes_outputs_and_preserves_source(tmp_path, capsys):
    source = write_summary(tmp_path / "input")
    original = source.read_bytes()
    reporter.main([str(source), "--output", str(tmp_path / "report/comparison")])
    assert (tmp_path / "report/comparison.csv").exists()
    assert (tmp_path / "report/comparison.md").exists()
    assert "Retrieval run comparison" in capsys.readouterr().out
    assert source.read_bytes() == original


def test_reporter_exposes_search_device_and_component_latency(tmp_path):
    source = write_summary(tmp_path / 'gpu')
    summary = json.loads(source.read_text())
    summary['config']['runtime'] = {'search_runtime': {
        'device': 'cuda:0', 'precision': 'float32', 'index_bytes': 2**30}}
    summary['component_latency'] = {'query_embedding': {'mean_seconds': .025},
                                    'vector_search': {'mean_seconds': .0005}}
    summary['gpu_telemetry'] = {'samples': 10, 'peak_device_used_bytes': 40 * 2**30}
    source.write_text(json.dumps(summary))
    row = reporter.flatten_summary(source)
    assert row['search_device'] == 'cuda:0'
    assert row['search_precision'] == 'float32'
    assert row['search_index_gib'] == 1
    assert row['mean_query_embedding_ms'] == 25
    assert row['mean_vector_search_ms'] == .5
    assert row['gpu_device_peak_used_gib'] == 40
    markdown = reporter.markdown_text([row])
    assert 'Search device' in markdown and 'cuda:0' in markdown
    assert 'Query embedding ms' in markdown and 'Vector search ms' in markdown
    assert 'Device peak GiB' in markdown
    # Pre-GPU-search runs still identify their known CPU search in comparisons.
    summary['config']['runtime'] = {'search': 'CPU exact dot product of normalized embeddings'}
    source.write_text(json.dumps(summary))
    legacy = reporter.flatten_summary(source)
    assert legacy['search_device'] == 'cpu'
    assert legacy['search_precision'] == ''


def test_reporter_handles_missing_fields_and_invalid_input(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"quality": {}, "latency": {}}))
    row = reporter.flatten_summary(path)
    assert row["work.input_tokens_per_query"] is None
    assert "—" in reporter.markdown_text([row])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No summary"):
        reporter.find_summaries([tmp_path / "empty"])
    path.write_text("{}")
    with pytest.raises(ValueError, match="Not an evaluation"):
        reporter.flatten_summary(path)


@pytest.mark.parametrize('choice_k,run_type', [(0, 'naive binary'), (20, 'naive choice')])
def test_decision_rows_include_actual_model_in_every_table_and_csv(tmp_path, choice_k, run_type):
    source = write_summary(tmp_path / 'pages-3')
    summary = json.loads(source.read_text())
    summary['config'] = {
        'method': 'decision', 'backend': 'vllm-logits:Qwen/Qwen3-8B',
        'model': 'Qwen/Qwen3-8B', 'model_revision': 'a' * 40,
        'retrieval': {'choice_k': choice_k, 'page_budget': 3},
        'args': {'backend': 'vllm', 'model': 'unused-user-alias', 'choice_k': 0},
        'runtime': {'model': 'Qwen/Qwen3-8B', 'runtime': 'vllm-http'},
    }
    source.write_text(json.dumps(summary))
    row = reporter.flatten_summary(source)
    assert row['run_type'] == run_type
    assert row['model'] == 'Qwen/Qwen3-8B'
    assert row['model_revision'] == 'a' * 40
    assert row['embedding_model'] == row['reranker_model'] == ''
    markdown = reporter.markdown_text([row])
    for table in ('Quality', 'Efficiency', 'Protocol'):
        section = markdown.split(f'## {table}\n', 1)[1].split('\n## ', 1)[0]
        assert f'{run_type} — Qwen/Qwen3-8B' in section
        assert f'[{source.parent.parent.name}/pages-3]' in section
    assert f'R1: `{source.parent}`' in markdown
    exported = next(csv.DictReader(io.StringIO(reporter.csv_text([row]))))
    assert exported['model'] == 'Qwen/Qwen3-8B'
    assert exported['run_type'] == run_type
    assert exported['run'] == str(source.parent)


@pytest.mark.parametrize('embedding,reranker', [
    ('Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Reranker-8B'),
    ('BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3'),
])
@pytest.mark.parametrize('method', ['dense', 'dense-rerank'])
def test_dense_rows_name_used_embedder_and_reranker(tmp_path, embedding, reranker, method):
    source = write_summary(tmp_path / 'dense-run')
    summary = json.loads(source.read_text())
    summary['config'] = {
        'method': method, 'engine': 'vllm', 'model': embedding,
        # This CLI default exists for pure dense too; it must not appear as a used model.
        'reranker_model': reranker,
        'runtime': {'embedding': {'model': embedding, 'revision': 'e' * 40}},
    }
    if method == 'dense-rerank':
        summary['config']['runtime']['reranker'] = {'model': reranker, 'revision': 'f' * 40}
    source.write_text(json.dumps(summary))
    row = reporter.flatten_summary(source)
    assert row['embedding_model'] == embedding
    assert row['embedding_revision'] == 'e' * 40
    assert row['model'] == ''
    markdown = reporter.markdown_text([row])
    for table in ('Quality', 'Efficiency', 'Protocol'):
        section = markdown.split(f'## {table}\n', 1)[1].split('\n## ', 1)[0]
        assert f'embedder: {embedding}' in section
        assert f'reranker: {reranker if method == "dense-rerank" else "none"}' in section
    if method == 'dense':
        assert row['reranker_model'] == row['reranker_revision'] == ''
        assert reranker not in markdown
    else:
        assert row['reranker_model'] == reranker
        assert row['reranker_revision'] == 'f' * 40
    exported = next(csv.DictReader(io.StringIO(reporter.csv_text([row]))))
    assert exported['embedding_model'] == embedding
    assert exported['reranker_model'] == (reranker if method == 'dense-rerank' else '')


def test_hf_runtime_models_and_trained_head_checkpoint_are_preserved():
    dense = reporter.model_identity({
        'method': 'dense-rerank', 'runtime': {'engine': 'sentence-transformers',
            'embedding_model': 'custom/embedder', 'embedding_revision': 'embedding-sha',
            'reranker_model': 'custom/reranker'},
    })
    assert dense['embedding_model'] == 'custom/embedder'
    assert dense['reranker_model'] == 'custom/reranker'
    assert dense['embedding_revision'] == 'embedding-sha'
    head = reporter.model_identity({
        'method': 'decision', 'backend': 'trained-decision-head:md2d',
        'model': 'Qwen/Qwen3-8B-Base', 'model_revision': 'base-sha', 'checkpoint': 'checkpoints/md2d',
        'retrieval': {'choice_k': 0}, 'args': {'backend': 'head', 'model': 'unused-model'},
    })
    assert head['run_label'] == 'trained head binary — Qwen/Qwen3-8B-Base; checkpoint: checkpoints/md2d'
    assert head['model_revision'] == 'base-sha'
    assert head['checkpoint'] == 'checkpoints/md2d'
    legacy_head = reporter.model_identity({
        'method': 'decision', 'backend': 'trained-decision-head:md2d',
        'args': {'backend': 'head', 'model': 'ignored-by-head-loader', 'checkpoint': 'checkpoints/md2d'},
    })
    assert legacy_head['model'] == ''
    assert 'model unknown' in legacy_head['run_label']
    assert 'ignored-by-head-loader' not in legacy_head['run_label']


def test_bm25_and_unknown_runs_do_not_claim_unused_neural_models(tmp_path):
    bm25 = reporter.model_identity({'method': 'bm25', 'model': 'unused', 'reranker_model': 'unused-reranker'})
    assert bm25['run_label'] == 'BM25 — no neural model'
    assert bm25['model'] == bm25['embedding_model'] == bm25['reranker_model'] == ''
    source = write_summary(tmp_path / 'naive-binary-Qwen-Qwen3-8B')
    row = reporter.flatten_summary(source)
    assert row['model'] == '' and row['method'] == 'unknown'
    assert row['run_label'] == 'unknown — model unknown'
    legacy = reporter.model_identity({'backend': 'hf-logits:org/archived-model'})
    assert legacy['model'] == 'org/archived-model'
    assert legacy['run_type'] == 'naive'
