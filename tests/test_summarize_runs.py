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
