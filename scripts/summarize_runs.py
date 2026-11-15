#!/usr/bin/env python3
"""Collect evaluation summaries into CSV and Markdown; no dependency needed for reporting."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

QUALITY = ("hit@1", "recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@10")
RECALL = ("recall@1", "recall@5", "recall@10", "recall@20")
MULTIDOC2DIAL_DOMAINS = ("DMV", "SSA", "VA", "Student Aid", "Unassigned")
SUMMARY_NAME = "summary.json"
TEXT_RECALL_SIDECAR = "summary.text-recall.json"


def model_identity(config):
    """Read recorded model identities, without guessing from directory names or current defaults."""
    args = config.get('args') or config
    runtime = config.get('runtime') or {}
    retrieval = config.get('retrieval') or {}
    backend = config.get('backend') or args.get('backend') or ''
    method = config.get('method') or args.get('method') or 'unknown'
    if method == 'unknown' and backend.startswith(('vllm-logits:', 'hf-logits:', 'trained-decision-head:', 'jev:')):
        method = 'decision'
    identity = {
        'method': method, 'backend': backend,
        'engine': config.get('engine') or runtime.get('engine') or runtime.get('runtime') or '',
        'model': '', 'model_revision': '', 'embedding_model': '', 'embedding_revision': '',
        'reranker_model': '', 'reranker_revision': '', 'checkpoint': '',
    }
    if method in {'dense', 'dense-rerank'}:
        embedding = runtime.get('embedding') or {}
        identity['embedding_model'] = (embedding.get('model') or runtime.get('embedding_model')
                                       or args.get('embedding_model') or args.get('model') or '')
        identity['embedding_revision'] = (embedding.get('revision') or runtime.get('embedding_revision')
                                          or args.get('embedding_revision') or '')
        if method == 'dense-rerank':
            reranker = runtime.get('reranker') or {}
            identity['reranker_model'] = (reranker.get('model') or runtime.get('reranker_model')
                                          or args.get('reranker_model') or '')
            identity['reranker_revision'] = (reranker.get('revision') or runtime.get('reranker_revision')
                                             or args.get('reranker_revision') or '')
        run_type = 'dense + rerank' if method == 'dense-rerank' else 'dense'
        reranker_label = (identity['reranker_model'] or 'unknown') if method == 'dense-rerank' else 'none'
        description = (f"{run_type} — embedder: {identity['embedding_model'] or 'unknown'}; "
                       f"reranker: {reranker_label}")
    elif method == 'decision':
        head = backend.startswith('trained-decision-head:') or args.get('backend') == 'head'
        identity['model'] = config.get('model') or runtime.get('base_model') or runtime.get('model') or ''
        if head:
            identity['checkpoint'] = config.get('checkpoint') or runtime.get('checkpoint') or args.get('checkpoint') or ''
        else:
            identity['model'] = identity['model'] or args.get('model') or ''
            if not identity['model'] and backend.startswith(('vllm-logits:', 'hf-logits:')):
                identity['model'] = backend.split(':', 1)[1]
            elif not identity['model'] and backend.startswith('jev:'):
                identity['model'] = backend.split(':', 2)[-1]
        identity['model_revision'] = (config.get('model_revision') or runtime.get('base_revision')
                                      or runtime.get('tokenizer_revision') or args.get('model_revision') or '')
        choice_k = retrieval.get('choice_k', args.get('choice_k'))
        scoring = 'choice' if choice_k is not None and choice_k > 0 else 'binary' if choice_k == 0 else ''
        if head:
            family = 'trained head'
        elif backend.startswith(('vllm-logits:', 'hf-logits:')) or args.get('backend') in {'vllm', 'hf'}:
            family = 'naive'
        else:
            family = 'decision'
        run_type = ' '.join(part for part in (family, scoring) if part)
        description = f"{run_type} — {identity['model'] or 'model unknown'}"
        if identity['checkpoint']:
            description += f"; checkpoint: {identity['checkpoint']}"
    elif method == 'bm25':
        identity['engine'] = 'bm25'
        run_type, description = 'BM25', 'BM25 — no neural model'
    else:
        run_type, description = method, f'{method} — model unknown'
    return {**identity, 'run_type': run_type, 'run_label': description}


def find_summaries(paths):
    directories = set()
    names = {SUMMARY_NAME, TEXT_RECALL_SIDECAR}
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Results path does not exist: {path}")
        if path.is_file():
            if path.name not in names:
                raise ValueError(f"Expected {SUMMARY_NAME} or {TEXT_RECALL_SIDECAR}, got {path}")
            directories.add(path.parent.resolve())
        else:
            directories.update(p.parent.resolve() for p in path.rglob("summary*.json") if p.name in names and p.is_file())
    if not directories:
        raise ValueError("No summary.json files found")
    return sorted(directory / (SUMMARY_NAME if (directory / SUMMARY_NAME).is_file() else TEXT_RECALL_SIDECAR)
                  for directory in directories)


def _read_summary(path):
    summary = path.with_name(SUMMARY_NAME)
    sidecar = path.with_name(TEXT_RECALL_SIDECAR)
    if path == sidecar and summary.is_file():
        path = summary
    data = json.loads(path.read_text(encoding="utf-8"))
    if sidecar.is_file():
        enriched = data if path == sidecar else json.loads(sidecar.read_text(encoding="utf-8"))
        data["text_recall"] = enriched["text_recall"]
    return data


def _flatten_text_recall(text_recall):
    row = {f"text_recall.{key}": text_recall.get(key) for key in (
        "ranking_limit", "max_saved_ids", "limited_to_saved_ranking",
    )}
    for series in ("id_quality", "quality"):
        row.update({f"text_recall.{series}.{metric}": value for metric, value in text_recall.get(series, {}).items()})
    for domain, scores in text_recall.get("domains", {}).items():
        prefix = f"text_recall.domains.{domain}"
        for count in ("queries", "labeled_queries"):
            row[f"{prefix}.{count}"] = scores.get(count)
        for series in ("id_quality", "quality"):
            row.update({f"{prefix}.{series}.{metric}": value for metric, value in scores.get(series, {}).items()})
    return row


def flatten_summary(path):
    path = Path(path)
    data = _read_summary(path)
    if "quality" not in data or "latency" not in data:
        raise ValueError(f"Not an evaluation summary: {path}")
    latency, work = data["latency"], data.get("work", {})
    count = latency.get("queries", 0)
    config = data.get("config", {})
    runtime = config.get('runtime', {})
    execution = data.get('execution', {})
    throughput = data.get('throughput', {})
    search = config.get('runtime', {}).get('search_runtime', {})
    legacy_search = config.get('runtime', {}).get('search', '')
    env = data.get("environment", {})
    query_hashes = sorted(v for k, v in data.get("inputs", {}).items() if Path(k).name == "queries.jsonl")
    corpus_hashes = sorted(v for k, v in data.get("inputs", {}).items() if Path(k).name == "corpus.jsonl")
    row = {
        "run": str(path.parent),
        **model_identity(config),
        "vllm_server_version": runtime.get('server_version') or runtime.get('embedding', {}).get('server_version', ''),
        "execution_mode": execution.get('mode', 'latency'),
        "query_concurrency": execution.get('query_concurrency', 1),
        "effective_query_concurrency": execution.get('effective_query_concurrency', 1),
        "observed_query_concurrency": execution.get('observed_query_concurrency'),
        "throughput.queries_per_second": throughput.get('queries_per_second'),
        "throughput.wall_seconds": throughput.get('wall_seconds'),
        "queries": count,
        "labeled_queries": data.get("labeled_queries", 0),
        "unlabeled_queries": data.get("unlabeled_queries", 0),
        "warmup_queries": data.get("warmup_queries", 0),
        "dedup_text": data.get("dedup_text", False),
        "query_sha256": ";".join(query_hashes),
        "corpus_sha256": ";".join(corpus_hashes),
        "git_commit": env.get("git_commit", ""),
        "git_dirty": env.get("git_dirty", ""),
        "gpu": env.get("gpu", ""),
        "timing_scope": data.get("timing_scope", ""),
        "search_device": search.get('device', 'cpu' if legacy_search.startswith('CPU exact') else ''),
        "search_precision": search.get('precision', ''),
        "search_index_gib": search['index_bytes'] / 2**30 if search.get('index_bytes') is not None else None,
        "config_json": json.dumps(config, sort_keys=True),
    }
    row.update({f"quality.{key}": value for key, value in data["quality"].items()})
    if data.get("text_recall") is not None:
        row.update(_flatten_text_recall(data["text_recall"]))
    row.update({f"latency.{key}": value for key, value in latency.items() if key != "queries"})
    for component in ('query_embedding', 'vector_search'):
        seconds = data.get('component_latency', {}).get(component, {}).get('mean_seconds')
        row[f'mean_{component}_ms'] = seconds * 1000 if seconds is not None else None
    for key in ("input_tokens", "forward_passes", "requests", "prompt_requests", "completion_tokens", "cached_tokens"):
        row[f"work.{key}"] = work.get(key, 0)
        value = work.get(key, 0)
        row[f"work.{key}_per_query"] = value / count if count and value is not None else None
    for kind in ("allocated", "reserved"):
        value = data.get(f"gpu_peak_{kind}_bytes")
        row[f"gpu_peak_{kind}_gib"] = value / 2**30 if value is not None else None
    telemetry = data.get("gpu_telemetry", {})
    row["gpu_device_peak_used_gib"] = telemetry.get("peak_device_used_bytes", 0) / 2**30 if telemetry.get("samples") else None
    row['gpu_mean_utilization_percent'] = telemetry.get('mean_utilization_percent')
    row["inference_process"] = data.get("inference_process", "benchmark_process")
    return row


def csv_text(rows):
    if not rows:
        raise ValueError("No summary rows")
    leading = ["run", "run_type", "model", "embedding_model", "reranker_model", "execution_mode", "query_concurrency",
               "queries", "labeled_queries", "unlabeled_queries"]
    remaining = set().union(*(row.keys() for row in rows)) - set(leading)
    fields = list(leading)
    legacy_recall_fields = {f"quality.{metric}" for metric in RECALL}
    for field in sorted(key for key in remaining if not key.startswith("text_recall.")):
        fields.append(field)
        remaining.remove(field)
        if field in legacy_recall_fields:
            metric = field.removeprefix("quality.")
            for series in ("id_quality", "quality"):
                extra = f"text_recall.{series}.{metric}"
                if extra in remaining:
                    fields.append(extra)
                    remaining.remove(extra)
    fields.extend(sorted(remaining))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _cell(value, *, percentage=False):
    if value is None or value == "":
        return "—"
    if percentage and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.3g}" if 0 < abs(value) < 0.001 else f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows, columns, percentages=()):
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_cell(row.get(key), percentage=key in percentages) for key, _ in columns) + " |"
        )
    return "\n".join(lines)


def _recall_columns(id_prefix, text_prefix):
    columns = []
    for metric in RECALL:
        columns.extend([(id_prefix + metric, f"ID {metric}"), (text_prefix + metric, f"Text {metric}")])
    return columns


def _saved_ranking(row):
    if "text_recall.limited_to_saved_ranking" not in row:
        return None
    if not row["text_recall.limited_to_saved_ranking"]:
        return "Full ranking"
    limit = row.get("text_recall.ranking_limit")
    maximum = row.get("text_recall.max_saved_ids")
    if limit is not None:
        depth = f"Top {limit} IDs"
        return f"{depth} (at most {maximum} saved)" if maximum is not None and maximum < limit else depth
    return f"At most {maximum} IDs saved" if maximum is not None else "Saved IDs"


def _domain_rows(rows):
    domains = []
    for row in rows:
        for domain in MULTIDOC2DIAL_DOMAINS:
            prefix = f"text_recall.domains.{domain}."
            fields = {key.removeprefix(prefix): value for key, value in row.items() if key.startswith(prefix)}
            if fields:
                domains.append({"run_label": row["run_label"], "domain": domain, **fields})
    return domains


def markdown_text(rows):
    quality_columns = [("run_label", "Run"), ("queries", "Queries"), ("labeled_queries", "Labeled")]
    display = [
        {
            **row,
            "report_id": f'R{index}',
            "run_label": (f"R{index}: {row.get('run_label', 'unknown — model unknown')} "
                          f"({row.get('execution_mode', 'latency')}, C={row.get('query_concurrency', 1)}) "
                          f"[{'/'.join(Path(row['run']).parts[-2:])}]"),
            "query_short": row.get("query_sha256", "")[:12],
            "corpus_short": row.get("corpus_sha256", "")[:12],
        }
        for index, row in enumerate(rows, 1)
    ]
    quality_percentages = [f"quality.{m}" for m in QUALITY]
    recall_notes = []
    if any("text_recall.limited_to_saved_ranking" in row for row in rows):
        recall_columns = _recall_columns("report_id_quality.", "text_recall.quality.")
        quality_columns += [("quality.hit@1", "hit@1"), *recall_columns,
                            ("quality.ndcg@10", "ndcg@10"), ("quality.mrr@10", "mrr@10"),
                            ("saved_ranking", "Saved ranking")]
        quality_percentages.extend(key for key, _ in recall_columns)
        for row in display:
            row["saved_ranking"] = _saved_ranking(row)
            for metric in RECALL:
                legacy = row.get(f"quality.{metric}") if not row.get("dedup_text") else None
                row[f"report_id_quality.{metric}"] = row.get(f"text_recall.id_quality.{metric}", legacy)
        recall_notes.extend([
            ("ID Recall is the primary metric. Text Recall counts the first occurrence of each identical text "
             "after whitespace normalization, with case preserved."),
            ("Text Recall uses the saved IDs shown in each row. A saved top-20 ranking can contain fewer than "
             "20 unique texts, so additional texts below the saved cutoff cannot contribute."),
            "",
        ])
    else:
        quality_columns += [(f"quality.{metric}", metric) for metric in QUALITY]
    domain_rows = _domain_rows(display)
    domain_section = []
    if domain_rows:
        domain_recall = _recall_columns("id_quality.", "quality.")
        domain_section.extend([
            "## MultiDoc2Dial by domain", "",
            _table(domain_rows, [("run_label", "Run"), ("domain", "Domain"), ("queries", "Queries"),
                                 ("labeled_queries", "Labeled"), *domain_recall],
                   [key for key, _ in domain_recall]),
            "",
        ])
    text = [
        "# Retrieval run comparison",
        "",
        "Quality values are percentages; latency is seconds and memory is GiB. Missing values are shown as —.",
        "Compare matched query/corpus hashes, label protocols, hardware, and timing scopes. These rows are not pooled averages.",
        ("Latency runs measure one query at a time. Throughput runs use closed-loop concurrent conversations; "
        "their latency includes resource contention and server queues, but excludes waiting for a conversation worker. "
        "Wall queries/s is completed queries divided by measured wall time; it is not the reciprocal of mean latency."),
        "",
        "## Quality",
        "",
        *recall_notes,
        _table(display, quality_columns, quality_percentages),
        "",
        *domain_section,
        "## Efficiency",
        "",
        _table(
            display,
            [
                ("run_label", "Run"),
                ("latency.p50_seconds", "p50 s"),
                ("latency.p95_seconds", "p95 s"),
                ("latency.p99_seconds", "p99 s"),
                ("latency.serial_queries_per_second", "Serial queries/s"),
                ("throughput.queries_per_second", "Wall queries/s"),
                ("throughput.wall_seconds", "Measured wall s"),
                ("observed_query_concurrency", "Peak active queries"),
                ("mean_query_embedding_ms", "Query embedding ms"),
                ("mean_vector_search_ms", "Vector search ms"),
                ("search_index_gib", "Index GiB"),
                ("gpu_device_peak_used_gib", "Device peak GiB"),
                ("gpu_mean_utilization_percent", "Mean GPU util %"),
                ("gpu_peak_allocated_gib", "GPU allocated GiB"),
                ("gpu_peak_reserved_gib", "GPU reserved GiB"),
                ("work.input_tokens_per_query", "Input tokens/query"),
                ("work.forward_passes_per_query", "Forwards/query"),
                ("work.requests_per_query", "Requests/query"),
            ],
        ),
        "",
        "## Protocol",
        "",
        _table(
            display,
            [
                ("run_label", "Run"),
                ("query_short", "Query SHA"),
                ("corpus_short", "Corpus SHA"),
                ("dedup_text", "Text dedup"),
                ("warmup_queries", "Warmups"),
                ("gpu", "GPU"),
                ("engine", "Inference engine"),
                ("vllm_server_version", "vLLM server version"),
                ("effective_query_concurrency", "Workers"),
                ("search_device", "Search device"),
                ("search_precision", "Search precision"),
                ("git_dirty", "Dirty tree"),
            ],
        ),
        "",
        "## Sources",
        "",
    ]
    text.extend(f"- {row['report_id']}: `{row['run']}` — {_cell(row.get('timing_scope'))}" for row in display)
    return "\n".join(text) + "\n"


def paired_comparison(left_dir, right_dir, metric="recall@10", samples=2000, seed=42):
    from decision_rag.metrics import paired_bootstrap
    from decision_rag.schema import read_jsonl

    left_dir, right_dir = Path(left_dir), Path(right_dir)
    left_summary = flatten_summary(left_dir / "summary.json")
    right_summary = flatten_summary(right_dir / "summary.json")
    for field in ("dedup_text", "query_sha256", "corpus_sha256"):
        if left_summary.get(field) != right_summary.get(field):
            raise ValueError(f"Cannot pair runs with different {field}")
    left, right = list(read_jsonl(left_dir / "queries.jsonl")), list(read_jsonl(right_dir / "queries.jsonl"))
    for rows in (left, right):
        if len({row["query_id"] for row in rows}) != len(rows):
            raise ValueError("Paired runs contain duplicate query IDs")
    groups_left = {row["query_id"]: row.get("conversation_id") for row in left}
    groups_right = {row["query_id"]: row.get("conversation_id") for row in right}
    if groups_left != groups_right:
        raise ValueError("Paired runs must use the same query IDs and conversation grouping")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    result = paired_bootstrap(left, right, metric=metric, samples=samples, seed=seed)
    return {"left": str(left_dir), "right": str(right_dir), **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", help="Result directories, summary.json or summary.text-recall.json paths"
    )
    parser.add_argument("--output", default="results/comparison", help="Output prefix; writes .csv and .md")
    parser.add_argument("--paired", nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--metric", default="recall@10")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rows = [flatten_summary(path) for path in find_summaries(args.paths)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = markdown_text(rows)
    if args.paired:
        comparison = paired_comparison(*args.paired, args.metric, args.bootstrap_samples, args.seed)
        Path(str(output) + ".paired.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        lo, hi = comparison["ci95"]
        report += (
            f"\n## Paired comparison\n\n{args.metric}: A minus B = "
            f"{comparison['delta_a_minus_b'] * 100:.2f} percentage points; "
            f"95% bootstrap interval [{lo * 100:.2f}, {hi * 100:.2f}]. "
            f"Resampling unit: {comparison['resampling_unit']}.\n"
        )
    Path(str(output) + ".csv").write_text(csv_text(rows), encoding="utf-8")
    Path(str(output) + ".md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
