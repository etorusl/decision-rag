#!/usr/bin/env python3
"""Collect evaluation summaries into CSV and Markdown; no dependency needed for reporting."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

QUALITY = ("hit@1", "recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@10")


def find_summaries(paths):
    found = set()
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Results path does not exist: {path}")
        if path.is_file():
            if path.name != "summary.json":
                raise ValueError(f"Expected summary.json, got {path}")
            found.add(path.resolve())
        else:
            found.update(p.resolve() for p in path.rglob("summary.json"))
    if not found:
        raise ValueError("No summary.json files found")
    return sorted(found)


def flatten_summary(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "quality" not in data or "latency" not in data:
        raise ValueError(f"Not an evaluation summary: {path}")
    latency, work = data["latency"], data.get("work", {})
    count = latency.get("queries", 0)
    config = data.get("config", {})
    env = data.get("environment", {})
    query_hashes = sorted(v for k, v in data.get("inputs", {}).items() if Path(k).name == "queries.jsonl")
    corpus_hashes = sorted(v for k, v in data.get("inputs", {}).items() if Path(k).name == "corpus.jsonl")
    row = {
        "run": str(path.parent),
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
        "config_json": json.dumps(config, sort_keys=True),
    }
    row.update({f"quality.{key}": value for key, value in data["quality"].items()})
    row.update({f"latency.{key}": value for key, value in latency.items() if key != "queries"})
    for key in ("input_tokens", "forward_passes", "requests"):
        row[f"work.{key}"] = work.get(key, 0)
        row[f"work.{key}_per_query"] = work.get(key, 0) / count if count else None
    for kind in ("allocated", "reserved"):
        value = data.get(f"gpu_peak_{kind}_bytes")
        row[f"gpu_peak_{kind}_gib"] = value / 2**30 if value is not None else None
    return row


def csv_text(rows):
    if not rows:
        raise ValueError("No summary rows")
    leading = ["run", "queries", "labeled_queries", "unlabeled_queries"]
    fields = leading + sorted(set().union(*(row.keys() for row in rows)) - set(leading))
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


def markdown_text(rows):
    quality_columns = [("run_label", "Run"), ("queries", "Queries"), ("labeled_queries", "Labeled")]
    quality_columns += [(f"quality.{metric}", metric) for metric in QUALITY]
    display = [
        {
            **row,
            "run_label": row["run"],
            "query_short": row.get("query_sha256", "")[:12],
            "corpus_short": row.get("corpus_sha256", "")[:12],
        }
        for row in rows
    ]
    text = [
        "# Retrieval run comparison",
        "",
        "Quality values are percentages; latency is seconds and memory is GiB. Missing values are shown as —.",
        "Compare matched query/corpus hashes, label protocols, hardware, and timing scopes. These rows are not pooled averages.",
        "",
        "## Quality",
        "",
        _table(display, quality_columns, [f"quality.{m}" for m in QUALITY]),
        "",
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
                ("git_dirty", "Dirty tree"),
            ],
        ),
        "",
        "## Sources",
        "",
    ]
    text.extend(f"- `{row['run']}` — {_cell(row.get('timing_scope'))}" for row in rows)
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
        "paths", nargs="+", help="Result directories or summary.json paths (searched recursively)"
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
