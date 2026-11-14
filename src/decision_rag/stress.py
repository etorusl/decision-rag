"""Deterministic synthetic corpora for performance stress, not natural-language quality claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from .schema import Corpus, Node, Query, write_jsonl

TOPICS = (
    "Astronomy",
    "Agriculture",
    "Architecture",
    "Biology",
    "Chemistry",
    "Computing",
    "Ecology",
    "Education",
    "Energy",
    "Engineering",
    "Geography",
    "Geology",
    "History",
    "Linguistics",
    "Manufacturing",
    "Mathematics",
    "Meteorology",
    "Oceanography",
    "Physics",
    "Transportation",
)
FILLER = [
    "The",
    "handbook",
    "describes",
    "routine",
    "observations",
    "and",
    "procedures",
    "for",
    "the",
    "research",
    "team.",
    "Each",
    "entry",
    "records",
    "a",
    "separate",
    "measurement",
    "under",
    "controlled",
    "conditions.",
    "The",
    "documentation",
    "explains",
    "how",
    "staff",
    "organize",
    "records",
    "and",
    "review",
    "the",
    "results.",
    "Readers",
    "can",
    "consult",
    "the",
    "relevant",
    "entry",
    "to",
    "locate",
    "its",
    "stored",
    "value.",
]


def generate(
    output: str | Path,
    *,
    pages: int = 1000,
    passages_per_page: int = 10,
    words_per_passage: int = 100,
    queries: int = 100,
    seed: int = 42,
    branching: int = 100,
) -> dict:
    if pages < 1 or passages_per_page < 1 or queries < 1 or words_per_passage < 16 or branching < 2:
        raise ValueError(
            "Require positive pages/passages/queries, words_per_passage >= 16 and branching >= 2"
        )
    total = pages * passages_per_page
    if queries > total:
        raise ValueError("queries cannot exceed the number of unique planted passage records")
    output = Path(output)
    if any((output / filename).exists() for filename in ("corpus.jsonl", "queries.jsonl", "manifest.json")):
        raise ValueError("Stress output already exists; choose another output directory")
    rng = random.Random(seed)
    # Construct the entire corpus before selecting queries. Every passage has the same
    # record/value structure, so no candidate is marked as gold or special.
    page_rows = []
    passages = []
    record_data = []
    for page_index in range(pages):
        topic = TOPICS[page_index % len(TOPICS)]
        title = f"{topic} handbook {page_index:06d}"
        page_id = f"page-{page_index:06d}"
        page_rows.append((page_id, title, page_index))
        for passage_index in range(passages_per_page):
            key = f"R{page_index:06d}-{passage_index:04d}"
            value = "V" + hashlib.sha256(f"{seed}:{page_index}:{passage_index}".encode()).hexdigest()[:16]
            head = f"{title}. Record {key} stores value {value}. This entry concerns {topic.lower()} measurements.".split()
            offset = rng.randrange(len(FILLER))
            words = head + [FILLER[(offset + i) % len(FILLER)] for i in range(words_per_passage - len(head))]
            pid = f"passage-{page_index:06d}-{passage_index:04d}"
            passages.append(Node(pid, " ".join(words), "passage", page_id, order=passage_index))
            record_data.append((pid, title, key, value))
    # Build balanced corpus-only groups from consecutive pages. Higher levels are
    # introduced when needed so every internal fan-out is bounded by branching.
    current = [(pid, title, index, index) for pid, title, index in page_rows]
    parents = {}
    groups = []
    level = 0
    while len(current) > branching:
        next_level = []
        for start in range(0, len(current), branching):
            chunk = current[start : start + branching]
            group_id = f"group-{level}-{start // branching:06d}"
            first, last = chunk[0][2], chunk[-1][3]
            topics = list(
                dict.fromkeys(TOPICS[i % len(TOPICS)] for i in range(first, min(last + 1, first + 20)))
            )
            description = f"Handbooks {first:06d} through {last:06d}; subjects: {', '.join(topics)}."
            groups.append((group_id, description, first))
            for child, *_ in chunk:
                parents[child] = group_id
            next_level.append((group_id, description, first, last))
        current = next_level
        level += 1
    for child, *_ in current:
        parents[child] = "root"
    nodes = [Node("root", "Synthetic research handbook library", "root")]
    nodes.extend(Node(gid, text, "group", parents[gid], text, order) for gid, text, order in groups)
    nodes.extend(Node(pid, title, "page", parents[pid], title, index) for pid, title, index in page_rows)
    nodes.extend(passages)
    corpus = Corpus(nodes)
    examples = []
    for query_index, selected in enumerate(rng.sample(range(total), queries)):
        pid, title, key, value = record_data[selected]
        examples.append(
            Query(
                f"stress-{query_index:06d}",
                f"In {title}, what is the stored value for record {key}?",
                {pid: 1.0},
                answers=[value],
                metadata={
                    "dataset": "synthetic-stress",
                    "split": "test",
                    "purpose": "performance stress only",
                },
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    corpus.save(output / "corpus.jsonl")
    write_jsonl(output / "queries.jsonl", (asdict(query) for query in examples))
    manifest = {
        "dataset": "synthetic-stress",
        "purpose": "Performance stress only; not natural-language retrieval quality",
        "seed": seed,
        "pages": pages,
        "passages": total,
        "passages_per_page": passages_per_page,
        "words_per_passage": words_per_passage,
        "queries": queries,
        "branching": branching,
        "group_levels": level,
        "nodes": len(nodes),
        "corpus_sha256": hashlib.sha256((output / "corpus.jsonl").read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256((output / "queries.jsonl").read_bytes()).hexdigest(),
        "expected_max_page_levels": max(1, math.ceil(math.log(pages, branching))) if pages > 1 else 1,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", type=int, default=1000)
    parser.add_argument("--passages-per-page", type=int, default=10)
    parser.add_argument("--words-per-passage", type=int, default=100)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--branching", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(generate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
