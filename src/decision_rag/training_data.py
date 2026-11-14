"""Retrieval supervision built exclusively from an explicit training query split.

This deterministic local recipe replaces unreleased Jev mining/general-training data.
It is not the paper's exact training distribution. Evaluation never calls this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .baselines import BM25Retriever
from .prompts import build_prompt
from .retrieval import DecisionRetriever, RetrievalConfig


def make_training_data(corpus, queries, output_dir, backend, *, dataset='multidoc2dial', seed=42,
                       max_tokens=8192, max_candidates=128, monitor_fraction=.03,
                       gold_page_probability=.85, max_queries=None):
    if any(q.metadata.get('split') not in {'train', 'training'} for q in queries):
        raise ValueError("Training records require queries marked metadata.split=train; never use evaluation labels")
    if not 0 < monitor_fraction < 1 or not 0 <= gold_page_probability <= 1:
        raise ValueError('Invalid training split/probability')
    if len({q.id for q in queries}) != len(queries):
        raise ValueError("Duplicate training query IDs")
    valid_passages = {p.id for p in corpus.passages}
    valid_pages = {p.id for p in corpus.pages}
    for q in queries:
        if any(not math.isfinite(float(v)) for v in q.relevant_ids.values()):
            raise ValueError("Relevance grades must be finite")
        if set(q.relevant_ids) - valid_passages:
            raise ValueError(f"Unknown gold passage IDs for {q.id}")
        if set(q.metadata.get('gold_page_ids', [])) - valid_pages:
            raise ValueError(f"Unknown gold page IDs for {q.id}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'train.jsonl').exists():
        raise ValueError('Training data already exist; choose another output directory')
    rng = random.Random(seed)
    # Full dialogues / source pages, never individual turns, form disjoint monitoring units.
    def unit(q):
        if q.conversation_id:
            return q.conversation_id
        pages = sorted(set(q.metadata.get('gold_page_ids', [])) |
                       {corpus.page_id(pid) for pid, rel in q.relevant_ids.items() if rel > 0})
        return pages[0] if pages else q.id
    units = sorted({unit(q) for q in queries})
    if len(units) < 2:
        raise ValueError('At least two training groups required for held-out monitoring')
    rng.shuffle(units)
    monitor = set(units[:max(1, round(len(units) * monitor_fraction))])
    scorer = DecisionRetriever(corpus, backend, RetrievalConfig(max_request_tokens=max_tokens,
                                                               max_candidates=max_candidates))
    bm25 = BM25Retriever(corpus, top_k=100)
    counts = {"train": 0, "monitor": 0}
    train_path, valid_path = output / "train.jsonl.partial", output / "monitor.jsonl.partial"
    seen = set()
    with train_path.open("w", encoding="utf-8") as train, valid_path.open("w", encoding="utf-8") as valid:
        for query in queries[:max_queries] if max_queries else queries:
            gold = {pid for pid, rel in query.relevant_ids.items() if rel > 0}
            gold_pages = set(query.metadata.get('gold_page_ids', [])) | {corpus.page_id(pid) for pid in gold}
            if not gold_pages:
                continue
            gold_texts = {' '.join(corpus.nodes[pid].text.split()).lower() for pid in gold}
            split = "monitor" if unit(query) in monitor else "train"
            target = valid if split == "monitor" else train
            memory = sorted(gold_pages)[:1] if query.conversation_id and rng.random() < .65 else []
            stages = []
            centers = corpus.children(corpus.root.id)
            if dataset == 'multidoc2dial':
                center_ids = {corpus.nodes[p].parent_id for p in gold_pages}
                stages.append(('centers', centers, center_ids, []))
                pages = [p for p in corpus.pages if p.parent_id in center_ids]
            else:
                pages = corpus.pages
            stages.append(('pages', pages, gold_pages, memory))
            # Hard negatives from BM25, no validation/dev/test queries are ever mined here.
            ranked_pages = list(dict.fromkeys(corpus.page_id(p) for p in bm25.retrieve(query).ids))
            count = 5 if dataset == 'conditionalqa' else 3
            candidates = [p for p in ranked_pages if p not in gold_pages][:count]
            if rng.random() < gold_page_probability:
                candidates = sorted(gold_pages) + candidates
                candidates = candidates[:count]
            if memory:
                candidates = list(dict.fromkeys(candidates + memory))
            passages = [p for page in candidates for p in corpus.descendants(page)]
            # Repeated gold text with a different ID must not become a negative example.
            passages = [p for p in passages if p.id in gold or ' '.join(p.text.split()).lower() not in gold_texts]
            if gold:
                stages.append(('passages', passages, gold, []))
            for stage, nodes, positives, notepad in stages:
                for chunk in scorer._chunks(query, nodes, notepad, False):
                    context, questions = build_prompt(query, chunk, corpus, notepad)
                    record = {'context': context, 'questions': questions, 'options': [['no','yes'] for _ in chunk],
                              'labels': [int(n.id in positives) for n in chunk],
                              'metadata': {'query_id': query.id, 'unit': unit(query), 'stage': stage,
                                           'candidate_ids': [n.id for n in chunk]}}
                    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                    if digest not in seen:
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts[split] += 1
                        seen.add(digest)
    if not counts["train"] or not counts["monitor"]:
        raise ValueError('Empty train or monitoring records; use more queries or a different seed')
    train_path.replace(output / 'train.jsonl')
    valid_path.replace(output / 'monitor.jsonl')
    manifest = {'dataset': dataset, 'seed': seed, 'training_records': counts['train'], 'monitor_records': counts['monitor'],
                'monitor_units': sorted(monitor), 'monitor_fraction': monitor_fraction,
                'max_tokens': max_tokens, 'max_candidates': max_candidates,
                'gold_page_probability': gold_page_probability,
                'recipe': 'Local BM25 negatives and simulated gold-page memory; not released paper training data'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
