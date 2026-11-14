"""Retrieval metrics in [0,1], with macro averaging over labeled queries."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def retrieval_metrics(ranking: Sequence[str], relevant: Mapping[str, float], ks=(1, 5, 10, 20)) -> dict:
    if any(not math.isfinite(float(v)) for v in relevant.values()):
        raise ValueError("Relevance grades must be finite")
    gold = {str(k): float(v) for k, v in relevant.items() if v > 0}
    if not gold:
        return {}
    ranking = list(dict.fromkeys(str(x) for x in ranking))
    scores = {}
    highest = max(gold.values())
    def gain(grade):
        return 2 ** (grade - highest) - 2 ** (-highest)
    for k in ks:
        top = ranking[:k]
        hits = sum(x in gold for x in top)
        scores[f"recall@{k}"] = hits / len(gold)
        scores[f"hit@{k}"] = float(hits > 0)
        dcg = sum(gain(gold.get(x, 0)) / math.log2(i + 2) for i, x in enumerate(top))
        ideal = sum(gain(v) / math.log2(i + 2)
                    for i, v in enumerate(sorted(gold.values(), reverse=True)[:k]))
        scores[f"ndcg@{k}"] = dcg / ideal
        scores[f"mrr@{k}"] = next((1 / (i + 1) for i, x in enumerate(top) if x in gold), 0.0)
    return scores


def aggregate(rows: Sequence[dict]) -> dict:
    names = sorted({key for row in rows for key in row})
    return {key: float(np.mean([r[key] for r in rows if key in r])) for key in names}


def latency_summary(seconds: Sequence[float]) -> dict:
    if not seconds:
        return {}
    values = np.asarray(seconds)
    return {"queries": len(values), "mean_seconds": float(values.mean()),
            **{f"p{p}_seconds": float(np.percentile(values, p)) for p in (50, 90, 95, 99)},
            "serial_queries_per_second": float(len(values) / values.sum()) if values.sum() else None,
            "total_seconds": float(values.sum())}


def paired_bootstrap(a: Sequence[dict], b: Sequence[dict], metric="recall@10", samples=2000, seed=42):
    """Resample complete conversations (individual queries for non-dialogue data)."""
    left, right = {r['query_id']: r for r in a}, {r['query_id']: r for r in b}
    if len(left) != len(a) or len(right) != len(b):
        raise ValueError("Duplicate query IDs in paired runs")
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    if left.keys() != right.keys():
        raise ValueError("Paired runs must contain identical query IDs")
    groups = defaultdict(list)
    for qid, l in left.items():
        r = right[qid]
        if l.get('conversation_id') != r.get('conversation_id'):
            raise ValueError('Conversation IDs differ between paired runs')
        if metric in l['metrics'] and metric in r['metrics']:
            groups[l.get('conversation_id') or qid].append(l['metrics'][metric] - r['metrics'][metric])
    if not groups:
        raise ValueError("No labeled paired queries for selected metric")
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    draws = [float(np.mean([d for i in rng.integers(len(values), size=len(values)) for d in values[i]]))
             for _ in range(samples)]
    return {"metric": metric, "delta_a_minus_b": float(np.mean([d for g in values for d in g])),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "samples": samples, "seed": seed,
            "resampling_unit": "conversation_or_query"}
