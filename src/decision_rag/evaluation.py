"""Reproducible evaluation with no answer caching and explicit timing scope."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

from .backends.base import sum_known
from .metrics import aggregate, latency_summary, retrieval_metrics


def synchronize():
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def environment():
    import importlib.metadata
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    result = {"python": sys.version, "platform": platform.platform(), "git_commit": git('rev-parse', 'HEAD'),
              "git_dirty": bool(git('status', '--porcelain')), "packages": {}}
    for name in ['decision-rag', 'numpy', 'torch', 'transformers', 'sentence-transformers', 'peft', 'vllm']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if 'torch' in sys.modules:
        import torch
        result['cuda'] = torch.version.cuda
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name()
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate(retriever, queries, output_dir, *, warmup=0, config=None, source_paths=(), dedup_text=False, initialization_seconds=None, gpu_monitor_index=None):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs")
    passages = {p.id for p in retriever.corpus.passages}
    for query in queries:
        if any(not math.isfinite(float(v)) for v in query.relevant_ids.values()):
            raise ValueError("Relevance grades must be finite")
        unknown = set(query.relevant_ids) - passages
        if unknown:
            raise ValueError(f"Unknown gold passage IDs for {query.id}: {sorted(unknown)[:5]}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'queries.jsonl').exists():
        raise ValueError("Output already contains a run. Choose a new --output directory.")
    # Ensure the chronological stateful protocol is unambiguous.
    last = {}
    for query in queries:
        if query.conversation_id:
            previous = last.get(query.conversation_id, -1)
            if query.turn_id <= previous:
                raise ValueError(f"Non-increasing turns in conversation {query.conversation_id}")
            last[query.conversation_id] = query.turn_id
    external = bool(getattr(retriever, "external_inference", False))
    def sync():
        if not external:
            synchronize()
    def prepare(query):
        if hasattr(retriever, "prepare_query"):
            retriever.prepare_query(query)
    retriever.reset()
    sync()
    warmup_start = time.perf_counter()
    for query in queries[:warmup]:
        prepare(query)
        retriever.retrieve(query)
    sync()
    warmup_seconds = time.perf_counter() - warmup_start
    retriever.reset()  # warmup decisions must never leak into measured conversations
    sync()
    if not external and 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    canonical = {}
    lowercase_dedup = bool(queries) and all(q.metadata.get("dataset") == "mtrag" for q in queries)
    if dedup_text:
        import re
        for p in retriever.corpus.passages:
            key = re.sub(r'\s+', ' ', p.text).strip()
            if lowercase_dedup:
                key = key.lower()
            canonical[p.id] = hashlib.sha256(key.encode()).hexdigest()
    metrics, latencies = [], []
    work = {key: 0 for key in ("input_tokens", "forward_passes", "requests")}
    from .telemetry import GPUMonitor
    monitor = GPUMonitor(gpu_monitor_index)
    with monitor, (output / 'queries.jsonl').open('w', encoding='utf-8') as stream:
        for query in queries:
            prepare(query)
            sync()
            start = time.perf_counter()
            result = retriever.retrieve(query)
            sync()
            elapsed = time.perf_counter() - start
            relevant = query.relevant_ids
            ranking = result.ids
            if canonical:
                ranking = [canonical[x] for x in ranking]
                relevant = {canonical[x]: max(v, max((w for y, w in relevant.items()
                                                      if canonical[y] == canonical[x]), default=0))
                            for x, v in relevant.items()}
            measured = retrieval_metrics(ranking, relevant)
            record = {**result.to_dict(), "conversation_id": query.conversation_id,
                      "turn_id": query.turn_id, "latency_seconds": elapsed, "metrics": measured,
                      "labeled": bool(measured)}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            metrics.append(measured)
            latencies.append(elapsed)
            for stage in result.stages:
                for key in ("prompt_requests", "completion_tokens", "cached_tokens"):
                    if key in stage:
                        work.setdefault(key, 0)
                for key, value in work.items():
                    work[key] = sum_known([value, stage.get(key, 0)])
    summary = {"quality": aggregate(metrics), "labeled_queries": sum(bool(m) for m in metrics),
               "unlabeled_queries": sum(not bool(m) for m in metrics),
               "latency": latency_summary(latencies), "warmup_queries": min(warmup, len(queries)),
               "initialization_seconds": initialization_seconds,
               "warmup_seconds": warmup_seconds,
               "timing_scope": "Complete serial retrieval after model/corpus/index initialization; no result cache",
               "dedup_text": dedup_text,
               "dedup_normalization": ("whitespace+lowercase" if lowercase_dedup else "whitespace") if dedup_text else None,
               "environment": environment(), "config": config or {},
               "inputs": {str(p): file_hash(p) for p in source_paths},
               "work": work,
               "inference_process": "external_vllm_server" if external else "benchmark_process",
               "gpu_telemetry": monitor.summary()}
    if not external and 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            summary['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary
