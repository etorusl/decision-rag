#!/usr/bin/env python3
"""Recover an interrupted suite in a fresh run, preserving all previous results."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from decision_rag.evaluation import file_hash
from decision_rag.schema import load_queries, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
METHOD_MODELS = {
    'bm25': (),
    'dense-vllm': ('embedding_model',),
    'dense-rerank-vllm': ('embedding_model', 'reranker_model'),
    'dense-bge-vllm': ('bge_embedding_model',),
    'dense-rerank-bge-vllm': ('bge_embedding_model', 'bge_reranker_model'),
    'naive-binary-vllm': ('naive_model',),
    'naive-choice-vllm': ('naive_model',),
}
RESTORE_OPTIONS = (
    'naive_model', 'embedding_model', 'reranker_model', 'bge_embedding_model', 'bge_reranker_model',
    'max_model_len', 'pool_max_model_len', 'bge_max_model_len', 'pool_memory', 'decision_memory',
    'search_device', 'port_base', 'startup_timeout', 'vllm_batch_size', 'batch_size',
)


def plan_recovery(run_dir, *, execution_mode='throughput', query_concurrency=8, run_tag=None,
                  gpu_index='1', model_manifest=None, root=ROOT):
    root, source = Path(root).resolve(), Path(run_dir).resolve()
    if execution_mode not in {'latency', 'throughput'}:
        raise ValueError('execution_mode must be latency or throughput')
    if type(query_concurrency) is not int or query_concurrency < 1:
        raise ValueError('query_concurrency must be a positive integer')
    if execution_mode == 'latency' and query_concurrency != 1:
        raise ValueError('Latency recovery requires query_concurrency=1')
    if not str(gpu_index).isascii() or not str(gpu_index).isdecimal():
        raise ValueError('gpu_index must identify one nonnegative physical GPU')
    config_path = source / 'server-config.json'
    config = json.loads(config_path.read_text())
    if config.get('ablations'):
        raise ValueError('Recovery currently supports the standard benchmark suite, not ablations')
    dataset = config.get('dataset')
    if dataset not in {'multidoc2dial', 'conditionalqa', 'mtrag', 'scifact'}:
        raise ValueError('Source server-config.json has an unsupported dataset')
    if source.name != dataset:
        raise ValueError('Pass the dataset run directory containing server-config.json')
    limit = str(config.get('limit', ''))
    if limit != 'all' and (not limit.isdecimal() or int(limit) < 1):
        raise ValueError('Source server-config.json has an invalid query limit')
    requested = config.get('methods')
    if requested is None:
        requested = [name for name in METHOD_MODELS if not (config.get('skip_bge') and '-bge-' in name)]
    if not isinstance(requested, list) or not requested or len(set(requested)) != len(requested):
        raise ValueError('Source methods must be a nonempty list without duplicates')
    if set(requested) - METHOD_MODELS.keys():
        raise ValueError('Source run contains unsupported methods')
    methods = [name for name in METHOD_MODELS if name in requested]
    corpus_path, queries_path = (root / 'data' / dataset / name for name in ('corpus.jsonl', 'queries.jsonl'))
    if not corpus_path.is_file() or not queries_path.is_file():
        raise ValueError(f'Existing corpus and queries are required in {root / "data" / dataset}')
    queries = load_queries(queries_path)
    if limit != 'all':
        queries = queries[:int(limit)]
    expected_ids = [query.id for query in queries]
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Existing query selection must be nonempty with unique IDs')
    current_hashes = {path.name: file_hash(path) for path in (corpus_path, queries_path)}
    completed, pending, partial = [], [], []
    for method in methods:
        directory = source / method
        summary_path = directory / 'summary.json'
        if not summary_path.exists():
            pending.append(method)
            if (directory / 'queries.jsonl').exists():
                partial.append(method)
            continue
        try:
            summary = json.loads(summary_path.read_text())
            if not isinstance(summary.get('quality'), dict) or summary.get('latency', {}).get('queries') != len(expected_ids):
                raise ValueError('incomplete summary or mismatched query count')
            for basename, digest in current_hashes.items():
                recorded = [value for key, value in summary.get('inputs', {}).items() if Path(key).name == basename]
                if recorded != [digest]:
                    raise ValueError(f'{basename} hash differs from the completed run')
            actual_ids = [row['query_id'] for row in read_jsonl(directory / 'queries.jsonl')]
            if actual_ids != expected_ids:
                raise ValueError('completed query IDs/order differ from the original query selection')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f'Cannot validate completed method {method}: {exc}') from exc
        completed.append(str(summary_path))
    plan = {'source_run': str(source), 'new_run': None, 'command': None,
            'completed_summaries': completed, 'pending_methods': pending, 'partial_methods': partial,
            'data_hash_validation': 'completed_summaries' if completed else 'unavailable'}
    if not pending:
        return plan
    revisions = config.get('revisions', {})
    for method in pending:
        for option in METHOD_MODELS[method]:
            model = config.get(option)
            if not isinstance(model, str) or not model:
                raise ValueError(f'Source config lacks {option}; model identity must not be guessed')
            if not re.fullmatch('[0-9a-fA-F]{40}', str(revisions.get(model, ''))):
                raise ValueError(f'Source config lacks an immutable revision for {model}')
    tag = run_tag or f'{source.parent.name}-resume-c{query_concurrency}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}'
    if not tag or Path(tag).name != tag or tag in {'.', '..'}:
        raise ValueError('run_tag must be one directory name')
    destination = root / 'results' / tag / dataset
    if destination.resolve() == source or destination.exists():
        raise ValueError(f'Recovery requires a new output directory: {destination}')
    command = [sys.executable, str(root / 'scripts/run_vllm_benchmark.py'), dataset, limit,
               '--run-tag', tag, '--methods', *pending,
               '--execution-mode', execution_mode, '--query-concurrency', str(query_concurrency),
               '--gpu-index', str(gpu_index), '--skip-prepare', '--reuse-revisions', str(config_path)]
    for option in RESTORE_OPTIONS:
        if config.get(option) is not None:
            command += ['--' + option.replace('_', '-'), str(config[option])]
    if config.get('skip_bge'):
        command.append('--skip-bge')
    manifest = model_manifest if model_manifest is not None else config.get('model_manifest')
    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        if not manifest_path.is_file():
            raise ValueError(f'Model manifest is missing: {manifest_path}; provide --model-manifest with its current path')
        command += ['--model-manifest', str(manifest_path.resolve())]
    return {**plan, 'new_run': str(destination), 'command': command}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='Old results/<tag>/<dataset> directory')
    parser.add_argument('--execution-mode', choices=['latency', 'throughput'], default='throughput')
    parser.add_argument('--query-concurrency', type=int)
    parser.add_argument('--run-tag')
    parser.add_argument('--gpu-index', default='1')
    parser.add_argument('--model-manifest')
    parser.add_argument('--dry-run', action='store_true', help='Print the recovery plan without starting servers or writing files')
    args = parser.parse_args(argv)
    concurrency = args.query_concurrency if args.query_concurrency is not None else (8 if args.execution_mode == 'throughput' else 1)
    plan = plan_recovery(args.run_dir, execution_mode=args.execution_mode, query_concurrency=concurrency,
                         run_tag=args.run_tag, gpu_index=args.gpu_index, model_manifest=args.model_manifest, root=ROOT)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or plan['command'] is None:
        return plan
    subprocess.run(plan['command'], check=True, cwd=ROOT)
    destination = Path(plan['new_run'])
    (destination / 'recovery-plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_runs.py'),
                    *plan['completed_summaries'], str(destination),
                    '--output', str(destination / 'recovery-comparison')], check=True, cwd=ROOT)
    print(f'Recovery report: {destination / "recovery-comparison.md"}')
    return plan


if __name__ == '__main__':
    main()
