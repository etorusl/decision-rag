#!/usr/bin/env python3
"""Run smoke checks, then the original choice baseline and new-model experiments."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

from decision_rag.evaluation import file_hash
from decision_rag.schema import load_queries, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'Qwen/Qwen3.5-9B': 'qwen35-9b', 'google/gemma-4-12B-it': 'gemma4-12b-it'}
METHODS = ('naive-binary-vllm', 'naive-choice-vllm')
SMOKE_QUERIES = 20
SPEC = importlib.util.spec_from_file_location('experiment_recovery', Path(__file__).with_name('resume_vllm_benchmark.py'))
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def _path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _save(path, state):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.experiment-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get('format_version') != 1 or not isinstance(manifest.get('models'), dict):
        raise ValueError('Expected a download_models.py manifest with immutable model revisions')
    revisions = {}
    for model in MODELS:
        revision = manifest['models'].get(model, {}).get('revision')
        if not isinstance(revision, str) or not re.fullmatch('[0-9a-fA-F]{40}', revision):
            raise ValueError(f'Model manifest lacks an immutable revision for {model}')
        revisions[model] = revision
    return revisions


def _python(root, modern):
    path = root / ('.venv-vllm-modern' if modern else '.venv-vllm') / 'bin/python'
    if not path.is_file():
        raise ValueError(f'Inference environment is missing: {path}')
    # Preserve the virtualenv path: resolving its symlink would select the base interpreter.
    return str(path)


def _run_paths(experiment, dataset, stage):
    return {model: str(experiment / f'{slug}-{stage}' / dataset) for model, slug in MODELS.items()}


def _tag(directory, dataset, root):
    if directory.name != dataset:
        raise ValueError('Run directory must end with the dataset name')
    return directory.parent.relative_to(root / 'results').as_posix()


def _new_model_command(root, config, model, manifest, directory, limit, concurrency, gpu_index):
    command = [_python(root, True), str(root / 'scripts/run_vllm_benchmark.py'), config['dataset'], str(limit),
               '--run-tag', _tag(directory, config['dataset'], root), '--methods', *METHODS,
               '--naive-model', model, '--model-manifest', str(manifest), '--skip-prepare',
               '--execution-mode', 'throughput', '--query-concurrency', str(concurrency),
               '--gpu-index', str(gpu_index)]
    model_options = {option for options in RECOVERY.METHOD_MODELS.values() for option in options}
    for option in RECOVERY.RESTORE_OPTIONS:
        if option not in model_options and config.get(option) is not None:
            command += ['--' + option.replace('_', '-'), str(config[option])]
    return command


def _validate_results(directory, model, revision, query_ids, data_hashes, concurrency, *, methods=METHODS):
    summaries, versions = [], {}
    for method in methods:
        choice_k = 20 if method == 'naive-choice-vllm' else 0
        path = directory / method / 'summary.json'
        try:
            summary = json.loads(path.read_text())
            config, execution = summary.get('config', {}), summary.get('execution', {})
            if not isinstance(summary.get('quality'), dict) or summary.get('latency', {}).get('queries') != len(query_ids):
                raise ValueError('incomplete summary or incorrect query count')
            if config.get('method') != 'decision' or config.get('model') != model or config.get('model_revision') != revision:
                raise ValueError('model identity or revision differs from the experiment pin')
            if config.get('retrieval', {}).get('choice_k') != choice_k:
                raise ValueError('binary/choice scoring mode differs from the experiment')
            if concurrency is not None and (execution.get('mode') != 'throughput'
                                            or execution.get('query_concurrency') != concurrency):
                raise ValueError('execution mode or concurrency differs from the experiment')
            for name, digest in data_hashes.items():
                values = [value for key, value in summary.get('inputs', {}).items() if Path(key).name == name]
                if values != [digest]:
                    raise ValueError(f'{name} differs from the experiment data')
            actual_ids = [row['query_id'] for row in read_jsonl(path.with_name('queries.jsonl'))]
            if actual_ids != query_ids:
                raise ValueError('query IDs/order differ from the experiment selection')
            version = config.get('runtime', {}).get('server_version')
            if not isinstance(version, str) or not version:
                raise ValueError('summary does not identify the vLLM server version')
            versions[method] = version
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f'Cannot validate {model} {method}: {exc}') from exc
        summaries.append(str(path))
    return summaries, versions


def _linked_recoveries(source, root):
    source_config = source / 'server-config.json'
    # Standard resume runs are siblings named with a timestamp. Try newer tags first.
    for candidate in sorted((root / 'results').glob(f'*/{source.name}/server-config.json'), reverse=True):
        if candidate.parent == source:
            continue
        with suppress(OSError, ValueError):
            config = json.loads(candidate.read_text())
            reference = config.get('reuse_revisions') if isinstance(config, dict) else None
            if isinstance(reference, str) and reference and _path(reference, root) == source_config:
                yield candidate.parent


def _recover_legacy_results(source, root, config, original, explicit_runs, query_ids, hashes):
    completed = {Path(path).parent.name: path for path in original}
    if set(METHODS).issubset(completed):
        return completed
    model = config['naive_model']
    revision = config.get('revisions', {}).get(model)
    explicit = list(dict.fromkeys(_path(value, root) for value in explicit_runs))
    for directory in explicit:
        if directory.name != source.name or not any((directory / method / 'summary.json').is_file()
                                                    for method in METHODS):
            raise ValueError(f'No completed legacy decision result for {source.name} in {directory}')
    for directory in [*explicit, *_linked_recoveries(source, root)]:
        for method in METHODS:
            if method in completed or not (directory / method / 'summary.json').is_file():
                continue
            if not isinstance(revision, str) or not re.fullmatch('[0-9a-fA-F]{40}', revision):
                raise ValueError(f'Source config lacks an immutable revision for {model}')
            try:
                summaries, _ = _validate_results(directory, model, revision, query_ids, hashes, None,
                                                  methods=(method,))
            except ValueError as exc:
                if directory in explicit:
                    raise ValueError(f'Cannot reuse legacy result in {directory}: {exc}') from exc
                print(f'Skipping incompatible recovery result in {directory}: {exc}', file=sys.stderr)
                continue
            completed[method] = summaries[0]
            print(f'Reusing {method} from {directory}', flush=True)
        if set(METHODS).issubset(completed):
            break
    return completed


@contextmanager
def _smoke_lock(experiment):
    # A status check alone permits two retries to read smoke_failed simultaneously.
    # Lock the directory inode so rejected retries leave every existing file intact.
    lock = os.open(experiment, os.O_RDONLY)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('A smoke stage is already running in this experiment directory') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        os.close(lock)


def _run_smoke_stage(initial, experiment, config, smoke_ids, root):
    state_path = experiment / 'experiment.json'
    if experiment.exists() and not state_path.is_file():
        raise ValueError(f'The smoke stage requires a fresh directory or a recorded experiment: {experiment}')
    experiment.mkdir(parents=True, exist_ok=True)
    with _smoke_lock(experiment):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('status') not in {'smoke_failed', 'smoke_complete'}:
                raise ValueError(f'Cannot retry smoke while experiment status is {state.get("status")!r}')
            dynamic = {'status', 'smoke_complete', 'smoke_completed_models', 'smoke_server_versions'}
            for key, expected in initial.items():
                if key not in dynamic and state.get(key) != expected:
                    raise ValueError(f'Experiment {key} changed since the first smoke attempt; use a fresh directory')
            completed = state.get('smoke_completed_models')
            if not isinstance(completed, list) or completed != list(MODELS)[:len(completed)]:
                raise ValueError('Invalid completed smoke model sequence')
            if state.get('smoke_complete') != (state['status'] == 'smoke_complete'):
                raise ValueError('Inconsistent smoke completion state')
            for model in completed:
                _, versions = _validate_results(
                    Path(state['smoke_runs'][model]), model, state['model_revisions'][model], smoke_ids,
                    state['data_hashes'], state['query_concurrency'])
                if versions != state.get('smoke_server_versions', {}).get(model):
                    raise ValueError('Completed smoke server version provenance changed')
            if state['status'] == 'smoke_complete':
                if completed != list(MODELS):
                    raise ValueError('Completed smoke is missing model results')
                return state
        else:
            state = initial
        state['status'] = 'smoke_running'
        _save(state_path, state)
        try:
            for model, slug in MODELS.items():
                if model in state['smoke_completed_models']:
                    continue
                directory = Path(state['smoke_runs'][model])
                if directory.exists():
                    archive = experiment / 'failed-attempts' / f'{slug}-{uuid.uuid4().hex}' / config['dataset']
                    archive.parent.mkdir(parents=True)
                    directory.rename(archive)
                    state.setdefault('smoke_failed_attempts', []).append({'model': model, 'directory': str(archive)})
                    _save(state_path, state)
                command = _new_model_command(
                    root, config, model, Path(state['model_manifest']), directory, SMOKE_QUERIES,
                    state['query_concurrency'], state['gpu_index'])
                subprocess.run(command, check=True, cwd=root)
                _, versions = _validate_results(
                    directory, model, state['model_revisions'][model], smoke_ids, state['data_hashes'],
                    state['query_concurrency'])
                state['smoke_completed_models'].append(model)
                state['smoke_server_versions'][model] = versions
                _save(state_path, state)
            state.update(status='smoke_complete', smoke_complete=True)
            _save(state_path, state)
        except BaseException:
            state['status'] = 'smoke_failed'
            with suppress(Exception):
                _save(state_path, state)
            raise
        return state


def run_experiments(source_run, *, stage, experiment_dir, model_manifest=None,
                    legacy_recovery_runs=(), query_concurrency=8, gpu_index='1', root=ROOT):
    root = Path(root).resolve()
    source, experiment = _path(source_run, root), _path(experiment_dir, root)
    results = root / 'results'
    if stage not in {'smoke', 'full'}:
        raise ValueError('stage must be smoke or full')
    if type(query_concurrency) is not int or query_concurrency < 1:
        raise ValueError('query_concurrency must be a positive integer')
    if not str(gpu_index).isascii() or not str(gpu_index).isdecimal():
        raise ValueError('gpu_index must identify one nonnegative physical GPU')
    if experiment == results or not experiment.is_relative_to(results) or experiment.is_relative_to(source):
        raise ValueError('experiment_dir must be a separate directory under project results/')
    config_path = source / 'server-config.json'
    config, recovery = RECOVERY.inspect_run(source, root=root)
    dataset = config['dataset']
    query_path, corpus_path = (root / 'data' / dataset / name for name in ('queries.jsonl', 'corpus.jsonl'))
    queries = load_queries(query_path)
    if len(queries) < SMOKE_QUERIES:
        raise ValueError(f'The dataset must contain at least {SMOKE_QUERIES} queries for the smoke checks')
    smoke_ids = [query.id for query in queries[:SMOKE_QUERIES]]
    full_queries = queries if str(config['limit']) == 'all' else queries[:int(config['limit'])]
    full_ids = [query.id for query in full_queries]
    hashes = {path.name: file_hash(path) for path in (query_path, corpus_path)}
    state_path = experiment / 'experiment.json'
    if stage == 'smoke':
        if legacy_recovery_runs:
            raise ValueError('--legacy-recovery-run applies only to the full stage')
        if model_manifest is None:
            raise ValueError('The smoke stage requires --model-manifest')
        manifest_path = _path(model_manifest, root)
        revisions = _manifest(manifest_path)
        _python(root, True)
        state = {
            'format_version': 1, 'status': 'smoke_running', 'smoke_complete': False,
            'source_run': str(source), 'source_config_sha256': file_hash(config_path),
            'source_revisions': config.get('revisions', {}), 'dataset': dataset, 'full_limit': str(config['limit']),
            'data_hashes': hashes, 'model_manifest': str(manifest_path),
            'model_manifest_sha256': file_hash(manifest_path), 'models': list(MODELS), 'model_revisions': revisions,
            'query_concurrency': query_concurrency, 'gpu_index': str(gpu_index), 'smoke_queries': SMOKE_QUERIES,
            'smoke_runs': _run_paths(experiment, dataset, 'smoke'),
            'full_runs': _run_paths(experiment, dataset, 'full'),
            'smoke_completed_models': [], 'smoke_server_versions': {},
        }
        return _run_smoke_stage(state, experiment, config, smoke_ids, root)
    if not state_path.is_file():
        raise ValueError('Run the smoke stage in this experiment directory before the full stage')
    state = json.loads(state_path.read_text())
    if (state.get('format_version') != 1 or state.get('status') != 'smoke_complete'
            or state.get('smoke_complete') is not True or state.get('models') != list(MODELS)):
        raise ValueError('The full stage requires a successfully completed smoke stage')
    manifest_path = Path(state['model_manifest'])
    if model_manifest is not None and _path(model_manifest, root) != manifest_path:
        raise ValueError('The full stage must use the manifest recorded by the smoke stage')
    revisions = _manifest(manifest_path)
    expected = {
        'source_run': str(source), 'source_config_sha256': file_hash(config_path),
        'source_revisions': config.get('revisions', {}), 'dataset': dataset, 'full_limit': str(config['limit']),
        'data_hashes': hashes, 'model_manifest_sha256': file_hash(manifest_path), 'model_revisions': revisions,
        'query_concurrency': query_concurrency, 'gpu_index': str(gpu_index), 'smoke_queries': SMOKE_QUERIES,
        'smoke_runs': _run_paths(experiment, dataset, 'smoke'), 'full_runs': _run_paths(experiment, dataset, 'full'),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'Experiment {key} changed since the smoke stage; repeat smoke in a fresh directory')
    for model in MODELS:
        _, versions = _validate_results(Path(state['smoke_runs'][model]), model, revisions[model],
                                        smoke_ids, hashes, query_concurrency)
        if versions != state.get('smoke_server_versions', {}).get(model):
            raise ValueError('Smoke server version provenance changed')
    legacy_results = _recover_legacy_results(source, root, config, recovery['completed_summaries'],
                                             legacy_recovery_runs, full_ids, hashes)
    completed = list(legacy_results.values())
    choice_method = METHODS[1]
    missing = [method for method in RECOVERY.METHOD_MODELS
               if method != choice_method and method not in legacy_results
               and (method in recovery['pending_methods'] or method == METHODS[0])]
    modern_python = _python(root, True)
    legacy_directory = experiment / 'legacy-qwen3-choice' / dataset
    for directory in [Path(value) for value in state['full_runs'].values()]:
        if directory.exists():
            raise ValueError(f'Full run output already exists: {directory}')
    legacy_command = None
    if choice_method not in legacy_results:
        legacy_python = _python(root, False)
        if legacy_directory.exists():
            raise ValueError(f'Legacy choice output already exists: {legacy_directory}')
        legacy_command = RECOVERY.recovery_command(
            source, config, (choice_method,), execution_mode='throughput', query_concurrency=query_concurrency,
            gpu_index=gpu_index, run_tag=_tag(legacy_directory, dataset, root), root=root)
        legacy_command[0] = legacy_python
    if missing:
        print(f'Legacy results absent from comparison: {", ".join(missing)}', file=sys.stderr, flush=True)
    state.update(status='full_running', full_completed_models=[], legacy_run=None, missing_legacy_methods=missing)
    _save(state_path, state)
    try:
        if legacy_command is not None:
            subprocess.run(legacy_command, check=True, cwd=root)
            summaries, _ = _validate_results(
                legacy_directory, config['naive_model'], config['revisions'][config['naive_model']],
                full_ids, hashes, query_concurrency, methods=(choice_method,))
            completed.extend(summaries)
            state['legacy_run'] = str(legacy_directory)
            _save(state_path, state)
        for model in MODELS:
            directory = Path(state['full_runs'][model])
            command = _new_model_command(root, config, model, manifest_path, directory,
                                         config['limit'], query_concurrency, gpu_index)
            subprocess.run(command, check=True, cwd=root)
            summaries, versions = _validate_results(directory, model, revisions[model], full_ids, hashes, query_concurrency)
            if versions != state['smoke_server_versions'][model]:
                raise ValueError(f'vLLM version for {model} changed between smoke and full runs')
            completed.extend(summaries)
            state['full_completed_models'].append(model)
            _save(state_path, state)
        subprocess.run([modern_python, str(root / 'scripts/summarize_runs.py'), *completed,
                        '--output', str(experiment / 'comparison')], check=True, cwd=root)
        state.update(status='full_complete', report=str(experiment / 'comparison.md'), report_summaries=completed)
        _save(state_path, state)
    except BaseException:
        state['status'] = 'full_failed'
        with suppress(Exception):
            _save(state_path, state)
        raise
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', help='Original results/<tag>/<dataset> directory')
    parser.add_argument('--stage', choices=['smoke', 'full'], required=True)
    parser.add_argument('--experiment-dir', required=True,
                        help='Directory under project results/; failed smoke stages can retry in the same directory')
    parser.add_argument('--model-manifest', help='Required for smoke; full uses the recorded manifest')
    parser.add_argument('--legacy-recovery-run', dest='legacy_recovery_runs', action='append', default=[],
                        help='Additional legacy results/<tag>/<dataset> directory to reuse in full (repeatable)')
    parser.add_argument('--query-concurrency', type=int, default=8)
    parser.add_argument('--gpu-index', default='1')
    args = parser.parse_args(argv)
    state = run_experiments(args.source_run, stage=args.stage, experiment_dir=args.experiment_dir,
                            model_manifest=args.model_manifest, legacy_recovery_runs=args.legacy_recovery_runs,
                            query_concurrency=args.query_concurrency,
                            gpu_index=args.gpu_index, root=ROOT)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


if __name__ == '__main__':
    main()
