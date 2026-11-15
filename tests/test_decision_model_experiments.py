"""Exercise the smoke gate and two-environment workflow without GPUs or downloads."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from decision_rag.evaluation import file_hash

SPEC = importlib.util.spec_from_file_location(
    'decision_model_experiments', Path(__file__).resolve().parents[1] / 'scripts/run_decision_model_experiments.py')
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)

OLD_MODELS = {'naive_model': 'Qwen/Qwen3-8B', 'embedding_model': 'Qwen/Qwen3-Embedding-8B',
              'reranker_model': 'Qwen/Qwen3-Reranker-8B', 'bge_embedding_model': 'BAAI/bge-m3',
              'bge_reranker_model': 'BAAI/bge-reranker-v2-m3'}
BASELINES = ['bm25', 'dense-vllm', 'dense-rerank-vllm', 'dense-bge-vllm', 'dense-rerank-bge-vllm']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def option(command, flag):
    return command[command.index(flag) + 1]


def snapshots(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob('*') if path.is_file()}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    source = root / 'results/20261114T192438644288Z/multidoc2dial'
    destination = root / 'results/new-model-experiments'
    data = root / 'data/multidoc2dial'
    queries = [{'id': f'q{index}', 'text': f'Question {index}', 'relevant_ids': {'p': 1},
                'conversation_id': f'c{index // 3}', 'turn_id': index} for index in range(45)]
    write_rows(data / 'queries.jsonl', queries)
    write_rows(data / 'corpus.jsonl', [
        {'id': 'root', 'kind': 'root', 'text': 'Corpus'},
        {'id': 'page', 'kind': 'page', 'text': 'Page', 'parent_id': 'root'},
        {'id': 'p', 'kind': 'passage', 'text': 'Evidence', 'parent_id': 'page'},
    ])
    config = {'dataset': 'multidoc2dial', 'limit': '30', 'skip_bge': False, 'ablations': False,
              **OLD_MODELS, 'revisions': {model: str(index) * 40 for index, model in enumerate(OLD_MODELS.values(), 1)},
              'max_model_len': 12288, 'pool_max_model_len': 8192, 'bge_max_model_len': 8192,
              'decision_memory': .83, 'pool_memory': .4, 'port_base': 8000, 'startup_timeout': 1200,
              'vllm_batch_size': 32, 'batch_size': 16, 'search_device': 'cuda', 'model_manifest': None}
    write_json(source / 'server-config.json', config)
    manifest = root / 'cache/new-decision-models.json'
    revisions = {model: letter * 40 for model, letter in zip(workflow.MODELS, ('a', 'b'), strict=True)}
    write_json(manifest, {'format_version': 1, 'models': {model: {'revision': revision}
                                                       for model, revision in revisions.items()}})
    for name in ('.venv-vllm', '.venv-vllm-modern'):
        python = root / name / 'bin/python'
        python.parent.mkdir(parents=True)
        python.touch()
    state = {'root': root, 'source': source, 'destination': destination, 'data': data,
             'queries': queries, 'config': config, 'manifest': manifest, 'revisions': revisions,
             'calls': [], 'failure_model': None, 'omit_choice': None, 'wrong_revision': None}

    def result(directory, method, model, revision, count, version='0.24.0', concurrency=8):
        selected = queries if count == 'all' else queries[:int(count)]
        write_rows(directory / method / 'queries.jsonl', [
            {'query_id': query['id'], 'conversation_id': query['conversation_id'], 'turn_id': query['turn_id']}
            for query in selected])
        write_json(directory / method / 'summary.json', {
            'quality': {'recall@10': .75}, 'latency': {'queries': len(selected), 'mean_seconds': .1},
            'execution': {'mode': 'throughput', 'query_concurrency': concurrency},
            'config': {'method': 'decision', 'model': model, 'model_revision': revision,
                       'retrieval': {'choice_k': 20 if method == 'naive-choice-vllm' else 0},
                       'runtime': {'server_version': version}},
            'environment': {'packages': {'vllm': version}},
            'inputs': {str(data / name): file_hash(data / name) for name in ('queries.jsonl', 'corpus.jsonl')},
        })
    state['result'] = result
    for method in [*BASELINES, 'naive-binary-vllm']:
        result(source, method, config['naive_model'], config['revisions'][config['naive_model']],
               config['limit'], version='0.11.2')

    def execute(command, **kwargs):
        state['calls'].append((command, kwargs))
        assert kwargs['check'] is True and kwargs['cwd'] == root
        script = Path(command[1]).name
        if script == 'summarize_runs.py':
            Path(option(command, '--output') + '.md').write_text('Combined comparison\n')
            return subprocess.CompletedProcess(command, 0)
        assert script == 'run_vllm_benchmark.py'  # no downloader/training subprocesses
        model = option(command, '--naive-model')
        dataset, limit = command[2:4]
        directory = root / 'results' / option(command, '--run-tag') / dataset
        if state['failure_model'] == model:
            log = directory / 'servers/decision.log'
            log.parent.mkdir(parents=True)
            log.write_text('vLLM startup failed; preserve this traceback\n')
            raise subprocess.CalledProcessError(1, command)
        assert not directory.exists()
        start = command.index('--methods') + 1
        end = next(index for index in range(start, len(command)) if command[index].startswith('--'))
        methods = command[start:end]
        if '--reuse-revisions' in command:
            pins = json.loads(Path(option(command, '--reuse-revisions')).read_text())['revisions']
        else:
            pins = {name: entry['revision'] for name, entry in json.loads(
                Path(option(command, '--model-manifest')).read_text())['models'].items()}
        version = '0.11.2' if Path(command[0]).parent.parent.name == '.venv-vllm' else '0.24.0'
        write_json(directory / 'server-config.json', {**config, 'methods': methods, 'limit': limit,
                   'naive_model': model, 'revisions': {model: pins[model]}})
        for method in methods:
            if state['omit_choice'] == model and method == 'naive-choice-vllm':
                continue
            revision = 'c' * 40 if state['wrong_revision'] == model else pins[model]
            result(directory, method, model, revision, limit, version,
                   int(option(command, '--query-concurrency')))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(workflow.subprocess, 'run', execute)
    monkeypatch.setattr(workflow, 'ROOT', root)
    return state


def invoke(experiment, stage, **kwargs):
    return workflow.run_experiments(experiment['source'], stage=stage,
                                   experiment_dir=experiment['destination'], root=experiment['root'], **kwargs)


def smoke(experiment):
    return invoke(experiment, 'smoke', model_manifest=experiment['manifest'])


def test_smoke_runs_exactly_twenty_queries_for_both_models_and_both_modes_in_modern_env(experiment):
    original = snapshots(experiment['source'])
    state = smoke(experiment)
    assert state['status'] == 'smoke_complete' and state['smoke_complete'] is True
    assert state['smoke_completed_models'] == list(workflow.MODELS)
    assert len(experiment['calls']) == 2
    for (command, _), model in zip(experiment['calls'], workflow.MODELS, strict=True):
        assert command[0] == str(experiment['root'] / '.venv-vllm-modern/bin/python')
        assert command[2:4] == ['multidoc2dial', '20']
        assert option(command, '--naive-model') == model
        assert option(command, '--model-manifest') == str(experiment['manifest'])
        assert option(command, '--query-concurrency') == '8'
        assert option(command, '--gpu-index') == '1'
        assert option(command, '--max-model-len') == '12288'
        assert option(command, '--decision-memory') == '0.83'
        assert option(command, '--vllm-batch-size') == '32'
        assert '--skip-prepare' in command and '--reuse-revisions' not in command
        start = command.index('--methods') + 1
        assert tuple(command[start:start + 2]) == workflow.METHODS
        assert str(experiment['destination']) in state['smoke_runs'][model]
    assert state['model_revisions'] == experiment['revisions']
    assert snapshots(experiment['source']) == original


def test_full_runs_legacy_choice_then_each_new_model_and_reports_full_results_only(experiment):
    smoke_state = smoke(experiment)
    original = snapshots(experiment['source'])
    experiment['calls'].clear()
    state = invoke(experiment, 'full')
    assert state['status'] == 'full_complete'
    assert len(experiment['calls']) == 4
    legacy, qwen, gemma, report = [command for command, _ in experiment['calls']]
    assert legacy[0] == str(experiment['root'] / '.venv-vllm/bin/python')
    assert option(legacy, '--naive-model') == 'Qwen/Qwen3-8B'
    assert option(legacy, '--reuse-revisions') == str(experiment['source'] / 'server-config.json')
    assert legacy[legacy.index('--methods') + 1] == 'naive-choice-vllm'
    assert 'naive-binary-vllm' not in legacy
    for command, model in zip((qwen, gemma), workflow.MODELS, strict=True):
        assert command[0] == str(experiment['root'] / '.venv-vllm-modern/bin/python')
        assert command[2:4] == ['multidoc2dial', '30']
        assert option(command, '--naive-model') == model
        assert option(command, '--model-manifest') == str(experiment['manifest'])
        assert option(command, '--query-concurrency') == '8'
    assert Path(report[1]).name == 'summarize_runs.py'
    assert len(state['report_summaries']) == 11  # six original + legacy choice + four new full runs
    assert all(path in report for path in state['report_summaries'])
    assert not any('-smoke/' in value for value in report)
    assert all(Path(path).is_file() for path in state['report_summaries'])
    assert state['model_revisions'] == smoke_state['model_revisions']
    assert snapshots(experiment['source']) == original


def test_smoke_may_run_before_original_binary_finishes_but_full_is_blocked(experiment):
    (experiment['source'] / 'naive-binary-vllm/summary.json').unlink()
    smoke(experiment)
    experiment['calls'].clear()
    with pytest.raises(ValueError, match='naive-binary'):
        invoke(experiment, 'full')
    assert experiment['calls'] == []


def test_full_skips_legacy_if_original_choice_already_completed(experiment):
    smoke(experiment)
    config = experiment['config']
    experiment['result'](experiment['source'], 'naive-choice-vllm', config['naive_model'],
                         config['revisions'][config['naive_model']], 30, version='0.11.2')
    experiment['calls'].clear()
    state = invoke(experiment, 'full')
    assert len(experiment['calls']) == 3
    assert all('.venv-vllm-modern/' in command[0] for command, _ in experiment['calls'])
    assert state['legacy_run'] is None
    assert len(state['report_summaries']) == 11


def test_full_preserves_original_all_limit_while_smoke_stays_twenty(experiment):
    config = experiment['config']
    config['limit'] = 'all'
    write_json(experiment['source'] / 'server-config.json', config)
    for method in [*BASELINES, 'naive-binary-vllm']:
        experiment['result'](experiment['source'], method, config['naive_model'],
                             config['revisions'][config['naive_model']], 'all', version='0.11.2')
    smoke(experiment)
    assert all(command[3] == '20' for command, _ in experiment['calls'])
    experiment['calls'].clear()
    invoke(experiment, 'full')
    assert [command[3] for command, _ in experiment['calls'][:3]] == ['all', 'all', 'all']


@pytest.mark.parametrize('broken', ['manifest', 'data', 'source_config', 'smoke_summary', 'smoke_ids', 'concurrency'])
def test_full_gate_revalidates_pins_data_source_and_every_smoke_result(experiment, broken):
    state = smoke(experiment)
    experiment['calls'].clear()
    kwargs = {}
    if broken == 'manifest':
        manifest = json.loads(experiment['manifest'].read_text())
        manifest['models']['Qwen/Qwen3.5-9B']['revision'] = 'd' * 40
        write_json(experiment['manifest'], manifest)
    elif broken == 'data':
        path = experiment['data'] / 'corpus.jsonl'
        path.write_text(path.read_text().replace('Evidence', 'Changed evidence'))
    elif broken == 'source_config':
        config = dict(experiment['config'], decision_memory=.8)
        write_json(experiment['source'] / 'server-config.json', config)
    elif broken == 'smoke_summary':
        path = Path(state['smoke_runs']['google/gemma-4-12B-it']) / 'naive-choice-vllm/summary.json'
        path.unlink()
    elif broken == 'smoke_ids':
        path = Path(state['smoke_runs']['Qwen/Qwen3.5-9B']) / 'naive-binary-vllm/queries.jsonl'
        path.write_text('{"query_id": "wrong"}\n')
    else:
        kwargs['query_concurrency'] = 4
    with pytest.raises(ValueError):
        invoke(experiment, 'full', **kwargs)
    assert experiment['calls'] == []


@pytest.mark.parametrize('failure', ['process', 'missing_choice', 'wrong_pin'])
def test_smoke_failure_stops_before_second_model_and_cannot_open_full_gate(experiment, failure):
    first = next(iter(workflow.MODELS))
    experiment[{'process': 'failure_model', 'missing_choice': 'omit_choice', 'wrong_pin': 'wrong_revision'}[failure]] = first
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        smoke(experiment)
    assert len(experiment['calls']) == 1
    state = json.loads((experiment['destination'] / 'experiment.json').read_text())
    assert state['status'] == 'smoke_failed' and not state['smoke_complete']
    experiment['calls'].clear()
    with pytest.raises(ValueError, match='smoke'):
        invoke(experiment, 'full')
    assert experiment['calls'] == []


@pytest.mark.parametrize('failure', ['process', 'wrong_pin'])
def test_legacy_failure_stops_before_any_new_full_model(experiment, failure):
    smoke(experiment)
    experiment['failure_model' if failure == 'process' else 'wrong_revision'] = 'Qwen/Qwen3-8B'
    experiment['calls'].clear()
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        invoke(experiment, 'full')
    assert len(experiment['calls']) == 1
    state = json.loads((experiment['destination'] / 'experiment.json').read_text())
    assert state['status'] == 'full_failed'
    assert not (experiment['destination'] / 'comparison.md').exists()


def test_new_full_model_failure_stops_before_next_model_and_final_report(experiment):
    smoke(experiment)
    experiment['failure_model'] = 'Qwen/Qwen3.5-9B'
    experiment['calls'].clear()
    with pytest.raises(subprocess.CalledProcessError):
        invoke(experiment, 'full')
    assert [option(command, '--naive-model') for command, _ in experiment['calls']] == [
        'Qwen/Qwen3-8B', 'Qwen/Qwen3.5-9B']
    assert not (experiment['destination'] / 'comparison.md').exists()


def test_full_stage_cannot_run_without_prior_smoke(experiment):
    with pytest.raises(ValueError, match='smoke'):
        invoke(experiment, 'full')
    assert experiment['calls'] == []


def test_smoke_requires_separate_manifest_and_modern_environment(experiment):
    with pytest.raises(ValueError, match='manifest'):
        invoke(experiment, 'smoke')
    (experiment['root'] / '.venv-vllm-modern/bin/python').unlink()
    with pytest.raises(ValueError, match='environment'):
        smoke(experiment)
    assert not experiment['destination'].exists()
    assert experiment['calls'] == []


def test_experiment_must_be_fresh_and_outside_original_run(experiment):
    for destination in (experiment['source'] / 'new', experiment['root'] / 'elsewhere'):
        with pytest.raises(ValueError, match='separate directory'):
            workflow.run_experiments(experiment['source'], stage='smoke', experiment_dir=destination,
                                     model_manifest=experiment['manifest'], root=experiment['root'])
    experiment['destination'].mkdir()
    with pytest.raises(ValueError, match='fresh'):
        smoke(experiment)
    assert experiment['calls'] == []


def test_cli_uses_same_workflow_and_records_reviewable_smoke_state(experiment, capsys):
    workflow.main([str(experiment['source']), '--stage', 'smoke', '--experiment-dir', str(experiment['destination']),
                   '--model-manifest', str(experiment['manifest'])])
    state = json.loads(capsys.readouterr().out)
    assert state['smoke_complete'] is True
    assert state['models'] == ['Qwen/Qwen3.5-9B', 'google/gemma-4-12B-it']


def fail_smoke(experiment, model):
    experiment['failure_model'] = model
    with pytest.raises(subprocess.CalledProcessError):
        smoke(experiment)
    experiment['failure_model'] = None
    experiment['calls'].clear()
    return json.loads((experiment['destination'] / 'experiment.json').read_text())


def test_same_command_retries_failed_smoke_preserving_previous_attempt_files(experiment):
    model = next(iter(workflow.MODELS))
    failed = fail_smoke(experiment, model)
    old_directory = Path(failed['smoke_runs'][model])
    write_json(old_directory / 'server-config.json', {'model': model, 'old_attempt': True})
    old_files = snapshots(old_directory)
    source_files = snapshots(experiment['source'])
    state = smoke(experiment)
    assert state['status'] == 'smoke_complete'
    assert len(experiment['calls']) == 2
    assert state['smoke_runs'] == failed['smoke_runs']
    archive = state['smoke_failed_attempts'][0]
    assert archive['model'] == model
    assert Path(archive['directory']).is_relative_to(experiment['destination'] / 'failed-attempts')
    assert snapshots(Path(archive['directory'])) == old_files
    assert snapshots(experiment['source']) == source_files
    assert (old_directory / 'naive-choice-vllm/summary.json').is_file()


def test_retry_skips_validated_completed_model_then_opens_full_gate(experiment):
    first, second = workflow.MODELS
    failed = fail_smoke(experiment, second)
    assert failed['smoke_completed_models'] == [first]
    completed_files = snapshots(Path(failed['smoke_runs'][first]))
    state = smoke(experiment)
    assert [option(command, '--naive-model') for command, _ in experiment['calls']] == [second]
    assert snapshots(Path(state['smoke_runs'][first])) == completed_files
    assert state['smoke_failed_attempts'][0]['model'] == second
    experiment['calls'].clear()
    full = invoke(experiment, 'full')
    assert full['status'] == 'full_complete'
    assert not any('/failed-attempts/' in value for value in full['report_summaries'])


@pytest.mark.parametrize('changed', ['manifest', 'manifest_path', 'data', 'source_config', 'concurrency', 'gpu'])
def test_failed_smoke_retry_rejects_changed_inputs_without_mutating_results(experiment, changed):
    fail_smoke(experiment, next(iter(workflow.MODELS)))
    kwargs = {'model_manifest': experiment['manifest']}
    if changed == 'manifest':
        value = json.loads(experiment['manifest'].read_text())
        value['models'][next(iter(workflow.MODELS))]['revision'] = 'c' * 40
        write_json(experiment['manifest'], value)
    elif changed == 'manifest_path':
        replacement = experiment['manifest'].with_name('another-manifest.json')
        replacement.write_bytes(experiment['manifest'].read_bytes())
        kwargs['model_manifest'] = replacement
    elif changed == 'data':
        path = experiment['data'] / 'corpus.jsonl'
        path.write_text(path.read_text().replace('Evidence', 'Changed evidence'))
    elif changed == 'source_config':
        write_json(experiment['source'] / 'server-config.json', dict(experiment['config'], decision_memory=.8))
    elif changed == 'concurrency':
        kwargs['query_concurrency'] = 4
    else:
        kwargs['gpu_index'] = '0'
    prior = snapshots(experiment['destination'])
    with pytest.raises(ValueError):
        invoke(experiment, 'smoke', **kwargs)
    assert snapshots(experiment['destination']) == prior
    assert experiment['calls'] == []


def test_retry_revalidates_completed_model_before_archiving_failed_model(experiment):
    first, second = workflow.MODELS
    failed = fail_smoke(experiment, second)
    path = Path(failed['smoke_runs'][first]) / 'naive-choice-vllm/summary.json'
    value = json.loads(path.read_text())
    value['config']['model_revision'] = 'f' * 40
    write_json(path, value)
    prior = snapshots(experiment['destination'])
    with pytest.raises(ValueError, match='model identity or revision'):
        smoke(experiment)
    assert snapshots(experiment['destination']) == prior
    assert experiment['calls'] == []


@pytest.mark.parametrize('status', ['smoke_running', 'full_running', 'full_failed', 'full_complete'])
def test_smoke_rejects_running_and_full_stage_states_without_changes(experiment, status):
    state = fail_smoke(experiment, next(iter(workflow.MODELS)))
    state['status'] = status
    write_json(experiment['destination'] / 'experiment.json', state)
    prior = snapshots(experiment['destination'])
    with pytest.raises(ValueError, match='Cannot retry smoke'):
        smoke(experiment)
    assert snapshots(experiment['destination']) == prior
    assert experiment['calls'] == []


def test_smoke_directory_lock_rejects_a_concurrent_retry_before_mutation(experiment):
    fail_smoke(experiment, next(iter(workflow.MODELS)))
    prior = snapshots(experiment['destination'])
    with workflow._smoke_lock(experiment['destination']), pytest.raises(ValueError, match='already running'):
        smoke(experiment)
    assert snapshots(experiment['destination']) == prior
    assert experiment['calls'] == []


def test_repeating_completed_smoke_revalidates_and_returns_without_rerun(experiment):
    first = smoke(experiment)
    prior = snapshots(experiment['destination'])
    experiment['calls'].clear()
    repeated = smoke(experiment)
    assert repeated == first
    assert snapshots(experiment['destination']) == prior
    assert experiment['calls'] == []


def test_repeated_failures_preserve_each_attempt_separately(experiment):
    model = next(iter(workflow.MODELS))
    fail_smoke(experiment, model)
    second = fail_smoke(experiment, model)
    assert len(second['smoke_failed_attempts']) == 1
    previous_archive = Path(second['smoke_failed_attempts'][0]['directory'])
    previous_files = snapshots(previous_archive)
    complete = smoke(experiment)
    archives = [Path(item['directory']) for item in complete['smoke_failed_attempts']]
    assert len(archives) == len(set(archives)) == 2
    assert snapshots(previous_archive) == previous_files
    assert all((directory / 'servers/decision.log').is_file() for directory in archives)
