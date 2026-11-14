"""Recovery reuses validated completed runs and restarts interrupted methods cleanly."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from decision_rag.evaluation import file_hash

SPEC = importlib.util.spec_from_file_location(
    'resume_vllm_benchmark', Path(__file__).resolve().parents[1] / 'scripts/resume_vllm_benchmark.py')
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)

METHODS = ['bm25', 'dense-vllm', 'dense-rerank-vllm', 'dense-bge-vllm', 'dense-rerank-bge-vllm',
           'naive-binary-vllm', 'naive-choice-vllm']
MODEL_NAMES = {'naive_model': 'Qwen/Qwen3-8B', 'embedding_model': 'Qwen/Qwen3-Embedding-8B',
               'reranker_model': 'Qwen/Qwen3-Reranker-8B', 'bge_embedding_model': 'BAAI/bge-m3',
               'bge_reranker_model': 'BAAI/bge-reranker-v2-m3'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def option(command, flag):
    return command[command.index(flag) + 1]


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob('*') if path.is_file()}


@pytest.fixture
def old_run(tmp_path):
    root = tmp_path / 'project'
    data = root / 'data' / 'multidoc2dial'
    write_rows(data / 'corpus.jsonl', [
        {'id': 'root', 'kind': 'root', 'text': 'Corpus'},
        {'id': 'page', 'kind': 'page', 'text': 'Page', 'parent_id': 'root'},
        {'id': 'p', 'kind': 'passage', 'text': 'Evidence', 'parent_id': 'page'},
    ])
    queries = [{'id': f'q{index}', 'text': f'Question {index}', 'relevant_ids': {'p': 1},
                'conversation_id': 'conversation', 'turn_id': index} for index in range(1, 6)]
    write_rows(data / 'queries.jsonl', queries)
    run = root / 'results' / 'old-run' / 'multidoc2dial'
    config = {
        'dataset': 'multidoc2dial', 'limit': '3', 'ablations': False, 'skip_bge': False,
        'run_tag': 'old-run', 'execution_mode': 'latency', 'query_concurrency': 1,
        **MODEL_NAMES, 'revisions': {name: str(index) * 40 for index, name in enumerate(MODEL_NAMES.values(), 1)},
        'max_model_len': 32768, 'pool_max_model_len': 8192, 'bge_max_model_len': 8192,
        'pool_memory': .4, 'decision_memory': .85, 'search_device': 'cuda',
        'port_base': 8000, 'startup_timeout': 1200, 'gpu_index': '1', 'model_manifest': None,
        'decision_max_num_seqs': 64, 'pool_max_num_seqs': 16,
        'decision_max_num_batched_tokens': 32768, 'pool_max_num_batched_tokens': None,
        'vllm_batch_size': 64, 'batch_size': 16, 'progress': 'off', 'log_interval': 30,
    }
    write_json(run / 'server-config.json', config)
    (run / 'run.log').write_text('Stopped during naive binary.\n')
    return root, run, config, queries


def complete(old_run, method, *, summary_update=None, rows=None):
    root, run, config, queries = old_run
    expected = queries if config['limit'] == 'all' else queries[:int(config['limit'])]
    data = root / 'data' / config['dataset']
    summary = {
        'quality': {'recall@10': 1.0}, 'latency': {'queries': len(expected), 'mean_seconds': .2},
        'execution': {'mode': 'latency', 'query_concurrency': 1},
        'inputs': {str(data / name): file_hash(data / name) for name in ('corpus.jsonl', 'queries.jsonl')},
    }
    summary.update(summary_update or {})
    write_json(run / method / 'summary.json', summary)
    write_rows(run / method / 'queries.jsonl', rows if rows is not None else [
        {'query_id': query['id'], 'conversation_id': query['conversation_id'], 'turn_id': query['turn_id'],
         'latency_seconds': .2, 'metrics': {'recall@10': 1.0}} for query in expected])


def interrupted_binary(old_run):
    for method in METHODS[:5]:
        complete(old_run, method)
    path = old_run[1] / 'naive-binary-vllm' / 'queries.jsonl'
    path.parent.mkdir()
    path.write_text('{"query_id": "q1", "latency_seconds": 9.0}\n{"query_id":')


def plan(old_run, **kwargs):
    root, run, *_ = old_run
    return resume.plan_recovery(run, root=root, run_tag='recovered', **kwargs)


def test_interrupted_binary_restarts_entire_method_in_new_run_with_new_concurrency(old_run):
    interrupted_binary(old_run)
    root, old, config, _ = old_run
    before = snapshot(old)
    result = plan(old_run)
    assert result['source_run'] == str(old)
    assert result['new_run'] == str(root / 'results/recovered/multidoc2dial')
    assert result['pending_methods'] == METHODS[5:]
    assert result['partial_methods'] == ['naive-binary-vllm']
    assert result['completed_summaries'] == [str(old / method / 'summary.json') for method in METHODS[:5]]
    assert result['data_hash_validation'] == 'completed_summaries'
    command = result['command']
    assert 'run_vllm_benchmark.py' in command[1]
    assert command[2:4] == ['multidoc2dial', '3']
    assert option(command, '--execution-mode') == 'throughput'
    assert option(command, '--query-concurrency') == '8'
    assert option(command, '--gpu-index') == '1'
    assert option(command, '--run-tag') == 'recovered'
    assert option(command, '--reuse-revisions') == str(old / 'server-config.json')
    assert '--skip-prepare' in command
    methods_index = command.index('--methods') + 1
    assert command[methods_index:methods_index + 2] == METHODS[5:]
    assert option(command, '--naive-model') == config['naive_model']
    assert option(command, '--max-model-len') == str(config['max_model_len'])
    assert option(command, '--decision-memory') == str(config['decision_memory'])
    assert snapshot(old) == before
    assert not Path(result['new_run']).exists()


def test_completed_runs_must_match_current_data_hashes(old_run):
    complete(old_run, 'bm25')
    root, _, _, _ = old_run
    path = root / 'data/multidoc2dial/corpus.jsonl'
    path.write_text(path.read_text().replace('Evidence', 'Changed evidence'))
    with pytest.raises(ValueError, match='hash|changed|match'):
        plan(old_run)


@pytest.mark.parametrize('invalid', ['wrong_count', 'missing_rows', 'wrong_ids', 'duplicate_ids',
                                    'wrong_order', 'truncated_rows', 'malformed_summary', 'missing_hashes',
                                    'wrong_quality_type'])
def test_present_summary_is_never_silently_trusted_or_discarded_when_invalid(old_run, invalid):
    complete(old_run, 'bm25')
    _, old, _, _ = old_run
    summary_path, rows_path = old / 'bm25/summary.json', old / 'bm25/queries.jsonl'
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if invalid == 'wrong_count':
        summary['latency']['queries'] = 2
    elif invalid == 'missing_rows':
        rows_path.unlink()
    elif invalid == 'wrong_ids':
        rows[0]['query_id'] = 'different'
    elif invalid == 'duplicate_ids':
        rows[1]['query_id'] = rows[0]['query_id']
    elif invalid == 'wrong_order':
        rows.reverse()
    elif invalid == 'truncated_rows':
        rows_path.write_text(rows_path.read_text() + '{')
    elif invalid == 'malformed_summary':
        summary_path.write_text('{')
    elif invalid == 'missing_hashes':
        summary['inputs'] = {}
    else:
        summary['quality'] = []
    if invalid in {'wrong_count', 'missing_hashes', 'wrong_quality_type'}:
        write_json(summary_path, summary)
    if invalid in {'wrong_ids', 'duplicate_ids', 'wrong_order'}:
        write_rows(rows_path, rows)
    before = snapshot(old)
    with pytest.raises(ValueError):
        plan(old_run)
    assert snapshot(old) == before


def test_no_completed_anchor_allows_full_restart_but_records_missing_historical_validation(old_run):
    result = plan(old_run)
    assert result['pending_methods'] == METHODS
    assert result['completed_summaries'] == []
    assert result['data_hash_validation'] == 'unavailable'


def test_all_complete_is_noop_without_new_directory_or_server_command(old_run):
    for method in METHODS:
        complete(old_run, method)
    before = snapshot(old_run[0])
    result = plan(old_run)
    assert result['pending_methods'] == result['partial_methods'] == []
    assert result['command'] is None and result['new_run'] is None
    assert snapshot(old_run[0]) == before


def test_limit_all_and_explicit_original_method_subset_are_respected(old_run):
    _, old, config, _ = old_run
    config.update(limit='all', methods=['bm25', 'naive-binary-vllm'])
    write_json(old / 'server-config.json', config)
    complete(old_run, 'bm25')
    result = plan(old_run)
    assert result['pending_methods'] == ['naive-binary-vllm']
    assert result['command'][2:4] == ['multidoc2dial', 'all']


def test_skip_bge_recovery_does_not_invent_new_methods(old_run):
    _, old, config, _ = old_run
    config['skip_bge'] = True
    write_json(old / 'server-config.json', config)
    result = plan(old_run)
    assert result['pending_methods'] == [method for method in METHODS if 'bge' not in method]


def test_pinned_revision_is_required_only_for_models_still_needed(old_run):
    interrupted_binary(old_run)
    _, old, config, _ = old_run
    config['revisions'] = {config['naive_model']: 'a' * 40}
    write_json(old / 'server-config.json', config)
    assert plan(old_run)['pending_methods'] == METHODS[5:]
    config['revisions'][config['naive_model']] = 'main'
    write_json(old / 'server-config.json', config)
    with pytest.raises(ValueError, match='revision|SHA|pin'):
        plan(old_run)


def test_missing_required_model_identity_never_falls_back_to_current_environment(old_run, monkeypatch):
    interrupted_binary(old_run)
    _, old, config, _ = old_run
    del config['naive_model']
    write_json(old / 'server-config.json', config)
    monkeypatch.setenv('NAIVE_MODEL', 'unexpected/replacement')
    with pytest.raises(ValueError, match='model'):
        plan(old_run)


def test_manifest_from_old_config_is_resolved_against_project_and_can_be_overridden(old_run):
    root, old, config, _ = old_run
    original_manifest, override = root / 'cache/manifest.json', root / 'other-manifest.json'
    write_json(original_manifest, {'fixture': True})
    write_json(override, {'fixture': 'override'})
    config['model_manifest'] = 'cache/manifest.json'
    write_json(old / 'server-config.json', config)
    assert option(plan(old_run)['command'], '--model-manifest') == str(original_manifest)
    assert option(plan(old_run, model_manifest=str(override))['command'], '--model-manifest') == str(override)
    with pytest.raises(ValueError, match='manifest|exist'):
        plan(old_run, model_manifest=str(root / 'missing.json'))


@pytest.mark.parametrize('update', [
    {'dataset': 'unknown'}, {'limit': '0'}, {'limit': '-2'}, {'limit': 'bad'}, {'ablations': True},
])
def test_invalid_source_protocol_is_rejected_before_any_execution(old_run, update):
    _, old, config, _ = old_run
    config.update(update)
    write_json(old / 'server-config.json', config)
    with pytest.raises(ValueError):
        plan(old_run)


def test_existing_destination_and_original_tag_cannot_be_overwritten(old_run):
    root, old, *_ = old_run
    (root / 'results/recovered/multidoc2dial').mkdir(parents=True)
    with pytest.raises(ValueError, match='exist|overwrite'):
        plan(old_run)
    with pytest.raises(ValueError, match='exist|overwrite|different|source'):
        resume.plan_recovery(old, root=root, run_tag='old-run')


def test_dry_run_prints_reviewable_plan_without_subprocesses_or_mutations(old_run, monkeypatch, capsys):
    interrupted_binary(old_run)
    root, old, *_ = old_run
    monkeypatch.setattr(resume, 'ROOT', root)
    monkeypatch.setattr(resume.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Dry run must not launch'))
    before = snapshot(root)
    resume.main([str(old), '--run-tag', 'recovered', '--dry-run'])
    result = json.loads(capsys.readouterr().out)
    assert result['pending_methods'] == METHODS[5:]
    assert option(result['command'], '--query-concurrency') == '8'
    assert snapshot(root) == before


def test_main_runs_pending_methods_then_reports_old_complete_and_new_run_only(old_run, monkeypatch):
    interrupted_binary(old_run)
    root, old, *_ = old_run
    monkeypatch.setattr(resume, 'ROOT', root)
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            (root / 'results/recovered/multidoc2dial').mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(resume.subprocess, 'run', execute)
    before = snapshot(old)
    resume.main([str(old), '--run-tag', 'recovered', '--execution-mode', 'throughput', '--query-concurrency', '4'])
    assert len(calls) == 2
    command, kwargs = calls[0]
    assert option(command, '--query-concurrency') == '4'
    assert kwargs['check'] is True and Path(kwargs['cwd']) == root
    report = calls[1][0]
    assert 'summarize_runs.py' in report[1]
    assert all(str(old / method / 'summary.json') in report for method in METHODS[:5])
    assert str(old / 'naive-binary-vllm') not in report
    assert str(old) not in report  # recursive discovery must not include an interrupted old method
    assert str(root / 'results/recovered/multidoc2dial') in report
    assert option(report, '--output') == str(root / 'results/recovered/multidoc2dial/recovery-comparison')
    assert snapshot(old) == before


def test_new_run_failure_does_not_generate_a_successful_combined_report(old_run, monkeypatch):
    interrupted_binary(old_run)
    root, old, *_ = old_run
    monkeypatch.setattr(resume, 'ROOT', root)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(resume.subprocess, 'run', fail)
    before = snapshot(old)
    with pytest.raises(subprocess.CalledProcessError):
        resume.main([str(old), '--run-tag', 'recovered'])
    assert len(calls) == 1
    assert snapshot(old) == before


def test_main_all_complete_does_not_launch_or_write_reports(old_run, monkeypatch):
    for method in METHODS:
        complete(old_run, method)
    root, old, *_ = old_run
    monkeypatch.setattr(resume, 'ROOT', root)
    monkeypatch.setattr(resume.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Completed run needs no subprocess'))
    before = snapshot(root)
    resume.main([str(old)])
    assert snapshot(root) == before
