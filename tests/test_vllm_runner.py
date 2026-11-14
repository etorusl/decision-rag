"""Exercise launcher ownership and inference-only routing without CUDA or network."""
import importlib.util
import io
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "vllm_runner", Path(__file__).resolve().parents[1] / "scripts" / "run_vllm_benchmark.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def option(command, name):
    return command[command.index(name) + 1]


def test_server_commands_match_scoring_and_cache_contract():
    decision = runner.server_command("vllm", "Qwen/Qwen3-8B", "a" * 40, 8000, "decision", .85, 32768)
    assert option(decision, "--runner") == "generate"
    assert option(decision, "--logprobs-mode") == "processed_logprobs"
    assert option(decision, "--max-logprobs") == "26"
    assert "--enable-prefix-caching" in decision
    assert option(decision, "--generation-config") == "vllm"
    assert option(decision, "--host") == "127.0.0.1"
    assert option(decision, "--revision") == option(decision, "--tokenizer-revision") == "a" * 40
    for kind in ("embedding", "reranker"):
        model = 'Qwen/Qwen3-Reranker-8B' if kind == 'reranker' else 'Qwen/Qwen3-Embedding-8B'
        command = runner.server_command("vllm", model, "a" * 40, 8001, kind, .4, 8192)
        assert option(command, "--runner") == "pooling"
        assert "--no-enable-prefix-caching" in command
        assert "--logprobs-mode" not in command
        assert "--enforce-eager" not in command
        if kind == "reranker":
            assert json.loads(option(command, "--hf-overrides")) == {
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            }


@pytest.mark.parametrize('kind,model', [('embedding', 'BAAI/bge-m3'),
                                       ('reranker', 'BAAI/bge-reranker-v2-m3')])
def test_bge_servers_use_native_pooling_without_qwen_conversion(kind, model):
    command = runner.server_command('vllm', model, 'b' * 40, 8001, kind, .4, 8192)
    assert option(command, '--runner') == 'pooling'
    assert option(command, '--dtype') == 'bfloat16'
    assert option(command, '--max-model-len') == '8192'
    assert '--no-enable-prefix-caching' in command
    assert '--no-enable-chunked-prefill' in command
    assert '--hf-overrides' not in command
    assert '--enforce-eager' not in command


class Probe:
    def __init__(self, occupied=False):
        self.occupied = occupied

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def connect_ex(self, address):
        return 0 if self.occupied else 1


class Process:
    def __init__(self, pid=4321, running=True, first_wait_times_out=False):
        self.pid = pid
        self.returncode = None if running else 1
        self.first_wait_times_out = first_wait_times_out
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.first_wait_times_out and len(self.waits) == 1:
            raise subprocess.TimeoutExpired("vllm", timeout)
        self.returncode = 0
        return 0


def test_occupied_port_never_launches_or_kills_an_existing_server(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.socket, "socket", lambda: Probe(occupied=True))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    monkeypatch.setattr(runner.os, "killpg", lambda *a: pytest.fail("must not kill unowned process"))
    with runner.Servers(tmp_path, "vllm") as servers, pytest.raises(RuntimeError, match="occupied"):
        servers.start("model", "a" * 40, 8000, "decision", .85, 32768)


@pytest.mark.parametrize('gpu_index', ['1', '2'])
def test_failed_startup_cleans_owned_process_group(monkeypatch, tmp_path, gpu_index):
    process = Process(running=False)
    launches, killed = [], []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
    monkeypatch.setattr(runner.socket, "socket", lambda: Probe())
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return process
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.os, "killpg", lambda *args: killed.append(args))
    with pytest.raises(RuntimeError, match="server exited"), runner.Servers(
        tmp_path, "vllm", gpu_index=gpu_index
    ) as servers:
        servers.start("model", "a" * 40, 8000, "decision", .85, 32768)
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["env"]["VLLM_USE_V1"] == "1"
    assert launches[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == gpu_index
    assert launches[0][1]["stdout"].closed
    assert killed == [(process.pid, signal.SIGTERM)]
    assert servers.processes == []


def test_cleanup_escalates_only_owned_group_and_is_idempotent(monkeypatch, tmp_path):
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda *args: killed.append(args))
    process, log = Process(first_wait_times_out=True), io.StringIO()
    servers = runner.Servers(tmp_path, "vllm")
    servers.processes.append((process, log))
    servers.close()
    servers.close()
    assert killed == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert process.waits == [30, 10]
    assert log.closed


def test_logging_failure_does_not_interrupt_remaining_server_cleanup(monkeypatch, tmp_path):
    killed = []
    monkeypatch.setattr(runner.os, 'killpg', lambda *args: killed.append(args))
    class FailingReporter:
        def event(self, *args, **kwargs):
            raise OSError('log filesystem is full')

    processes = [Process(pid=4321), Process(pid=4322)]
    logs = [(tmp_path / f'{process.pid}.log').open('w') for process in processes]
    servers = runner.Servers(tmp_path, 'vllm', reporter=FailingReporter())
    servers.processes.extend(zip(processes, logs, strict=True))
    with pytest.raises(OSError, match='filesystem is full'):
        servers.close()
    assert killed == [(4322, signal.SIGTERM), (4321, signal.SIGTERM)]
    assert all(process.waits == [30] for process in processes)
    assert all(log.closed for log in logs)
    assert servers.processes == []
    servers.close()  # no stale processes or repeated logging failures
    assert len(killed) == 2


def test_readiness_requires_requested_model(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.socket, "socket", lambda: Probe())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(runner.os, "killpg", lambda *a: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    models = iter(["wrong-model", "model"])
    requests = []
    def response(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": next(models)}]})
    client = httpx.Client(transport=httpx.MockTransport(response))
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: client)
    with runner.Servers(tmp_path, "vllm") as servers:
        assert servers.start("model", "a" * 40, 8000, "decision", .85, 32768) == "http://127.0.0.1:8000"
    assert len(requests) == 2
    assert all(request.url.path == "/v1/models" for request in requests)


def test_server_startup_failure_preserves_log_location_and_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.socket, 'socket', lambda: Probe())
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: Process(running=False))
    monkeypatch.setattr(runner.os, 'killpg', lambda *a: None)
    with pytest.raises(RuntimeError, match='server exited'), runner.RunReporter(
        tmp_path, progress='off', name='benchmark-suite'
    ) as reporter, runner.Servers(tmp_path / 'servers', 'vllm', reporter=reporter) as servers:
        servers.start('model', 'a' * 40, 8000, 'decision', .85, 32768)
    log = (tmp_path / 'run.log').read_text()
    assert 'Waiting for decision server' in log
    assert str(tmp_path / 'servers' / 'decision.log') in log
    assert 'RuntimeError' in log
    assert not servers.processes


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    monkeypatch.delenv('DECISION_RAG_GPU_INDEX', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.with_name("vllm").touch()
    monkeypatch.setattr(runner.sys, "executable", str(executable))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda _: SimpleNamespace(sha="a" * 40))))
    calls, events, masks = [], [], []
    def run(command, **kwargs):
        calls.append(command)
        masks.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        assert kwargs['env']['PYTHONUNBUFFERED'] == '1'
        assert 'stdout' not in kwargs and 'stderr' not in kwargs  # keep child terminal progress live
    monkeypatch.setattr(runner.subprocess, "run", run)
    class FakeServers:
        def __init__(self, *args, model_env=None):
            self.kinds = []
            events.append(('server_gpu', args[3]))
            events.append(('server_logs', str(args[0])))
            events.append(('server_env', model_env))

        def __enter__(self):
            events.append(("enter",))
            return self

        def start(self, model, revision, port, kind, memory, max_len, **kwargs):
            self.kinds.append(kind)
            events.append(("start", kind))
            events.append(('server_command', runner.server_command(
                'vllm', model, revision, port, kind, memory, max_len, **kwargs)))
            return f"http://127.0.0.1:{port}"

        def __exit__(self, *args):
            events.append(("close", *self.kinds))
    monkeypatch.setattr(runner, "Servers", FakeServers)
    return calls, events, masks


def test_default_pipeline_uses_ready_vllm_models_and_releases_pooling_before_lm(fake_run):
    calls, events, masks = fake_run
    runner.main(["scifact", "3", "--run-tag", "test"])
    benchmarks = [call for call in calls if "benchmark" in call]
    assert [Path(option(call, "--output")).name for call in benchmarks] == [
        "bm25", "dense-vllm", "dense-rerank-vllm", "dense-bge-vllm", "dense-rerank-bge-vllm",
        "naive-binary-vllm", "naive-choice-vllm"]
    assert all(option(call, "--limit") == "3" for call in benchmarks)
    assert all(option(call, '--gpu-monitor-index') == '1' for call in benchmarks)
    assert all(option(call, '--execution-mode') == 'throughput' for call in benchmarks)
    assert all(option(call, '--query-concurrency') == '8' for call in benchmarks)
    assert all(option(call, '--vllm-batch-size') == '64' for call in benchmarks)
    assert all(option(call, '--batch-size') == '16' for call in benchmarks)
    commands = [event[1] for event in events if event[0] == 'server_command']
    assert [option(command, '--max-num-seqs') for command in commands] == ['64'] * 4 + ['128']
    assert [option(command, '--max-num-batched-tokens') for command in commands] == ['16384'] * 4 + ['32768']
    assert set(masks) == {'1'}
    assert [event for event in events if event[0] == 'server_gpu'] == [('server_gpu', '1')] * 3
    for call in benchmarks[1:5]:
        assert option(call, "--engine") == "vllm"
        assert option(call, "--embedding-revision") == "a" * 40
        assert option(call, '--search-device') == 'cuda'
    assert option(benchmarks[0], '--search-device') == 'cuda'
    assert option(benchmarks[1], '--model') == 'Qwen/Qwen3-Embedding-8B'
    assert option(benchmarks[2], '--reranker-model') == 'Qwen/Qwen3-Reranker-8B'
    assert option(benchmarks[3], '--model') == 'BAAI/bge-m3'
    assert option(benchmarks[4], '--reranker-model') == 'BAAI/bge-reranker-v2-m3'
    assert option(benchmarks[1], '--index-dir') != option(benchmarks[3], '--index-dir')
    for call in benchmarks[5:]:
        assert option(call, "--backend") == "vllm"
        assert option(call, "--preset") == "generic"
        assert option(call, "--model-revision") == "a" * 40
        assert '--search-device' not in call
    assert events.index(("close", "embedding", "reranker")) < events.index(("start", "decision"))
    server_logs = [event[1] for event in events if event[0] == 'server_logs']
    assert server_logs[1] == str(Path(server_logs[0]) / 'bge')
    assert [event[0] for event in events if event[0] in ('enter', 'close')] == ['enter', 'close'] * 3
    assert not any("training" in str(call) or "--backend hf" in str(call) for call in calls)


def test_explicit_latency_profile_preserves_original_scheduler_limits(fake_run):
    calls, events, _ = fake_run
    runner.main(['scifact', '3', '--run-tag', 'latency', '--execution-mode', 'latency'])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert len(benchmarks) == 7
    assert all(option(call, '--execution-mode') == 'latency' for call in benchmarks)
    assert all(option(call, '--query-concurrency') == '1' for call in benchmarks)
    commands = [event[1] for event in events if event[0] == 'server_command']
    assert [option(command, '--max-num-seqs') for command in commands] == ['16'] * 4 + ['64']
    assert [option(command, '--max-num-batched-tokens') for command in commands] == ['16384'] * 4 + ['32768']
    assert [option(command, '--gpu-memory-utilization') for command in commands] == ['0.4'] * 4 + ['0.85']


def test_throughput_tuning_controls_reach_all_clients_and_servers(fake_run, tmp_path):
    calls, events, _ = fake_run
    runner.main(['scifact', '3', '--run-tag', 'tuned', '--query-concurrency', '4',
                 '--decision-max-num-seqs', '256', '--pool-max-num-seqs', '128',
                 '--decision-max-num-batched-tokens', '8192', '--pool-max-num-batched-tokens', '32768',
                 '--vllm-batch-size', '32', '--batch-size', '64'])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert all(option(call, '--query-concurrency') == '4' for call in benchmarks)
    assert all(option(call, '--vllm-batch-size') == '32' for call in benchmarks)
    assert all(option(call, '--batch-size') == '64' for call in benchmarks)
    assert all('--concurrency' not in call for call in benchmarks)  # no nested decision HTTP/chunk concurrency
    commands = [event[1] for event in events if event[0] == 'server_command']
    assert [option(command, '--max-num-seqs') for command in commands] == ['128'] * 4 + ['256']
    assert [option(command, '--max-num-batched-tokens') for command in commands] == ['32768'] * 4 + ['8192']
    config = json.loads((tmp_path / 'results/tuned/scifact/server-config.json').read_text())
    assert config['execution_mode'] == 'throughput'
    assert config['query_concurrency'] == 4
    assert config['pool_max_num_seqs'] == 128


def test_automatic_pool_token_budget_tracks_each_model_context(fake_run):
    _, events, _ = fake_run
    runner.main(['scifact', '3', '--run-tag', 'contexts', '--pool-max-model-len', '32768'])
    commands = [event[1] for event in events if event[0] == 'server_command']
    assert [option(command, '--max-num-batched-tokens') for command in commands[:4]] == [
        '32768', '32768', '16384', '16384']


@pytest.mark.parametrize('options', [
    ['--execution-mode', 'latency', '--query-concurrency', '2'],
    ['--query-concurrency', '0'], ['--decision-max-num-seqs', '0'], ['--pool-max-num-seqs', '-1'],
    ['--decision-max-num-batched-tokens', '0'], ['--pool-max-num-batched-tokens', '-1'],
    ['--vllm-batch-size', '0'], ['--batch-size', '-1'],
    ['--decision-max-num-seqs', '256', '--decision-max-num-batched-tokens', '128'],
    ['--pool-max-num-seqs', '32768'], ['--pool-max-num-batched-tokens', '4096'],
])
def test_invalid_throughput_settings_fail_before_launch(fake_run, options):
    calls, events, _ = fake_run
    with pytest.raises(SystemExit):
        runner.main(['scifact', '3', '--run-tag', 'invalid', *options])
    assert not calls
    assert not events


def test_original_matrix_remains_available_unchanged(fake_run):
    calls, events, _ = fake_run
    runner.main(['scifact', '3', '--run-tag', 'original', '--skip-bge'])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert [Path(option(call, '--output')).name for call in benchmarks] == [
        'bm25', 'dense-vllm', 'dense-rerank-vllm', 'naive-binary-vllm', 'naive-choice-vllm']
    assert all('BAAI/' not in str(call) for call in calls)
    assert [event for event in events if event[0] == 'server_gpu'] == [('server_gpu', '1')] * 2


@pytest.mark.parametrize('method,models,kinds', [
    ('bm25', [], []),
    ('dense-vllm', ['Qwen/Qwen3-Embedding-8B'], ['embedding']),
    ('dense-rerank-vllm', ['Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Reranker-8B'], ['embedding', 'reranker']),
    ('dense-bge-vllm', ['BAAI/bge-m3'], ['embedding']),
    ('dense-rerank-bge-vllm', ['BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3'], ['embedding', 'reranker']),
    ('naive-binary-vllm', ['Qwen/Qwen3-8B'], ['decision']),
    ('naive-choice-vllm', ['Qwen/Qwen3-8B'], ['decision']),
])
def test_selected_method_only_resolves_and_starts_its_dependencies(fake_run, monkeypatch, tmp_path,
                                                               method, models, kinds):
    calls, events, _ = fake_run
    resolved = []
    def model_info(model):
        resolved.append(model)
        return SimpleNamespace(sha='c' * 40)
    def api():
        assert models, 'BM25 must not construct a Hugging Face client'
        return SimpleNamespace(model_info=model_info)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=api))
    runner.main(['scifact', '3', '--run-tag', 'selected', '--methods', method])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert [Path(option(call, '--output')).name for call in benchmarks] == [method]
    assert resolved == models
    assert [event[1] for event in events if event[0] == 'start'] == kinds
    commands = [event[1] for event in events if event[0] == 'server_command']
    assert [command[2] for command in commands] == models
    config = json.loads((tmp_path / 'results/selected/scifact/server-config.json').read_text())
    assert config['methods'] == [method]
    assert config['revisions'] == dict.fromkeys(models, 'c' * 40)
    records = [json.loads(line) for line in (tmp_path / 'results/selected/scifact/events.jsonl').read_text().splitlines()]
    completed = next(record for record in records if record['event'] == 'suite_completed')
    assert completed['completed'] == completed['total'] == 1


def test_selected_methods_keep_canonical_order_and_share_decision_server(fake_run):
    calls, events, _ = fake_run
    runner.main(['scifact', '3', '--run-tag', 'ordered', '--methods',
                 'naive-choice-vllm', 'bm25', 'naive-binary-vllm'])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert [Path(option(call, '--output')).name for call in benchmarks] == [
        'bm25', 'naive-binary-vllm', 'naive-choice-vllm']
    assert [event for event in events if event[0] == 'start'] == [('start', 'decision')]


@pytest.mark.parametrize('options', [
    ['--methods', 'bm25', 'bm25'],
    ['--methods', 'naive-binary-vllm', '--ablations'],
    ['--methods', 'dense-bge-vllm', '--skip-bge'],
    ['--methods', 'dense-rerank-bge-vllm', '--skip-bge'],
])
def test_incompatible_method_selection_fails_before_launch(fake_run, options):
    calls, events, _ = fake_run
    with pytest.raises(SystemExit):
        runner.main(['multidoc2dial', '3', '--run-tag', 'invalid-methods', *options])
    assert not calls and not events


def test_reuse_revisions_uses_exact_previous_pin_without_online_resolution(fake_run, monkeypatch, tmp_path):
    calls, events, _ = fake_run
    previous = tmp_path / 'previous-server-config.json'
    previous.write_text(json.dumps({'revisions': {'Qwen/Qwen3-8B': 'd' * 40,
                                                 'unused/model': 'mutable-unused-value'}}))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: pytest.fail('Reusing revisions must not query Hugging Face')))
    runner.main(['scifact', '3', '--run-tag', 'reuse', '--reuse-revisions', str(previous),
                 '--methods', 'naive-binary-vllm', 'naive-choice-vllm'])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert all(option(call, '--model-revision') == 'd' * 40 for call in benchmarks)
    commands = [event[1] for event in events if event[0] == 'server_command']
    assert len(commands) == 1 and option(commands[0], '--revision') == 'd' * 40
    config = json.loads((tmp_path / 'results/reuse/scifact/server-config.json').read_text())
    assert config['revisions'] == {'Qwen/Qwen3-8B': 'd' * 40}


@pytest.mark.parametrize('revisions', [{}, {'Qwen/Qwen3-8B': 'main'}, [], {'Qwen/Qwen3-8B': 1 * 10**39}])
def test_invalid_previous_pins_fail_before_prepare_or_servers(fake_run, tmp_path, revisions):
    calls, events, _ = fake_run
    previous = tmp_path / 'previous-server-config.json'
    previous.write_text(json.dumps({'revisions': revisions}))
    with pytest.raises(ValueError, match='revision'):
        runner.main(['scifact', '3', '--run-tag', 'invalid-pins', '--reuse-revisions', str(previous),
                     '--methods', 'naive-binary-vllm'])
    assert not calls and not events


def test_skip_prepare_reuses_existing_dataset_bytes(fake_run, tmp_path):
    calls, _, _ = fake_run
    data = tmp_path / 'data/scifact'
    data.mkdir(parents=True)
    for filename in ('corpus.jsonl', 'queries.jsonl'):
        (data / filename).write_text('preserve existing prepared bytes\n')
    runner.main(['scifact', '3', '--run-tag', 'prepared', '--methods', 'bm25', '--skip-prepare'])
    assert not any('prepare' in call for call in calls)
    assert all((data / filename).read_text() == 'preserve existing prepared bytes\n'
               for filename in ('corpus.jsonl', 'queries.jsonl'))
    benchmark = next(call for call in calls if 'benchmark' in call)
    assert option(benchmark, '--corpus') == 'data/scifact/corpus.jsonl'
    assert option(benchmark, '--queries') == 'data/scifact/queries.jsonl'


@pytest.mark.parametrize('missing', ['corpus.jsonl', 'queries.jsonl'])
def test_skip_prepare_fails_before_resolution_if_dataset_file_is_missing(fake_run, tmp_path, monkeypatch, missing):
    calls, events, _ = fake_run
    data = tmp_path / 'data/scifact'
    data.mkdir(parents=True)
    (data / ('corpus.jsonl' if missing == 'queries.jsonl' else 'queries.jsonl')).touch()
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: pytest.fail('Missing prepared files must be rejected before model resolution')))
    with pytest.raises(ValueError, match=missing):
        runner.main(['scifact', '3', '--run-tag', 'missing-data', '--skip-prepare'])
    assert not calls and not events


def make_manifest(tmp_path):
    models = ['Qwen/Qwen3-8B', 'Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Reranker-8B',
              'BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3']
    cache = tmp_path / 'hf-cache'
    entries = {}
    for model in models:
        snapshot = cache / ('models--' + model.replace('/', '--')) / 'snapshots' / ('b' * 40)
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        entries[model] = {'revision': 'b' * 40, 'snapshot_path': str(snapshot), 'files': ['config.json']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'format_version': 1, 'cache_dir': str(cache), 'models': entries}))
    return path


@pytest.mark.parametrize('same_revision', [True, False])
def test_selected_model_manifest_must_match_reused_revision(fake_run, tmp_path, monkeypatch, same_revision):
    calls, events, _ = fake_run
    manifest = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document['models'] = {'Qwen/Qwen3-8B': document['models']['Qwen/Qwen3-8B']}
    manifest.write_text(json.dumps(document))
    previous = tmp_path / 'previous-server-config.json'
    previous.write_text(json.dumps({'revisions': {'Qwen/Qwen3-8B': ('b' if same_revision else 'c') * 40}}))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: pytest.fail('Pinned model selection must not query Hugging Face')))
    arguments = ['scifact', '3', '--run-tag', 'manifest-pins', '--methods', 'naive-binary-vllm',
                 '--model-manifest', str(manifest), '--reuse-revisions', str(previous)]
    if same_revision:
        runner.main(arguments)
        benchmark = next(call for call in calls if 'benchmark' in call)
        assert option(benchmark, '--model-revision') == 'b' * 40
        assert [event for event in events if event[0] == 'start'] == [('start', 'decision')]
    else:
        with pytest.raises(ValueError, match='differs'):
            runner.main(arguments)
        assert not calls and not events


@pytest.mark.parametrize('use_manifest', [True, False])
def test_manifest_cache_overrides_legacy_parent_cache_in_server(monkeypatch, tmp_path, use_manifest):
    legacy_names = ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE')
    for name in ('HF_HUB_CACHE', *legacy_names):
        monkeypatch.setenv(name, str(tmp_path / 'wrong-cache'))
    _, model_env = runner.cached_revisions(make_manifest(tmp_path), ['Qwen/Qwen3-8B'])
    environments = []
    def popen(command, **kwargs):
        environments.append(kwargs['env'])
        return Process(running=False)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.socket, 'socket', lambda: Probe())
    monkeypatch.setattr(runner.os, 'killpg', lambda *args: None)
    with pytest.raises(RuntimeError, match='server exited'), runner.Servers(
        tmp_path / 'servers', 'vllm', model_env=model_env if use_manifest else None
    ) as servers:
        servers.start('Qwen/Qwen3-8B', 'b' * 40, 8000, 'decision', .85, 32768)
    env = environments[0]
    assert env['HF_HUB_CACHE'] == str(tmp_path / ('hf-cache' if use_manifest else 'wrong-cache'))
    for name in legacy_names:
        if use_manifest:
            assert name not in env
        else:
            assert env[name] == str(tmp_path / 'wrong-cache')
        assert runner.os.environ[name] == str(tmp_path / 'wrong-cache')


def test_manifest_pins_models_offline_without_disabling_dataset_downloads(fake_run, tmp_path, monkeypatch):
    _, events, _ = fake_run
    manifest = make_manifest(tmp_path)
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    legacy_names = ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE')
    for name in ('HF_HUB_CACHE', *legacy_names):
        monkeypatch.setenv(name, str(tmp_path / 'wrong-cache'))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: pytest.fail('A downloaded manifest must not query Hugging Face')))
    environments = []
    original = runner.subprocess.run
    def capture(command, **kwargs):
        environments.append((command, kwargs['env']))
        return original(command, **kwargs)
    monkeypatch.setattr(runner.subprocess, 'run', capture)
    runner.main(['scifact', '3', '--run-tag', 'manifest', '--model-manifest', str(manifest)])
    for command, env in environments:
        if 'benchmark' in command:
            assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
            assert env['HF_HUB_CACHE'] == str(tmp_path / 'hf-cache')
            assert all(name not in env for name in legacy_names)
        else:
            assert 'HF_HUB_OFFLINE' not in env
            assert all(env[name] == str(tmp_path / 'wrong-cache') for name in ('HF_HUB_CACHE', *legacy_names))
    for event in events:
        if event[0] == 'server_env':
            assert event[1] == {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                'HF_HUB_CACHE': str(tmp_path / 'hf-cache')}
    config = json.loads((tmp_path / 'results/manifest/scifact/server-config.json').read_text())
    assert set(config['revisions'].values()) == {'b' * 40}


@pytest.mark.parametrize('problem', ['missing_model', 'mutable_revision', 'missing_file', 'wrong_snapshot'])
def test_incomplete_manifest_fails_before_any_inference(fake_run, tmp_path, problem):
    calls, events, _ = fake_run
    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    entry = manifest['models']['BAAI/bge-m3']
    if problem == 'missing_model':
        del manifest['models']['BAAI/bge-m3']
    elif problem == 'mutable_revision':
        entry['revision'] = 'main'
    elif problem == 'missing_file':
        (Path(entry['snapshot_path']) / 'config.json').unlink()
    else:
        entry['snapshot_path'] = manifest['models']['Qwen/Qwen3-8B']['snapshot_path']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.main(['scifact', '3', '--run-tag', problem, '--model-manifest', str(path)])
    assert not calls
    assert not any(event[0] == 'start' for event in events)


def test_ablations_run_only_inference_with_no_pooling_servers(fake_run):
    calls, events, _ = fake_run
    runner.main(["multidoc2dial", "all", "--ablations", "--run-tag", "test"])
    benchmarks = [call for call in calls if "benchmark" in call]
    assert len(benchmarks) == 7
    assert all(option(call, "--backend") == "vllm" for call in benchmarks)
    assert all("--limit" not in call for call in benchmarks)
    assert all('--search-device' not in call for call in benchmarks)
    assert [event for event in events if event[0] == "start"] == [("start", "decision")]
    assert events[-1] == ("close", "decision")


def test_cpu_search_requires_explicit_selection_and_is_recorded(fake_run, tmp_path):
    calls, _, masks = fake_run
    runner.main(['scifact', '3', '--run-tag', 'cpu-search', '--search-device', 'cpu'])
    dense = [call for call in calls if '--engine' in call]
    assert len(dense) == 4
    assert all(option(call, '--search-device') == 'cpu' for call in dense)
    bm25 = next(call for call in calls if '--method' in call and option(call, '--method') == 'bm25')
    assert option(bm25, '--search-device') == 'cpu'
    assert set(masks) == {'1'}  # Model inference remains on the selected GPU.
    config = json.loads((tmp_path / 'results/cpu-search/scifact/server-config.json').read_text())
    assert config['search_device'] == 'cpu'


def test_benchmark_failure_still_exits_pooling_context(fake_run, monkeypatch, tmp_path):
    _, events, _ = fake_run
    def fail_dense(command, **kwargs):
        if "--method" in command and option(command, "--method") == "dense":
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(runner.subprocess, "run", fail_dense)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(["scifact", "3", "--run-tag", "failure"])
    assert events[-1] == ("close", "embedding")
    assert ("start", "decision") not in events
    output = tmp_path / 'results' / 'failure' / 'scifact'
    assert 'CalledProcessError' in (output / 'run.log').read_text()
    records = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert not any(record['event'] == 'suite_completed' for record in records)
    assert [record['benchmark'] for record in records if record['event'] == 'benchmark_completed'] == ['bm25']


def test_suite_reports_stages_and_forwards_progress_settings(fake_run, tmp_path, capsys):
    calls, _, _ = fake_run
    runner.main(['scifact', '3', '--run-tag', 'logging', '--progress', 'off', '--log-interval', '7'])
    benchmarks = [call for call in calls if 'benchmark' in call]
    assert all(option(call, '--progress') == 'off' for call in benchmarks)
    assert all(float(option(call, '--log-interval')) == 7 for call in benchmarks)
    output = tmp_path / 'results' / 'logging' / 'scifact'
    records = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert [row['completed'] for row in records if row['event'] == 'benchmark_completed'] == list(range(1, 8))
    completed = next(row for row in records if row['event'] == 'suite_completed')
    assert completed['completed'] == completed['total'] == 7
    assert 'Completed 7/7: naive-choice-vllm: model=Qwen/Qwen3-8B' in (output / 'run.log').read_text()
    assert 'reranker_model=BAAI/bge-reranker-v2-m3' in (output / 'run.log').read_text()
    captured = capsys.readouterr()
    assert 'Benchmark 1/7: bm25' in captured.err
    assert '\r' not in captured.err and '\x1b[' not in captured.err


@pytest.mark.parametrize('cli_args,expected', [([], '2'), (['--gpu-index', '3'], '3')])
def test_explicit_gpu_selection_routes_servers_clients_and_monitor_together(fake_run, monkeypatch, cli_args, expected):
    calls, events, masks = fake_run
    monkeypatch.setenv('DECISION_RAG_GPU_INDEX', '2')
    runner.main(['scifact', '3', '--run-tag', 'override', *cli_args])
    assert set(masks) == {expected}
    assert [event for event in events if event[0] == 'server_gpu'] == [('server_gpu', expected)] * 3
    assert all(option(call, '--gpu-monitor-index') == expected for call in calls if 'benchmark' in call)


@pytest.mark.parametrize('gpu_index', ['0,1', '-1', 'all'])
def test_gpu_selection_rejects_multiple_or_invalid_devices(gpu_index):
    with pytest.raises(SystemExit) as exc:
        runner.main(['--gpu-index', gpu_index])
    assert exc.value.code == 2
