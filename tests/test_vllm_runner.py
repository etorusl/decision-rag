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
        command = runner.server_command("vllm", "model", "a" * 40, 8001, kind, .4, 8192)
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
    monkeypatch.setattr(runner.subprocess, "run", run)
    class FakeServers:
        def __init__(self, *args):
            self.kinds = []
            events.append(('server_gpu', args[3]))

        def __enter__(self):
            events.append(("enter",))
            return self

        def start(self, model, revision, port, kind, memory, max_len):
            self.kinds.append(kind)
            events.append(("start", kind))
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
        "bm25", "dense-vllm", "dense-rerank-vllm", "naive-binary-vllm", "naive-choice-vllm"]
    assert all(option(call, "--limit") == "3" for call in benchmarks)
    assert all(option(call, '--gpu-monitor-index') == '1' for call in benchmarks)
    assert set(masks) == {'1'}
    assert [event for event in events if event[0] == 'server_gpu'] == [('server_gpu', '1')] * 2
    for call in benchmarks[1:3]:
        assert option(call, "--engine") == "vllm"
        assert option(call, "--embedding-revision") == "a" * 40
    for call in benchmarks[3:]:
        assert option(call, "--backend") == "vllm"
        assert option(call, "--preset") == "generic"
        assert option(call, "--model-revision") == "a" * 40
    assert events.index(("close", "embedding", "reranker")) < events.index(("start", "decision"))
    assert not any("training" in str(call) or "--backend hf" in str(call) for call in calls)


def test_ablations_run_only_inference_with_no_pooling_servers(fake_run):
    calls, events, _ = fake_run
    runner.main(["multidoc2dial", "all", "--ablations", "--run-tag", "test"])
    benchmarks = [call for call in calls if "benchmark" in call]
    assert len(benchmarks) == 7
    assert all(option(call, "--backend") == "vllm" for call in benchmarks)
    assert all("--limit" not in call for call in benchmarks)
    assert [event for event in events if event[0] == "start"] == [("start", "decision")]
    assert events[-1] == ("close", "decision")


def test_benchmark_failure_still_exits_pooling_context(fake_run, monkeypatch):
    _, events, _ = fake_run
    def fail_dense(command, **kwargs):
        if "--method" in command and option(command, "--method") == "dense":
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(runner.subprocess, "run", fail_dense)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(["scifact", "3", "--run-tag", "failure"])
    assert events[-1] == ("close", "embedding")
    assert ("start", "decision") not in events


@pytest.mark.parametrize('cli_args,expected', [([], '2'), (['--gpu-index', '3'], '3')])
def test_explicit_gpu_selection_routes_servers_clients_and_monitor_together(fake_run, monkeypatch, cli_args, expected):
    calls, events, masks = fake_run
    monkeypatch.setenv('DECISION_RAG_GPU_INDEX', '2')
    runner.main(['scifact', '3', '--run-tag', 'override', *cli_args])
    assert set(masks) == {expected}
    assert [event for event in events if event[0] == 'server_gpu'] == [('server_gpu', expected)] * 2
    assert all(option(call, '--gpu-monitor-index') == expected for call in calls if 'benchmark' in call)


@pytest.mark.parametrize('gpu_index', ['0,1', '-1', 'all'])
def test_gpu_selection_rejects_multiple_or_invalid_devices(gpu_index):
    with pytest.raises(SystemExit) as exc:
        runner.main(['--gpu-index', gpu_index])
    assert exc.value.code == 2
