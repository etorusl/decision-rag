#!/usr/bin/env python3
"""Run ready-model inference benchmarks against owned localhost vLLM processes."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RERANK_OVERRIDES = {
    'architectures': ['Qwen3ForSequenceClassification'],
    'classifier_from_token': ['no', 'yes'], 'is_original_qwen3_reranker': True,
}


def server_command(executable, model, revision, port, kind, memory, max_model_len):
    command = [str(executable), 'serve', model, '--host', '127.0.0.1', '--port', str(port),
               '--served-model-name', model, '--revision', revision, '--tokenizer-revision', revision,
               '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
               '--gpu-memory-utilization', str(memory), '--max-model-len', str(max_model_len),
               '--runner', 'generate' if kind == 'decision' else 'pooling',
               '--max-num-seqs', '64' if kind == 'decision' else '16',
               '--max-num-batched-tokens', str(max(16384, max_model_len)),
               '--generation-config', 'vllm', '--enable-prompt-tokens-details']
    if kind == 'decision':
        command += ['--logprobs-mode', 'processed_logprobs', '--max-logprobs', '26', '--enable-prefix-caching']
    else:
        command += ['--no-enable-prefix-caching']
    if kind == 'reranker':
        command += ['--hf-overrides', json.dumps(RERANK_OVERRIDES, separators=(',', ':'))]
    return command


class Servers:
    def __init__(self, log_dir, executable, startup_timeout=1200, gpu_index='1'):
        self.log_dir, self.executable = Path(log_dir), executable
        self.startup_timeout = startup_timeout
        self.gpu_index = gpu_index
        self.processes = []

    def start(self, model, revision, port, kind, memory, max_model_len):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is occupied. Stop that server or choose another --port-base.')
        command = server_command(self.executable, model, revision, port, kind, memory, max_model_len)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / f'{kind}-command.json').write_text(json.dumps(command, indent=2) + '\n')
        log = (self.log_dir / f'{kind}.log').open('w')
        env = {**os.environ, 'VLLM_USE_V1': '1', 'CUDA_VISIBLE_DEVICES': self.gpu_index}
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.processes.append((process, log))
        endpoint = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + self.startup_timeout
        with httpx.Client(timeout=5, trust_env=False) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'{kind} server exited ({process.returncode}); inspect {log.name}')
                try:
                    response = client.get(endpoint + '/v1/models')
                    if response.status_code == 200 and any(
                        item.get('id') == model for item in response.json().get('data', [])
                    ):
                        return endpoint
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(1)
        raise TimeoutError(f'{kind} startup timed out; inspect {log.name}')

    def close(self):
        for process, log in reversed(self.processes):
            try:
                # Only terminate process groups started by this orchestrator.
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            log.close()
        self.processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='multidoc2dial',
                        choices=['multidoc2dial', 'conditionalqa', 'mtrag', 'scifact'])
    parser.add_argument('limit', nargs='?', default='100', help='Number of queries, or all')
    parser.add_argument('--ablations', action='store_true')
    parser.add_argument('--run-tag', default=os.getenv('RUN_TAG'))
    parser.add_argument('--naive-model', default=os.getenv('NAIVE_MODEL', 'Qwen/Qwen3-8B'))
    parser.add_argument('--embedding-model', default=os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-8B'))
    parser.add_argument('--reranker-model', default=os.getenv('RERANKER_MODEL', 'Qwen/Qwen3-Reranker-8B'))
    parser.add_argument('--port-base', type=int, default=8000)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--pool-max-model-len', type=int, default=8192)
    parser.add_argument('--pool-memory', type=float, default=.40)
    parser.add_argument('--decision-memory', type=float, default=.85)
    parser.add_argument('--gpu-index', default=os.getenv('DECISION_RAG_GPU_INDEX') or '1',
                        help='Single GPU index for CUDA_VISIBLE_DEVICES and nvidia-smi monitoring (default: 1)')
    parser.add_argument('--startup-timeout', type=float, default=1200)
    args = parser.parse_args(argv)
    if not args.gpu_index.isascii() or not args.gpu_index.isdecimal():
        parser.error('--gpu-index must be one nonnegative GPU index')
    if args.limit != 'all' and int(args.limit) < 1:
        parser.error('limit must be positive or all')
    if not 0 < args.pool_memory < .5 or not 0 < args.decision_memory < 1:
        parser.error('Each pooled server must use <0.5 GPU; decision memory must be between0 and1')
    if args.ablations and args.dataset == 'scifact':
        parser.error('Ablations require a structured paper dataset')
    executable = Path(sys.executable).parent / 'vllm'
    if not executable.exists():
        parser.error('Run setup_vllm_h100.sh, then use .venv-vllm/bin/python')
    os.chdir(ROOT)
    tag = args.run_tag or datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = ROOT / 'results' / tag / args.dataset
    if run_dir.exists():
        raise ValueError(f'Run directory already exists: {run_dir}')
    run_dir.mkdir(parents=True)
    from huggingface_hub import HfApi
    api = HfApi()
    models = [args.naive_model] if args.ablations else [args.naive_model, args.embedding_model, args.reranker_model]
    revisions = {model: api.model_info(model).sha for model in dict.fromkeys(models)}
    (run_dir / 'server-config.json').write_text(json.dumps({**vars(args), 'revisions': revisions}, indent=2) + '\n')
    cli = [sys.executable, '-m', 'decision_rag']
    def run(*command):
        subprocess.run([str(x) for x in command], check=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu_index})
    run(*cli, 'prepare', args.dataset, '--output', f'data/{args.dataset}')
    common = ['--corpus', f'data/{args.dataset}/corpus.jsonl', '--queries', f'data/{args.dataset}/queries.jsonl',
              '--top-k', '20', '--warmup', '2', '--gpu-monitor-index', args.gpu_index]
    if args.limit != 'all':
        common += ['--limit', args.limit]
    def benchmark(name, extra):
        run(*cli, 'benchmark', *common, *extra, '--output', run_dir / name)
    if not args.ablations:
        benchmark('bm25', ['--method', 'bm25'])
        with Servers(run_dir / 'servers', executable, args.startup_timeout, args.gpu_index) as servers:
            embedding_url = servers.start(args.embedding_model, revisions[args.embedding_model], args.port_base + 1,
                                          'embedding', args.pool_memory, args.pool_max_model_len)
            dense = ['--engine', 'vllm', '--model', args.embedding_model, '--embedding-url', embedding_url,
                     '--embedding-revision', revisions[args.embedding_model], '--index-dir',
                     f'cache/vllm-dense/{args.dataset}/{args.embedding_model.replace("/", "_")}-{revisions[args.embedding_model]}']
            benchmark('dense-vllm', ['--method', 'dense', *dense])
            reranker_url = servers.start(args.reranker_model, revisions[args.reranker_model], args.port_base + 2,
                                         'reranker', args.pool_memory, args.pool_max_model_len)
            benchmark('dense-rerank-vllm', ['--method', 'dense-rerank', *dense,
                      '--reranker-model', args.reranker_model, '--reranker-url', reranker_url,
                      '--reranker-revision', revisions[args.reranker_model]])
    with Servers(run_dir / 'servers', executable, args.startup_timeout, args.gpu_index) as servers:
        decision_url = servers.start(args.naive_model, revisions[args.naive_model], args.port_base,
                                     'decision', args.decision_memory, args.max_model_len)
        naive = ['--backend', 'vllm', '--model', args.naive_model, '--model-revision', revisions[args.naive_model],
                 '--base-url', decision_url, '--max-context-tokens', str(args.max_model_len),
                 '--max-request-tokens', str(min(30000, args.max_model_len - 256)),
                 '--preset', 'generic' if args.dataset == 'scifact' else args.dataset]
        if args.ablations:
            for pages in (3, 5, 10):
                benchmark(f'pages-{pages}', [*naive, '--pages', str(pages)])
            for size in (1, 16, 64):
                benchmark(f'context-{size}', [*naive, '--max-candidates', str(size)])
            benchmark('no-memory', [*naive, '--no-memory'])
        else:
            benchmark('naive-binary-vllm', [*naive, '--choice-k', '0'])
            benchmark('naive-choice-vllm', [*naive, '--choice-k', '20'])
    run(sys.executable, 'scripts/summarize_runs.py', run_dir, '--output', run_dir / 'comparison')
    print(f'Results: {run_dir}')


if __name__ == '__main__':
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
