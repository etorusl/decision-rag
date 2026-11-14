#!/usr/bin/env bash
# Ready-model inference only, using LOCAL vLLM servers on one H100.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-vllm/bin/python scripts/run_vllm_benchmark.py "$@"
