#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-vllm/bin/python scripts/run_vllm_benchmark.py "$@" --ablations
