#!/usr/bin/env bash
# Inference-only environment: never install training constraints here.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
python_bin="${DECISION_RAG_PYTHON:-python3.12}"
"$python_bin" -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm==0.11.2' --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[inference,data,dev]'
python -m pip check
python - <<'PY'
import os
import torch
import vllm
assert torch.cuda.is_available(), 'CUDA GPU is required for the local vLLM server'
print({'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
       'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
PY
python -m pip freeze > environment-vllm-h100.txt
printf '%s\n' 'Ready: bash scripts/benchmark_h100.sh multidoc2dial 100'
