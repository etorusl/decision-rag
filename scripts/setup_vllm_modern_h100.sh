#!/usr/bin/env bash
# Separate inference environment for Qwen3.5 / Gemma4 Unified; old runs keep .venv-vllm.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
python_bin="${DECISION_RAG_PYTHON:-python3.12}"
"$python_bin" -m venv .venv-vllm-modern
source .venv-vllm-modern/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu129
# The ordinary 0.24.0 wheel targets CUDA 13. Use the released CUDA 12.9 build
# with its published digest, matching the PyTorch wheels above.
python -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=597949743f2a00c0539d9e2ff0f67b32c608a378e973f99e7529fd6fa9445f70' \
  -e '.[modern-inference,data,dev]'
python -m pip check
python -m decision_rag.vllm_compat
python - <<'PY'
import os
from importlib.metadata import version
import torch
import transformers
import vllm

assert version('vllm') == '0.24.0+cu129', 'This modern profile requires the vLLM 0.24.0 CUDA 12.9 wheel'
assert transformers.__version__ == '5.10.4', 'This modern profile requires Transformers 5.10.4'
assert torch.version.cuda == '12.9', 'Install the CUDA 12.9 PyTorch wheels for this vLLM profile'
assert torch.cuda.is_available(), 'CUDA GPU is required; check the driver for this CUDA 12.9 environment'
probe = torch.ones((16, 16), dtype=torch.bfloat16, device='cuda:0')
assert torch.isfinite(probe @ probe).all().item(), 'BF16 CUDA preflight failed'
torch.cuda.synchronize()
print({'vllm': vllm.__version__, 'transformers': transformers.__version__,
       'torch': torch.__version__, 'cuda': torch.version.cuda,
       'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
PY
python -m pytest -q tests/test_vllm_backend.py tests/test_vllm_runner.py tests/test_cli_concurrency.py
python -m pip freeze > environment-vllm-modern-h100.txt
printf '%s\n' 'Modern inference environment ready. Download model snapshots before the 20-query smoke stage.'
