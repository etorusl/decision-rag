#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
python_bin="${DECISION_RAG_PYTHON:-python3.12}"
if ! command -v "$python_bin" >/dev/null; then
  printf '%s\n' 'Python 3.12 is recommended. Set DECISION_RAG_PYTHON to a Python 3.11–3.13 executable.' >&2
  exit 1
fi
"$python_bin" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-h100.txt -e '.[local,data,dev]'
python -m pip check
python scripts/h100_preflight.py
python -m pytest -q
python -m pip freeze > environment-h100.txt
printf '%s\n' 'Setup complete. Activate with: source .venv/bin/activate'
