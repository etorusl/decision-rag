#!/usr/bin/env bash
# Local supervised recipe, not the paper's unavailable general checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONUNBUFFERED=1
source .venv/bin/activate
dataset="${1:-multidoc2dial}"
mode="${2:-smoke}"
case "$dataset" in multidoc2dial|conditionalqa) ;; *) exit 1 ;; esac
case "$mode" in smoke|full) ;; *) printf '%s\n' 'Second argument: smoke or full'; exit 1 ;; esac
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
base_model="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
train_data="data/train/$dataset"
records="data/training/$dataset-$run_tag"
checkpoint="checkpoints/$dataset-$mode-$run_tag"
decision-rag prepare "$dataset" --split train --output "$train_data"
record_args=(--corpus "$train_data/corpus.jsonl" --queries "$train_data/queries.jsonl" --dataset "$dataset" --tokenizer "$base_model" --output "$records" --max-tokens 30000 --max-candidates 10000)
train_args=(--train "$records/train.jsonl" --dev "$records/monitor.jsonl" --output "$checkpoint" --model "$base_model" --mode lora --epochs 1 --max-context-tokens 32768 --max-packed-tokens 40960)
if [[ "$mode" == smoke ]]; then
  record_args+=(--limit 500)
  train_args+=(--max-steps 10 --head-only-steps 2 --save-every 0)
else
  train_args+=(--head-only-steps 200)
fi
decision-rag training-data "${record_args[@]}"
python -m decision_rag.training "${train_args[@]}"
decision-rag prepare "$dataset" --output "data/$dataset"
eval_args=(--corpus "data/$dataset/corpus.jsonl" --queries "data/$dataset/queries.jsonl" --preset "$dataset" --backend head --checkpoint "$checkpoint" --choice-k 0 --warmup 2 --gpu-monitor-index "$CUDA_VISIBLE_DEVICES" --output "results/$run_tag/$dataset/trained-$mode")
if [[ "$mode" == smoke ]]; then eval_args+=(--limit 20); fi
decision-rag benchmark "${eval_args[@]}"
printf 'Checkpoint: %s\n' "$checkpoint"
