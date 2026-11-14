#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
dataset="${1:-multidoc2dial}"
limit="${2:-100}"
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
decision-rag prepare "$dataset" --output "data/$dataset"
common=(--corpus "data/$dataset/corpus.jsonl" --queries "data/$dataset/queries.jsonl" --preset "$dataset" --backend hf --model "${NAIVE_MODEL:-Qwen/Qwen3-8B}" --limit "$limit" --warmup 2)
for pages in 3 5 10; do
  decision-rag benchmark "${common[@]}" --pages "$pages" --output "results/$run_tag/ablations/$dataset/pages-$pages"
done
for size in 1 16 64; do
  decision-rag benchmark "${common[@]}" --max-candidates "$size" --output "results/$run_tag/ablations/$dataset/context-$size"
done
decision-rag benchmark "${common[@]}" --no-memory --output "results/$run_tag/ablations/$dataset/no-memory"
python scripts/summarize_runs.py "results/$run_tag/ablations/$dataset" --output "results/$run_tag/ablations/$dataset/comparison"
