#!/usr/bin/env bash
# Same corpus/query subset/hardware for all five systems. No API keys.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
dataset="${1:-multidoc2dial}"
limit="${2:-100}"
case "$dataset" in
  multidoc2dial|conditionalqa|mtrag) preset="$dataset" ;;
  scifact) preset=generic ;;
  *) printf '%s\n' 'Use multidoc2dial, conditionalqa, mtrag, or scifact.' >&2; exit 1 ;;
esac
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="results/$run_tag/$dataset"
mkdir -p "$run_dir"
decision-rag prepare "$dataset" --output "data/$dataset"
common=(--corpus "data/$dataset/corpus.jsonl" --queries "data/$dataset/queries.jsonl" --top-k 20 --warmup 2)
if [[ "$limit" != all ]]; then common+=(--limit "$limit"); fi
naive_model="${NAIVE_MODEL:-Qwen/Qwen3-8B}"
embedding_model="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
reranker_model="${RERANKER_MODEL:-Qwen/Qwen3-Reranker-8B}"
index_dir="cache/dense/$dataset/${embedding_model//\//_}"
decision-rag benchmark "${common[@]}" --method bm25 --output "$run_dir/bm25"
decision-rag benchmark "${common[@]}" --method dense --model "$embedding_model" --index-dir "$index_dir" --output "$run_dir/dense"
decision-rag benchmark "${common[@]}" --method dense-rerank --model "$embedding_model" --reranker-model "$reranker_model" --index-dir "$index_dir" --output "$run_dir/dense-rerank"
decision-rag benchmark "${common[@]}" --preset "$preset" --backend hf --model "$naive_model" --choice-k 0 --output "$run_dir/naive-binary"
decision-rag benchmark "${common[@]}" --preset "$preset" --backend hf --model "$naive_model" --choice-k 20 --output "$run_dir/naive-choice"
python scripts/summarize_runs.py "$run_dir" --output "$run_dir/comparison"
printf 'Results: %s\n' "$run_dir"
