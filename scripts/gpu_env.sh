#!/usr/bin/env bash
# Shared GPU selection for H100 setup and training scripts. Source this file.
decision_rag_gpu_index="${DECISION_RAG_GPU_INDEX:-1}"
if [[ ! "$decision_rag_gpu_index" =~ ^[0-9]+$ ]]; then
  printf '%s\n' 'DECISION_RAG_GPU_INDEX must be one nonnegative GPU index.' >&2
  return 1
fi
export CUDA_VISIBLE_DEVICES="$decision_rag_gpu_index"
