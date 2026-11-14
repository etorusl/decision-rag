# Local vLLM embedding and reranking

These inference adapters call local vLLM HTTP servers. They load no model weights into
the benchmark process and do not import the training backend. The supported reference
configuration is **vLLM 0.11.2** with official Qwen3 Embedding/Reranker and BGE-M3 weights.

## Servers on one H100

Run the servers in the separate vLLM environment created by the repository's inference
setup. Pin each model with its immutable Hugging Face commit using `--revision` and
`--tokenizer-revision`; the benchmark must receive the same revision strings.
The suite's `scripts/download_models.py` downloads all five models before inference and
writes `cache/model-manifest.json`; pass it to the runner with `--model-manifest` to use
the verified cached revisions offline. The full seven-run sequence is in [the runbook](run-benchmarks.md).

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Embedding-8B \
  --host 127.0.0.1 --port 8001 --runner pooling \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.40 \
  --no-enable-prefix-caching \
  --revision "$EMBEDDING_REVISION" --tokenizer-revision "$EMBEDDING_REVISION"
```

In another process, on the same GPU:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Reranker-8B \
  --host 127.0.0.1 --port 8002 --runner pooling \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.40 \
  --no-enable-prefix-caching \
  --revision "$RERANKER_REVISION" --tokenizer-revision "$RERANKER_REVISION" \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
```

The two 0.40 budgets are deliberate. Do not run two servers each reserving the default
0.90 of the same GPU. Stop both before starting the separate generative-model decision
server. Actual model memory and latency still require measurement on the target H100.
The dense index also lives on this GPU in the benchmark client. Its FP32 matrix needs
`passages × embedding_dimension × 4` bytes, plus CUDA context and search workspace.
The remaining nominal 20% is shared by that index and other allocations; it is not a
guaranteed usable index capacity. If the index does not fit, reduce `--pool-memory` after
checking model requirements or use a smaller embedding model. CUDA errors
stop the run rather than moving search to CPU.

The official Qwen3 reranker is converted by vLLM at load time into an equivalent two-token
classification head. This is inference-only conversion of existing weights, not training.
The client supplies the original query/instruction prefix and document/assistant suffix,
then calls `/score` with `use_activation=true`. Supplying unformatted query/document pairs
would change the model's prompt and is not done here.

The additional BGE pair uses native vLLM `XLMRobertaModel` and
`XLMRobertaForSequenceClassification` implementations. Both use CLS pooling; BGE-M3 produces
normalized dense embeddings without a query instruction. The BGE reranker receives the
plain query/document pair through `/score`, with sigmoid activation. It does not use the
Qwen chat prompt or Qwen architecture overrides. The suite disables chunked prefill and
prefix caching for the BGE servers and keeps their context limit at 8192 tokens.
Only the dense output of BGE-M3 is evaluated, not its sparse or ColBERT outputs.
Qwen and BGE server pairs run sequentially on GPU 1 and use separate model/revision index
caches and separate logs. Qwen configurations remain unchanged.

## Measurement and index identity

`VLLMDenseRetriever` calls `/v1/embeddings`, L2-normalizes float embeddings, and performs
exact dot-product search and top-k selection on GPU by default (`--search-device cuda`).
The normalized FP32 index is uploaded once during initialization and remains in GPU memory;
each query uploads its vector and returns only the top-k IDs/scores to the CPU. Search uses
FP32 with TF32 disabled; the embedding model still runs in BF16. With the launcher mask
`CUDA_VISIBLE_DEVICES=1`, local `cuda:0` is physical GPU 1, shared with the vLLM servers.
A live query embedding is requested on **every** retrieval.
Corpus indexing and GPU upload are outside per-query latency; indexing usage is recorded
in `runtime_metadata.indexing_usage`. `runtime_metadata.search_runtime` records the search
device, precision, index size and initialization details.
The persistent index stores corpus content hash, model name, immutable model revision,
vLLM version, declared dtype/quantization, normalization, and query-prefix policy. A mismatched identity fails instead
of reusing stale vectors. Without an immutable revision, persistent caching is refused.
Revision metadata records an explicit server-launch argument; `/v1/models` itself does
not prove the resolved Hugging Face revision. Launch the server and client consistently.

`VLLMDenseRerankRetriever` measures the complete sequence: live query embedding, exact
search, and live reranking of the returned dense candidates. It does not load saved
per-query candidates or report reranking-only time as complete retrieval latency.
The dense stage reports `query_embedding_seconds` and `vector_search_seconds` separately;
search timing includes query upload and completed top-k download. The reranker stage has
its own timing. `--search-device cpu` is an explicit diagnostic experiment. Older results
using the CPU search implementation must be rerun for the default GPU comparison.

Each stage records completed HTTP wall time, HTTP request count, number of prompts,
and server-reported prompt/cached tokens when available. **GPU forward-pass counts are
unknown through these APIs**, represented as `null`; an HTTP batch is not called a GPU
forward pass. GPU memory belongs to both the vLLM server processes and the client's resident
index, so client-side PyTorch counters cannot measure the whole pipeline. Use the repository's
device monitor for the combined sampled footprint.

Both model IDs are checked against `/v1/models`; server versions come from `/version`.
Response indices must cover the input batch exactly. Invalid, nonfinite or zero vectors,
wrong model names and invalid relevance probabilities fail visibly.

The reference launcher disables prefix caching on both pooling servers. vLLM 0.11.2
can otherwise reuse exact warmup query prefixes during measured embedding/reranking calls;
these pooling APIs do not expose request cache salts. The HTTP client cannot infer this
launch setting, so the launcher configuration is part of experiment provenance.

Inputs are never silently truncated. If a query/document pair or embedding input exceeds
the server's configured context length, vLLM returns an error and the benchmark stops.
For Qwen, increase `--max-model-len` after measuring memory or define and report an explicit
preprocessing policy. Server HTTP failures do not trigger a hidden Transformers fallback.
BGE-M3 and its reranker are limited to 8192 tokens; a longer Qwen limit does not increase
the separate BGE limit.

## Verified primary references

- [vLLM 0.11.2 pooling APIs](https://docs.vllm.ai/en/v0.11.2/models/pooling_models/).
- [vLLM 0.11.2 OpenAI-compatible embeddings and score API](https://docs.vllm.ai/en/v0.11.2/serving/openai_compatible_server/).
- [Official original-Qwen3-reranker conversion and pair prompt](https://github.com/vllm-project/vllm/blob/v0.11.2/examples/offline_inference/pooling/qwen3_reranker.py).
- [Score protocol and response schemas](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/entrypoints/openai/protocol.py).
- [Score prompt concatenation](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/entrypoints/score_utils.py).
- [Official BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
- [Official BGE reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- [Pinned vLLM RoBERTa/XLM-R implementations and CLS pooling](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/models/roberta.py).
