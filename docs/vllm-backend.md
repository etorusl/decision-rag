# Local vLLM decision inference

`VLLMDecisionBackend` scores an existing causal language model served locally by
vLLM. It needs no training and never loads model weights into the benchmarking
client. The client loads only the corresponding tokenizer. The server contract is
verified against **vLLM 0.11.2**; setup and benchmark commands are in the main runbook.

Every candidate receives the same evidence context and its own relevance question.
The backend sends independent tokenized prompts to `/v1/completions`, allowing vLLM
to batch them and reuse their common prefix. By default it completes the first real
judgment before submitting the remaining candidates, so the prefix can already be
cached. This first judgment is included in all latency and work measurements.
This is a serving-based naive model; it does not execute the custom decision head.

Each request computes probabilities for one answer token, restricted to `no`/`yes`,
or to at most 26 single-token choice letters. The sampled text is ignored. The
client requests both/all label logprobs and normalizes their probabilities. Missing
labels, incompatible tokenization, invalid probability mass and token-budget
violations fail the run instead of silently changing the scoring method.

Start the causal server with:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --logprobs-mode processed_logprobs --max-logprobs 26 \
  --generation-config vllm
```

The server must return processed logprobs after the allowed-token mask. Client
sampling parameters use temperature 1 and disable top-k/top-p pruning and penalties.
Thus scores preserve the label logits' relative probabilities; there is no
multi-token answer generation or reasoning trace. One completion token per judgment
is still produced and counted.

Default cache policy `per-query` assigns a new `cache_salt` before each query,
including warmup queries. Requests/stages of that query share the salt. It permits
within-query prefix reuse while preventing warmup or repeated queries from reusing
an earlier query's KV prefix. `warm` omits the salt and explicitly measures persistent
server caching. Use a dedicated server for controlled latency measurements.

The stage trace reports HTTP requests, logical prompt sequences, submitted prompt
tokens, completion tokens and cached tokens when supplied by the server. Logical
input tokens count the shared context in each sequence; cached tokens measure the
server-reported reuse. Physical GPU forward count is **unknown**, because it depends
on vLLM scheduling/chunked prefill and is not exposed by this API. Client-side CUDA
memory measurements cannot describe a separate server process.

`/v1/models`, `/version` and `/tokenize` verify the selected model and tokenization.
When available, `/server_info?config_format=json` contributes a whitelisted subset of
configuration. This endpoint is optional; development mode is not required or
enabled. No complete configuration containing possible credentials is saved.

Primary references: [completion request schema](https://docs.vllm.ai/en/v0.11.2/api/vllm/entrypoints/openai/protocol/),
[logprob modes](https://docs.vllm.ai/en/v0.11.2/api/vllm/config/model/),
and [automatic prefix caching](https://docs.vllm.ai/en/v0.11.2/features/automatic_prefix_caching/).
