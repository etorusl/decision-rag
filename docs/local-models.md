# Local models and training

Two local modes are implemented. Neither needs an API key.

- `HFLogitBackend`: Qwen3-8B in non-thinking mode, scoring the next-token logits of `no` and `yes`. All candidates are visible in one shared context. Binary questions are independent branches in one packed backbone forward pass; no answer tokens are generated. Multiple-choice reranking uses a softmax over up to 26 single-token letters. This is a **zero-shot baseline**, not Jev and not an Appendix G checkpoint.
- `DecisionHeadBackend`: Qwen3-Base with a trained cosine-similarity head, following Appendix G's architecture. Question and answer-option end states have separate learned projections into a common space. A learned inverse temperature is capped at 100; a softmax runs over each question's options. The loader requires a checkpoint with nonzero training steps and never silently uses a random head.

Both modes use the same packed tree attention. Context is causal; each question sees the context and itself; each option sees the context, its question, and itself. Question positions reset to the position immediately after the shared context, and options reset to the end of their own question. Different questions cannot attend to each other. `option_links` optionally allows the option-end tokens of the same question to attend to each other, as in the paper's general and MD2D-only variants. It is off by default, matching the LoRA variant.

CUDA uses PyTorch FlexAttention with a sparse block mask; application code does not explicitly allocate a dense token-by-token attention mask. PyTorch's compiled `create_block_mask` still evaluates the logical token-pair mask before reducing it to blocks, so temporary memory depends on compiler fusion and must be measured at the target context length on H100. CPU uses SDPA with an explicit mask for numerical tests and refuses packed lengths above 4,096 tokens. The first CUDA requests compile kernels; report cold and warmed-up timings separately. The full vocabulary LM projection is never applied to the shared context: the naive backend gathers only decision states and projects only requested output labels. The trained head batches its question and option projections; binary softmax and training loss also run over all questions together.

## Install and model choice

Use Python 3.12 and the repository's H100 setup instructions. The implementation targets PyTorch >=2.7 and Transformers 4.57.x with Qwen3's dictionary mask support. The recommended naive model is [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), and the training backbone is [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base). Smaller Qwen3 dense models can be selected to debug the workflow. Other model families are rejected rather than assumed compatible with custom branch attention.

## Train on prepared requests

Training consumes JSONL with explicit zero-based class labels:

```json
{"context":"Query and numbered candidate descriptions", "questions":["Is Page 1 relevant?", "Is Page 2 relevant?"], "options":[["no","yes"],["no","yes"]], "labels":[1,0]}
```

Metadata is allowed and ignored by the trainer. Split by dialogue for conversational data and by page for ConditionalQA before constructing these requests; evaluation labels must never appear in training. The retrieval-stage preparation command in the main runbook builds requests from the training split. The trainer supports multi-class examples as well as binary retrieval questions, so public general decision tasks can also be supplied in the same schema.

```bash
python -m decision_rag.training \
  --train data/training/train.jsonl \
  --dev data/training/monitor.jsonl \
  --output checkpoints/md2d-lora \
  --model Qwen/Qwen3-8B-Base \
  --mode lora \
  --head-only-steps 200 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --gradient-accumulation 4 \
  --max-context-tokens 32768 \
  --max-packed-tokens 40960
```

LoRA uses rank 64, alpha 128, no dropout, and all attention and MLP projections. The head is trained in FP32; backbone defaults to BF16. AdamW uses betas 0.9/0.95, no weight decay, gradient clipping 1.0, 50 warm-up updates and cosine decay towards 10% of the peak rate. `--head-only-steps 200` initializes the head with a frozen backbone at 1e-3 before LoRA training. Full fine-tuning is available with `--mode full`, but 8B AdamW state plus activations can exceed a single H100's memory; LoRA is the single-H100 default. `--mode head` freezes the backbone throughout and is useful for a cheap diagnostic.

`--max-context-tokens` caps the maximum rotary position (context plus longest question/option), while `--max-packed-tokens` caps total packed tokens including all branches. Whole overlong examples are dropped and counted, never truncated. If most records are dropped, reduce the preparation request size or raise the budgets after measuring VRAM. Start with a small `--max-steps 10` smoke run and a new output directory. Actual peak memory must be measured on the target H100; it has not been measured on the development laptop.

Checkpoints contain a tokenizer, a safetensors head, a LoRA adapter or full backbone where appropriate, `decision_config.json`, and training logs. Periodic checkpoints use `step-N` directories. Optimizer-state resume is not implemented; use final or periodic inference checkpoints, and start a new training directory for another run. The final validation report contains loss and classification accuracy; retrieval quality must be measured separately with the retrieval evaluator.

## Reproduction limits

The paper's main results use hosted Jev 1.13. Its general-trained Qwen decision checkpoint, exact general-training mixture, projection dimension, special-token format, and seeds were not provided with the supplied article. This implementation therefore uses explicit independent choices: 1,024-dimensional projections, existing EOS markers after questions/options, and a documented local training recipe. Starting LoRA from Qwen3-Base plus a freshly initialized head does not reproduce the paper's “general + MD2D + CQA (LoRA)” initialization. The user must measure local results, and paper scores must not be copied into local result tables.

The published trained-head evaluations rank binary decisions only. Disable choice reranking when comparing that setup; the generic head supports choice requests, but binary-only training does not establish their calibration. The naive model can be tested both with and without choice reranking as an ablation.

Implementation references: [Qwen3 model code, Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py), [Transformers FlexAttention integration](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/flex_attention.py), and Appendix G of the supplied article.
