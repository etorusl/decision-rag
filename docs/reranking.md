# Dense retrieval plus Qwen3 reranking

`DenseRerankRetriever` first runs the configured embedding model and exact corpus-vector search, then independently scores the dense top 100 passages with a Qwen3-Reranker. Both the 0.6B and 8B variants are supported. The 8B embedding + 8B reranker configuration matches the model sizes in the paper's strong dense baseline; the 0.6B pair is useful for a faster first run.

The implementation uses the official [Qwen3-Reranker model-card prompt](https://huggingface.co/Qwen/Qwen3-Reranker-8B#using-transformers): instruction, query, document, and an assistant prefix that disables thinking. It reads the logits of `no` and `yes` and normalizes over those two values. It never generates tokens and only projects the last hidden state into the two relevant output vocabulary rows. This gives the same two-label softmax as the full LM head without constructing vocabulary logits at every input position.

Every measured query includes fresh query embedding, dense search, prompt construction, tokenization, all reranking batches, and GPU synchronization. Corpus indexing/model loading are offline and excluded. Dense candidates are not loaded from saved query answers. Per-query traces report dense and reranking durations, candidate count, actual input tokens, padded tokens, forward passes, and truncation counts. This allows a controlled comparison with DecisionRAG on the same H100, corpus, query set, and warm-up policy.

Defaults are 8,192 tokens per pair, at most 32,768 padded tokens per forward pass, and at most 16 candidates per batch. Candidates are batched by length and returned in their original order before sorting by probability; tied scores retain the dense ranking.

The default `document_tail` policy removes only the end of an overlong document body while preserving the complete query/instruction and the fixed assistant suffix. The number of affected candidates and removed tokens is recorded. If the query/instruction alone consumes the budget, scoring fails explicitly. Choose `truncation="error"` for a run where every candidate must be evaluated in full, or increase `max_length` up to the model's native context. Record this policy when comparing quality.

Python interface:

```python
from decision_rag.reranking import DenseRerankRetriever

retriever = DenseRerankRetriever(
    corpus,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    reranker_model="Qwen/Qwen3-Reranker-8B",
    candidates=100,
    top_k=20,
    device="cuda",
    dtype="bfloat16",
    batch_size=16,
    max_length=8192,
    max_batch_tokens=32768,
    index_dir="data/index-qwen3-8b",
)
result = retriever.retrieve(query)
```

Use the repository's benchmark CLI/runbook to record quality and end-to-end latency together. The document text passed to this reranker is the corpus passage text; descriptions and hierarchy metadata are not silently added. No published score or speedup is asserted before running on the target data and GPU.
