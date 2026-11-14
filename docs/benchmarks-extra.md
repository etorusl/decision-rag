# Additional quality tests and performance stress

Start with **SciFact**, a small transfer test for scientific claim evidence. Then
use **FiQA-2018** for finance-related questions and **NFCorpus** for biomedical
retrieval from the [official BEIR collection](https://github.com/beir-cellar/beir#beers-available-datasets).
These test domain transfer beyond the four datasets in the DecisionRAG paper.
Preserve the original corpus and test qrels; do not select a corpus using gold labels.

```bash
python -m decision_rag.datasets scifact --output-dir data/scifact
# For any other downloaded/extracted BEIR export:
python -m decision_rag.datasets beir --source-dir /path/to/fiqa --split test --output-dir data/fiqa
python -m decision_rag.datasets beir --source-dir /path/to/nfcorpus --split test --output-dir data/nfcorpus
```

Use the same queries and original document IDs for BM25, dense retrieval,
dense-plus-reranker and DecisionRAG. Compare nDCG@10, Recall@10 and MRR@10 with
complete retrieval p50/p95 latency and peak GPU memory. These transfer results
should be reported separately from task-trained in-domain results.

## Synthetic size and passage-length sweeps

The generator creates a deterministic hierarchy of topic handbooks and uniform
passages with unique record/value pairs. It creates the whole corpus before
sampling query targets. Values do not appear in group/page descriptions. Its
recall is a plumbing check; **it is not evidence of natural-language quality**.

```bash
python -m decision_rag.stress --output data/stress-1000 \
  --pages 1000 --passages-per-page 10 --words-per-passage 100 \
  --queries 100 --seed 42 --branching 100

# GPU BM25 and GPU traversal, timed with the same query stream:
CUDA_VISIBLE_DEVICES=1 decision-rag benchmark --method bm25 --corpus data/stress-1000/corpus.jsonl \
  --queries data/stress-1000/queries.jsonl --output results/stress-1000-bm25

CUDA_VISIBLE_DEVICES=1 decision-rag benchmark --method decision --backend hf --model Qwen/Qwen3-8B \
  --preset generic --pages 10 --max-request-tokens 8192 \
  --corpus data/stress-1000/corpus.jsonl --queries data/stress-1000/queries.jsonl \
  --output results/stress-1000-hf
```

Sweep one variable at a time: `--pages 100/1000/10000`,
`--words-per-passage 100/500/1000`, `--passages-per-page 10/50`, and pruning budget
`--pages` on the benchmark command. Keep the seed, backend, precision, request token
limit, choice setting, and hardware fixed. Record actual model forward passes and
input tokens alongside latency; a single H100 does not provide unlimited parallel
serving capacity. Very large generated corpora also consume CPU RAM.

`--branching` bounds the fan-out of grouping nodes. More grouping levels are added
when necessary. Each run writes hashes and generation parameters to its manifest;
existing generator outputs are protected against accidental overwrite.

## CSV and Markdown comparison

```bash
python scripts/summarize_runs.py results --output results/comparison

# Optional paired bootstrap, resampling whole conversations when present:
python scripts/summarize_runs.py results/md2d-bm25 results/md2d-decision \
  --output results/md2d-comparison \
  --paired results/md2d-decision results/md2d-bm25 --metric recall@10
```

The report writes `.csv` and `.md`, with quality, latency, memory, tokens per query,
forward passes, and input hashes. Reporting itself needs only the Python standard
library. The optional paired comparison uses the installed package's bootstrap
implementation and writes `.paired.json`; it rejects mismatched query/corpus
hashes or duplicate IDs. The CI is A minus B, in percentage points.

The tool reports runs independently. Compare matching input hashes, relevance
labels, hardware, memory mode, and timing scope; the table does not average results
across different datasets. Missing metrics or GPU measurements remain missing.
