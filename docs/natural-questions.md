# Natural Questions / DAPR at corpus scale

This is the paper's largest retrieval benchmark: 108,626 Wikipedia pages and 2,682,017 paragraph
units. Use the DAPR version, not a differently segmented Wikipedia or the original NQ long-answer
candidate set. The corpus conversion preserves paragraph IDs so the existing qrels remain valid.
The converter produces an unclustered `root → page → passage` corpus. Build the topic level once,
then reuse it for all retrieval experiments.

## Build on the H100 server

With the repository's `local` and `data` extras installed, first convert a local DAPR release:

```bash
python -m decision_rag.datasets nq-dapr \
  --source-dir data/raw/nq-dapr --output-dir data/nq-dapr --split test
```

The source directory contains `corpus.jsonl` and `test.jsonl` in the DAPR document/chunk schema. Retain its
source manifest and do not filter the corpus to pages present in the evaluation qrels.

Install the optional GPU clustering dependency in the HF environment and build the hierarchy:

```bash
source .venv/bin/activate
python -m pip install -e '.[gpu-clustering]' --extra-index-url https://pypi.nvidia.com
CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/test_hierarchy.py
CUDA_VISIBLE_DEVICES=1 python -m decision_rag.hierarchy \
  data/nq-dapr/corpus.jsonl data/nq-dapr/corpus.clustered.jsonl \
  --clusters 1000 --seed 42 --device cuda --batch-size 64 \
  --model-name intfloat/e5-base-v2 --summary-model Qwen/Qwen3-8B
```

The builder encodes page title plus the first sentence of its first paragraph. It applies the
`query: ` prefix recommended for clustering, attention-mask mean pooling, L2 normalization and
E5's 512-token input limit. Truncated page-feature inputs are counted in the manifest. These
choices follow the [E5 model card](https://huggingface.co/intfloat/e5-base-v2); the DecisionRAG
paper does not specify its tokenizer or sentence splitter.

It then runs GPU k-means with 1,000 clusters, a fixed seed, 10 initializations and at most
300 iterations through the direct [cuML 26.10 KMeans API](https://docs.nvidia.com/cuml/26.10/api/generated/cuml.cluster.KMeans/).
This uses FP32 and `scalable-k-means++` initialization on logical `cuda:0`, which is physical
GPU 1 under the mask above. cuML is an optional Linux/CUDA 12 dependency; a missing dependency,
unavailable CUDA device or GPU allocation error stops the build without CPU fallback.
The test command above runs actual tiny cuML clustering and GPU centroid/title-selection
checks when CUDA and the optional dependency are installed; these tests are skipped on CPU machines.
`--device cpu` explicitly selects the older scikit-learn Lloyd solver for a separate CPU experiment.
The chosen device, implementation, precision and initialization are recorded. The GPU solver can
produce different assignments from old CPU runs even with the same seed; reuse saved assignments
to hold the hierarchy constant. `--n-init` and `--max-iter` permit explicit cost/quality experiments.
All construction costs are recorded separately from retrieval latency.

Finally, Qwen3-8B describes each cluster from at most 300 titles ordered by Euclidean distance to
the cluster centroid. Distances and top-k selection also run on GPU; exact ties use ascending
page IDs. When reusing assignments alone, centroids are recomputed on GPU as well. Text extraction,
title strings and artifact I/O remain on CPU. The embedding model, clustering estimator and
numeric preprocessing tensors are released before loading the summarizer. The summarizer uses the paper's
instruction, local greedy generation and `enable_thinking=False`, supported by the
[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B). No paid API is used.

Descriptions contain at most 30 words. If a generated answer contains multiple sentences, the
first is used; an overlong first sentence is shortened, with the raw answer and shortening flag
saved for review. Titles are dropped from the least typical end only if the 16,384-token summary
input budget is exceeded, and the count is recorded. These are explicit implementation policies,
not claimed author settings. An empty output or unfinished thinking block fails the run.

## Reuse and audit artifacts

Alongside the output corpus, `corpus.clustered.hierarchy/` contains:

- `embeddings.npy` and its identity sidecar, preserving page vectors and source/order hashes.
- `assignments.json`, with page IDs, cluster labels and the assignment hash.
- `summary_titles.json`, the typical-title shortlist used for each generated summary.
- `summaries.json`, bound to both corpus and assignment hashes.
- `metadata.json`, with model revisions, library versions, input/output hashes, cluster sizes,
  generation settings and offline timing for each stage.

To rebuild the exact same tree without loading either model:

```bash
CUDA_VISIBLE_DEVICES=1 python -m decision_rag.hierarchy \
  data/nq-dapr/corpus.jsonl data/nq-dapr/corpus.reused.jsonl --clusters 1000 \
  --assignments-path data/nq-dapr/corpus.clustered.hierarchy/assignments.json \
  --summaries-path data/nq-dapr/corpus.clustered.hierarchy/summaries.json
```

Reuse `--embeddings-path` to experiment with a different k-means seed without re-encoding pages.
For a new fully repeatable build, set `--model-revision` and `--summary-revision` to resolved model
commit hashes from a previous manifest. Fixed random seeds alone do not guarantee bit-identical
GPU or library behavior; reusing saved assignments and summaries avoids this problem.

The builder preserves all page/passage IDs, texts and order fields. It changes only page parents,
inserts topic nodes, and adds hierarchy provenance to the root. An already clustered corpus is
rejected so accidental double clustering cannot silently alter its structure.

## Evaluate honestly

Use the `nq` retrieval preset on `corpus.clustered.jsonl`: top 150 clusters, top 30 pages, the
paper's rank-dependent paragraph budgets, page-probability weighting, and optional top-five
choice reranking. Keep the full corpus even for a small query smoke test.

The paper evaluates 500 sampled queries from DAPR's 3,610 test questions. The exact 500 IDs and
sampling seed are not specified in the supplied paper. Evaluate the full test split, or
save a separately seeded subset of query IDs and label it as your sample. It is not an exact
reproduction of the paper's 500-query scores. Likewise, newly generated cluster summaries and
local zero-shot judges define a distinct experiment.

Report retrieval Hit@1/5/10, Recall@5/10, nDCG@10, stagewise gold-page survival, p50/p95/p99,
throughput and peak GPU memory. Compare against the same full-corpus BM25/dense baseline; include
offline construction cost separately. The paper's remote Jev service can parallelize requests
across serving hardware, so its reported latency is not a one-H100 target.
