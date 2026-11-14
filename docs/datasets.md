# Data preparation and benchmark selection

All adapters produce `corpus.jsonl`, `queries.jsonl`, and a `manifest.json` with split,
query count, full corpus size, and SHA-256 hashes of the source files. Downloads use
public dataset artifacts, not executable dataset scripts. The original licenses
remain applicable. Use separate directories for training and evaluation.

```bash
python -m decision_rag.datasets multidoc2dial --output-dir data/md2d-validation
python -m decision_rag.datasets multidoc2dial --split train --output-dir data/md2d-train
python -m decision_rag.datasets conditionalqa --output-dir data/cqa-dev
python -m decision_rag.datasets conditionalqa --split train --output-dir data/cqa-train
python -m decision_rag.datasets scifact --output-dir data/scifact
python -m decision_rag.datasets mtrag --output-dir data/mtrag-human
```

`--source-dir` reads already downloaded source data. `--limit N` selects the first
N queries in source order **without reducing the corpus**. For conversations this
keeps their order and previous utterances. Complete conversations are preferable
for final memory/latency comparisons; a prefix may end partway through the last
conversation. Never tune budgets or model weights on the evaluation splits.

## Recommended sequence

| Benchmark | What it tests | Default evaluation | Corpus | Compatibility |
|---|---|---|---|---|
| MultiDoc2Dial | Help-center routing, dialogue history and notepad | validation, 4,201 turns | 413 pages / 4,110 passages | Structure segmentation from the original MultiDoc2Dial benchmark |
| ConditionalQA | Scenario understanding, several relevant paragraphs | dev, 285 questions, 271 paragraph-labeled | 652 pages / 58,483 units | Original evidence strings mapped to released element IDs |
| SciFact / BEIR | Transfer to scientific claim evidence | test, 300 claims | 5,183 abstracts | Additional benchmark; not in the DecisionRAG paper |
| MTRAG IBM Cloud | Longer technical dialogues and broad page screening | human, 205 turns, 188 labeled | 8,578 pages / 72,442 passages | Original gold labels; not the paper's expanded relevance pool |
| Natural Questions / DAPR | Large-corpus open-domain retrieval and pruning | full test by default | 108,626 pages / 2,682,017 paragraphs | Requires clustering after import; not the paper's exact random 500-query sample |

The first two are the recommended initial H100 run: their complete corpora are
small enough to debug stage recall and latency without conflating corpus sampling
with retrieval quality. SciFact is a useful held-out domain test. Run MTRAG next,
then NQ after checking CPU RAM/disk capacity and building the hierarchy.

## MultiDoc2Dial

Source: [official archive](https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip)
and [original IBM preprocessing specification](https://github.com/IBM/multidoc2dial/blob/main/scripts/data_preprocessor.py).

The archive is extracted automatically. Pages merge document titles after removing
the terminal `#<part>` suffix. Consecutive spans sharing a section ID **or** heading
are grouped, matching the official DialDoc retrieval segmentation. Passage IDs are
the zero-based sequential IDs of that segmentation, encoded as strings.

Each query is a user turn with the following usable agent response. Gold passages
are found from the response's grounding span using the official containment
convention. The target response is stored as an answer reference; it is never put
in the query history. Earlier conversational responses are observed dialogue
context. No gold labels affect page descriptions or corpus structure. The notepad
must be built by the retriever, not by the adapter. The released test split is a
dummy file, so the adapter refuses it.

Validated locally against the actual archive: **4 centers, 413 pages, 4,110 passages,
4,201 validation queries, and 21,451 training queries**, all grounding IDs resolved.

## ConditionalQA

Source: [official v1.0 release](https://github.com/haitian-sun/ConditionalQA/tree/77bd295952daf415548b3244db10880d3d55cfe0/v1_0).
The downloader pins that commit. IDs are `P<page-index>:e<element-index>`, retaining
the order in `documents.json`. A page description includes its title and `h1` part
headings; each passage carries its page / `h1` / `h2` heading path in metadata.

The paper describes `p`, `li`, and `tr` units, but counting those elements in the
original `documents.json` yields 56,127 units. Including its 2,354 `h3` and two
`h4` elements yields the reported 58,483. This implementation retains all elements
except `h1` and `h2` as an explicit reconstruction choice supported by those
counts, not a confirmed specification from the DecisionRAG authors. Gold `h1`/`h2`
evidence is excluded from paragraph qrels.
Duplicate evidence strings within the same page map to all matching element IDs;
normalization is used only as an exact-text fallback. Unresolved evidence raises
an error instead of silently lowering recall.

Validated locally on actual data: **652 pages, 58,483 units, 285 dev questions,
271 with paragraph evidence**. These are original evidence labels, not DAPR's
alternative graded labels. The manifest records excluded heading evidence.

## SciFact and other BEIR exports

Source: [BEIR dataset collection](https://github.com/beir-cellar/beir#beers-available-datasets).
SciFact downloads its official BEIR archive. Other datasets can be imported:

```bash
python -m decision_rag.datasets beir --source-dir /path/to/beir-dataset \
  --split test --output-dir data/my-beir-dataset
```

The input directory must contain `corpus.jsonl`, `queries.jsonl`, and
`qrels/test.tsv` (or another selected split). Document `_id` values become passage
IDs, and graded relevance is preserved. Each original document becomes one page
with one passage; abstracts are not artificially split. Only queries present in
the selected qrels are evaluated. Any qrel referring to a missing passage fails
validation. For a fair transfer result use the complete corpus and report nDCG@10,
Recall@10/100, and MRR, alongside latency.

## MTRAG IBM Cloud

Source: [official IBM release](https://github.com/IBM/mt-rag-benchmark/tree/e0970de1b1207c2a9506adf238e6e1a22037e38a).
The downloader pins that commit and downloads the original passage/document ZIPs,
human generation tasks, and Cloud relevance judgments.

Only IBM Cloud tasks are selected. Source passage IDs and text remain unchanged;
URL path segments define products. Short lines surrounded by blank lines provide
section candidates for large-page pruning. This independent layout heuristic is
an implementation choice inferred from the paper’s layout description. Queries retain
conversation IDs and turn indices and are sorted chronologically per conversation.
Unlabeled turns remain in the query stream so they can update conversational memory.

The paper's Table 12 uses additional GPT-5 judgments pooled from 18 systems. Those
labels are not in the original benchmark: this adapter uses **official gold only**.
Label resulting reports accordingly; their metrics cannot be directly compared
with Table 12. The paper additionally normalizes identical passage text, so compare
an ID-based report and a whitespace/lowercase text-deduplicated report separately.

Validated locally on the actual pinned release: **172 products, 8,578 pages,
72,442 passages, 205 human turns, and 188 labeled turns**; every gold ID resolves.

## Natural Questions / DAPR

Source: [DAPR release and schemas](https://github.com/UKPLab/acl2024-dapr).
Download the original prebuilt dump, then import:

```bash
mkdir -p data/raw/nq-dapr
wget -c https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/NaturalQuestions/corpus.jsonl \
  -O data/raw/nq-dapr/corpus.jsonl
wget -c https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/NaturalQuestions/test.jsonl \
  -O data/raw/nq-dapr/test.jsonl
python -m decision_rag.datasets nq-dapr --source-dir data/raw/nq-dapr \
  --split test --output-dir data/nq-dapr
```

The adapter accepts DAPR's native `id/title/chunks` corpus and
`query/judged_chunks` query files. It preserves chunk IDs and grades. Import is
page-to-paragraph only: build the 1,000-cluster hierarchy before applying the
paper's cluster-screening configuration. That offline stage needs e5-base-v2 title
and first-sentence embeddings, k-means, and Qwen3-8B cluster descriptions.

The paper uses a fixed random 500-query subset of the 3,610 test questions, but the
exact subset is not assumed here. Default import retains the full split;
`--limit 500` is explicitly a **prefix**, not a reproduction of that subset.
Record a selected query-ID list and seed for any sampled experiment. This adapter
materializes the corpus as Python objects, so large-corpus runs require substantial
CPU memory in addition to the H100's GPU memory.

## Reproducibility and validation

The adapters validate the corpus tree, unique query IDs, and every gold passage
reference. Unit tests use synthetic source-shaped fixtures and require no network:

```bash
pytest tests/test_datasets.py -q
```

Manifests record hashes rather than trusting mutable download URLs. Preserve them
with results and record the repository commit. Raw data and generated corpora live
under ignored `data/`; only adapter code and small synthetic tests are committed.
