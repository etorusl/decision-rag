# Paper fidelity and reproduction audit

Reference: Wan et al., *Retriever-Free Retrieval: Navigating Documents with Decision Models*,
[arXiv:2610.35868v1](https://arxiv.org/abs/2610.35868v1), especially Algorithm 1 and Appendices A, B, F, G.
This is an independent implementation. A working retrieval pipeline is not evidence that its
scores or speed reproduce the paper; those require the matching data, judge and measured runs.

## What the method requires

1. Build a corpus hierarchy offline. At query time, score all children of the retained frontier;
   no lexical or embedding retriever supplies the initial candidates.
2. For each stage, put the query and numbered candidate descriptions in one shared context.
   Ask a separate binary question about each candidate. Normalize each answer over yes/no,
   not over candidates. Split contexts only when the request budget requires it.
3. Keep the highest-scoring branches, expand their children, and discard the rest. Earlier
   scores are not put into subsequent prompts. Pruning is irreversible.
4. Score candidate passages jointly. Optionally rerank a shortlist with a single multiple-choice
   question, whose probabilities sum to one. Preserve the remaining binary order.

Candidate contexts preserve document/reading order, including within the final choice shortlist;
the model must not receive a shortlist ordered by the scores it is meant to refine. Conversational
input contains the latest six dialogue messages, including the current user turn.

## Dataset-specific protocol

| Dataset | Paper corpus and evaluation | Navigation |
|---|---|---|
| MultiDoc2Dial | 4 centers, 413 merged pages, 4,110 official passages; validation: 661 dialogues / 4,201 turns | Route to best center, second if best P(yes) < 0.8, and center of newest notepad page; top 10 pages; add passages from newest two notepad pages; choice top 20 |
| ConditionalQA | 652 guides, 58,483 paragraph units; dev: 285 questions, 271 with evidence | All pages → top 3 → all their paragraphs under heading paths; choice top 20 |
| MTRAG IBM Cloud | 172 products, 8,578 pages, 72,442 passages; 26 conversations / 205 turns, 188 labeled | All page titles in product-preserving request batches → pooled top 50 jointly rescored with product/title/headings → top 10 pages; pages with >60 passages use top 8 headings then first 60 eligible passages in reading order; choice top 20 promotes only the winner |
| NQ (DAPR) | 1,000 clusters, 108,626 pages, 2,682,017 paragraphs; fixed sample of 500 of 3,610 test questions | Eight cluster requests → top 150 clusters; page-title requests ≤20k tokens → top 30 pages; all paragraphs of pages 1–3, first 10 of pages 4–10, first 3 of pages 11–30; multiply paragraph probability by its page probability; choice top 5 |

The notepad is an ordered list of distinct pages from the system's own previous retrievals.
After a turn, move the final rank-one passage's page to its front. Never seed or update it from
gold evidence. Process each conversation chronologically and reset state between conversations.
Past gold/reference replies can be supplied as the benchmark's observed dialogue history;
the current reply and current evidence must not appear in the query.

NQ hierarchy construction uses `intfloat/e5-base-v2` embeddings of page title plus first sentence,
k-means with 1,000 clusters, then Qwen3-8B descriptions. The summary prompt requests at most
30 words covering both typical and less typical subjects, from up to 300 titles ordered by
cluster typicality. The published random sample IDs, seed and exact summaries are needed for
an exact NQ comparison; newly generated ones define a separate reproducible run.

## The judge distinction

**The paper's main zero-shot model is hosted Jev 1.13.** The paper does not claim that an arbitrary
untuned causal LM with yes/no token logits matches Jev. The optional `JevBackend` implements the
documented System One schema and supports TypeSafe directly or OpenRouter. It pins versioned
model IDs and validates response probabilities. No hosted calls are made by the offline tests.

**A local zero-shot LM backend is an alternative judge.** Shared candidate context and isolated
question attention reproduce the information flow, while the yes/no token readout and model
weights differ. Those results must be labeled with the actual model and backend. A standard
independent query–passage reranker does not implement the shared-candidate judgment in Eq. 2.

**Appendix G is a trained model, not just a prompt.** It uses Qwen3-Base 1.7B/4B/8B with the LM
head replaced by a trainable decision head. The head projects the question-end and option-end
hidden states, takes cosine similarity scaled by a learned inverse temperature capped at 100,
and softmaxes over options. Question branches start immediately after the context and cannot
attend to one another. The general and MD2D-only checkpoints additionally link the two option-end
tokens; the reported LoRA checkpoint does not.

The general model is trained on a mixture of public classification and multiple-choice data,
then adapted to retrieval. The paper's 60.7 R@1 model is LoRA rank 64, alpha 128 on attention and
MLP projections, with a fully trained head, using 30% of MD2D training dialogues plus CQA training
examples. It ranks by binary scores without the Jev choice step. A freshly initialized head or
a newly trained checkpoint cannot be labeled the authors' checkpoint. The publication does not
fully specify the general mixture, sample seeds, prompt serialization or head dimensions.

## Author code availability and limits

On 2026-11-14, the public [kilian-group/DecisionRAG repository](https://github.com/kilian-group/DecisionRAG)
contains loaders and walkers for MD2D, CQA and MTRAG; an OpenRouter Jev adapter; SQLite response
caching; metrics; offline tests; and [`scripts/paper_runs.sh`](https://github.com/kilian-group/DecisionRAG/blob/main/scripts/paper_runs.sh).
Its README was updated on November 13. Its recursive source tree does not contain NQ preparation
or retrieval, local decision models, Appendix G training code/checkpoints, dense/reranker baseline
runners, or the full downstream QA and expanded MTRAG relevance-labeling pipeline.

The released scripts include variants and should not automatically be treated as an exact
reproduction of every published row. In particular, their MC examples use 3 MD2D pages or 5 CQA
pages, while the main paper configurations retain 10 and 3, respectively. The released `choose()`
helper reuses the full binary context, whereas Appendix B.6 specifies shortlist-only context.
The README also describes filling ranks below judged candidates from BM25; Algorithm 1 does not
include that fallback. Preserve and report such differences when comparing implementations.

No license file was present in the inspected author tree. This project uses the paper and public
interfaces as references; the source is implemented independently rather than vendoring author code.

## API contract and token counting

- TypeSafe: [`POST https://api.typesafe.ai/v1/systemone`](https://docs.typesafe.ai/api/system-one).
  The body contains `model`, shared `state`, and named `questions`. Binary questions use
  `type: noul` and `instructions`; read `answers[id].noul`. Choice questions additionally use
  a `criteria` map; read `answers[id].probabilities` in the original option order.
- Direct TypeSafe model: `jev-1.13.0`. OpenRouter model: `typesafe/jev-1.13`; the author's
  [configuration](https://github.com/kilian-group/DecisionRAG/blob/main/configs/judges.toml) pins
  response snapshot `typesafe/jev-1.13-20260917`. A mismatch fails explicitly.
- The [current TypeSafe limits](https://docs.typesafe.ai/models) distinguish 64k tokens per request
  from 32k per question, counting shared state plus that question. Long context requires an
  explicit TypeSafe header and raises both limits to 262k. Choice supports at most 255 options.
- Jev's tokenizer is not public. The adapter's packing estimate is UTF-8 bytes, a conservative
  bound rather than a measured token count. This can split stages earlier than the original
  implementation and therefore affect both quality and timing. Actual usage is recorded from
  responses. Prompt formatting/question overhead must also fit the request limit.

## What results can be compared

Use official evidence IDs and the same full corpus. Do not silently evaluate only the gold page,
sample documents to make retrieval easier, or tune budgets on the evaluation set. MD2D recall is
passage-ID recall; CQA/NQ can have multiple relevant paragraphs, so Hit@k and Recall@k differ.
Report skipped unlabeled questions and denominator sizes.

MTRAG's main paper scores use gold plus GPT-5-expanded relevance judgments, with identical-text
passages deduplicated. Official-qrels-only evaluation is useful but does not reproduce that table.
Confidence intervals in the paper are paired bootstrap intervals with 10,000 resamples of whole
dialogues/conversations (or independent questions), not independently resampled conversational turns.

The reported Jev retrieval latency (roughly 0.7–1.8 seconds) is a remote service measurement with
parallel serving capacity. Its local reranker timing uses an RTX A6000. These are not a controlled
one-H100 speed comparison. Measure warm and cold behavior separately, disable response caches,
synchronize CUDA, record p50/p95/p99, throughput, memory and every scored token/candidate/request.
Report offline corpus preparation/indexing separately. A single H100 may serialize batches that
the remote service processes concurrently, especially for NQ's roughly 17,500 candidate page titles.

Downstream answer evaluation is a separate experiment: the paper uses Qwen3.5-27B with FP8 weights,
thinking disabled and greedy decoding, given top five passages. It reports answer EM on CQA/NQ
and judged correctness on MTRAG. Retrieval scores alone do not establish end-to-end answer quality.
