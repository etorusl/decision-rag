"""Algorithm 1 and dataset-specific navigation policies from Appendix B."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .backends.base import DecisionBackend
from .prompts import build_prompt, label
from .schema import Corpus, Query


@dataclass
class RetrievalConfig:
    preset: str = "generic"
    budgets: list[int] = field(default_factory=lambda: [10])
    top_k: int = 20
    choice_k: int = 0
    choice_mode: str = "order"
    max_request_tokens: int = 8192
    max_candidates: int = 10000
    concurrency: int = 1
    memory_pages: int = 2
    use_memory: bool = True
    route_threshold: float = 0.8
    page_budget: int = 10
    rescore_pages: int = 50
    cluster_budget: int = 150
    max_page_passages: int = 60
    section_budget: int = 8
    shared_context: bool = True

    def __post_init__(self):
        if self.preset not in {"generic", "multidoc2dial", "conditionalqa", "mtrag", "nq"}:
            raise ValueError(f"Unknown preset {self.preset}")
        if min(self.top_k, self.max_request_tokens, self.max_candidates, self.concurrency,
               self.page_budget, self.cluster_budget, self.rescore_pages,
               self.max_page_passages, self.section_budget) <= 0:
            raise ValueError("Budgets and token limits must be positive")
        if any(k <= 0 for k in self.budgets) or not self.budgets:
            raise ValueError("budgets must contain positive integers")
        if not 0 <= self.choice_k <= self.top_k or self.choice_k > 255:
            raise ValueError("Require 0 <= choice_k <= min(top_k,255)")
        if self.choice_mode not in {"order", "top1"}:
            raise ValueError("choice_mode must be order or top1")
        if not 0 <= self.route_threshold <= 1 or self.memory_pages < 0:
            raise ValueError("Invalid routing threshold or memory size")

    @classmethod
    def for_dataset(cls, name: str, **kwargs) -> RetrievalConfig:
        defaults = {
            "generic": {}, "multidoc2dial": {"page_budget": 10},
            "conditionalqa": {"page_budget": 3},
            "mtrag": {"page_budget": 10, "choice_mode": "top1"},
            "nq": {"page_budget": 30, "max_request_tokens": 20000},
        }
        if name not in defaults:
            raise ValueError(name)
        return cls(**{"preset": name, **defaults[name], **kwargs})


@dataclass
class RetrievalResult:
    query_id: str
    ids: list[str]
    scores: list[float]
    latency_seconds: float
    stages: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionRetriever:
    def __init__(self, corpus: Corpus, backend: DecisionBackend, config: RetrievalConfig,
                 observer: Callable | None = None):
        self.corpus, self.backend, self.config = corpus, backend, config
        if config.preset == "nq":
            roots = corpus.children(corpus.root.id)
            if not roots or any(n.kind != "group" or not corpus.descendants(n.id, "page") for n in roots):
                raise ValueError("NQ requires topic groups above pages. Build the clustered corpus with "
                                 "python -m decision_rag.hierarchy before retrieval.")
        self.memory: dict[str, list[str]] = {}
        self.observer = observer
        # Global reading order is independent of probabilities or query.
        ordered, stack = [], [corpus.root]
        while stack:
            node = stack.pop()
            ordered.append(node)
            stack.extend(reversed(corpus.children(node.id)))
        self._node_order = {n.id: i for i, n in enumerate(ordered)}

    def reset(self):
        self.memory.clear()

    def _reading_order(self, nodes):
        return sorted({n.id: n for n in nodes}.values(), key=lambda n: self._node_order[n.id])

    def _rank(self, nodes, scores, k):
        return sorted(nodes, key=lambda n: (-scores[n.id], self._node_order[n.id]))[:k]

    def _chunks(self, query, nodes, memory, title_only, candidate_limit=None):
        config = self.config
        size_limit = config.max_candidates if config.shared_context else 1
        if candidate_limit is not None:
            size_limit = min(size_limit, candidate_limit)
        chunks, chunk = [], []
        base, _ = build_prompt(query, [], self.corpus, memory, title_only)
        base_size = self.backend.count_tokens(base) + 128
        running = base_size
        # Linear token accounting. Single-candidate framing overestimates repeated
        # page headings; verify each completed request once instead of O(n²) tokenizing.
        def finish(items):
            if not items:
                return
            context, questions = build_prompt(query, items, self.corpus, memory, title_only)
            exact = self.backend.count_tokens(context) + sum(
                self.backend.count_tokens(q) + 24 for q in questions) + 128
            if exact > config.max_request_tokens:
                if len(items) == 1:
                    raise ValueError(f"Candidate {items[0].id} exceeds max_request_tokens="
                                     f"{config.max_request_tokens}; increase the budget")
                middle = len(items) // 2
                finish(items[:middle])
                finish(items[middle:])
            else:
                chunks.append(items)
        # Product groups remain together when the complete group fits one request.
        units = []
        for node in self._reading_order(nodes):
            if title_only and config.preset == "mtrag" and units and units[-1][0].parent_id == node.parent_id:
                units[-1].append(node)
            else:
                units.append([node])
        for unit in units:
            costs = []
            for node in unit:
                context, questions = build_prompt(query, [node], self.corpus, memory, title_only)
                cost = self.backend.count_tokens(context[len(base):]) + self.backend.count_tokens(questions[0]) + 32
                if base_size + cost > config.max_request_tokens:
                    raise ValueError(f"Candidate {node.id} exceeds max_request_tokens={config.max_request_tokens}; "
                                     "increase the limit or explicitly preprocess long passages")
                costs.append(cost)
            if chunk and (running + sum(costs) > config.max_request_tokens or len(chunk)+len(unit) > size_limit):
                finish(chunk)
                chunk, running = [], base_size
            for node, cost in zip(unit, costs):
                if chunk and (running + cost > config.max_request_tokens or len(chunk) >= size_limit):
                    finish(chunk)
                    chunk, running = [], base_size
                chunk.append(node)
                running += cost
        finish(chunk)
        return chunks

    @staticmethod
    def _validate(output, n, choice=False):
        p = output.probabilities
        if len(p) != n or any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
            raise ValueError("Decision backend returned invalid probabilities")
        if choice and not math.isclose(sum(p), 1, abs_tol=1e-4):
            raise ValueError("Choice probabilities must sum to one")

    def _score(self, query, nodes, stage, trace, memory=(), title_only=False):
        started = time.perf_counter()
        chunks = self._chunks(query, nodes, memory, title_only,
                              125 if self.config.preset == "nq" and stage == "clusters" else None)
        def run(chunk):
            context, questions = build_prompt(query, chunk, self.corpus, memory, title_only)
            if self.observer:
                self.observer(query, stage, chunk, context, questions)
            output = self.backend.binary(context, questions)
            self._validate(output, len(chunk))
            return chunk, output
        if self.config.concurrency > 1 and len(chunks) > 1:
            if not getattr(self.backend, "thread_safe", False):
                raise ValueError("concurrency > 1 requires a thread-safe API backend")
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
                results = list(pool.map(run, chunks))
        else:
            results = [run(chunk) for chunk in chunks]
        scores = {node.id: p for chunk, out in results for node, p in zip(chunk, out.probabilities)}
        trace.append({"stage": stage, "candidate_ids": list(scores), "probabilities": list(scores.values()),
                      "requests": len(chunks), "input_tokens": sum(o.input_tokens for _, o in results),
                      "forward_passes": sum(o.forward_passes for _, o in results),
                      "seconds": time.perf_counter() - started})
        return scores

    def retrieve(self, query: Query) -> RetrievalResult:
        start, trace = time.perf_counter(), []
        cfg, corpus = self.config, self.corpus
        memory = self.memory.get(query.conversation_id, []) if cfg.use_memory and query.conversation_id else []
        page_scores: dict[str, float] = {}
        if cfg.preset == "generic":
            candidates, leaves, depth = corpus.children(corpus.root.id), [], 0
            while candidates:
                leaves.extend(n for n in candidates if n.kind == "passage")
                internal = [n for n in candidates if n.kind != "passage"]
                if not internal:
                    break
                scores = self._score(query, internal, f"level_{depth}", trace)
                kept = self._rank(internal, scores, cfg.budgets[min(depth, len(cfg.budgets)-1)])
                candidates = [child for node in kept for child in corpus.children(node.id)]
                depth += 1
            candidates = leaves
        else:
            pages = corpus.pages
            if cfg.preset == "multidoc2dial":
                centers = corpus.children(corpus.root.id)
                scores = self._score(query, centers, "centers", trace)
                ranked = self._rank(centers, scores, len(centers))
                kept = ranked[:2 if scores[ranked[0].id] < cfg.route_threshold else 1]
                if memory:
                    cur = corpus.nodes[memory[0]]
                    while cur.parent_id != corpus.root.id:
                        cur = corpus.nodes[cur.parent_id]
                    kept.append(cur)
                center_ids = {n.id for n in kept}
                pages = [n for n in pages if n.parent_id in center_ids]
            elif cfg.preset == "nq":
                clusters = corpus.children(corpus.root.id)
                scores = self._score(query, clusters, "clusters", trace)
                kept = self._rank(clusters, scores, cfg.cluster_budget)
                pages = [p for n in kept for p in corpus.descendants(n.id, "page")]
            if cfg.preset == "mtrag":
                scores = self._score(query, pages, "pages_screen", trace, memory, title_only=True)
                pages = self._rank(pages, scores, cfg.rescore_pages)
            page_scores = self._score(query, pages, "pages", trace, memory)
            pages = self._rank(pages, page_scores, cfg.page_budget)
            if cfg.preset in {"multidoc2dial", "mtrag"}:
                pages += [corpus.nodes[x] for x in memory[:cfg.memory_pages] if x not in {p.id for p in pages}]
            candidates = []
            for rank, page in enumerate(pages, 1):
                passages = corpus.descendants(page.id)
                if cfg.preset == "nq":
                    passages = passages if rank <= 3 else passages[:10 if rank <= 10 else 3]
                if cfg.preset == "mtrag" and len(passages) > cfg.max_page_passages:
                    sections = corpus.descendants(page.id, "section")
                    if not sections:
                        raise ValueError(f"Large MTRAG page {page.id} needs section nodes")
                    scores = self._score(query, sections, "sections", trace)
                    kept = self._rank(sections, scores, cfg.section_budget)
                    selected = {p.id for section in kept for p in corpus.descendants(section.id)}
                    passages = [p for p in passages if p.id in selected][:cfg.max_page_passages]
                candidates.extend(passages)
        candidates = self._reading_order(candidates)
        scores = self._score(query, candidates, "passages", trace)
        if cfg.preset == "nq":
            scores = {node.id: scores[node.id] * page_scores[corpus.page_id(node.id)] for node in candidates}
        ranked = self._rank(candidates, scores, cfg.top_k)
        if cfg.choice_k and ranked:
            began = time.perf_counter()
            shortlist = self._reading_order(ranked[:cfg.choice_k])
            context, _ = build_prompt(query, shortlist, corpus)
            question = "Which candidate contains the best evidence for answering the user's question?"
            options = [f"{label(n, query.conversation_id is not None)} {i}" for i, n in enumerate(shortlist, 1)]
            if self.backend.count_tokens(context + question + "\n".join(options)) + 128 > cfg.max_request_tokens:
                raise ValueError("Choice shortlist exceeds token budget; reduce choice_k")
            output = self.backend.choice(context, question, options)
            self._validate(output, len(shortlist), choice=True)
            choices = dict(zip((n.id for n in shortlist), output.probabilities))
            reordered = self._rank(shortlist, choices, len(shortlist))
            if cfg.choice_mode == "top1":
                ranked = [reordered[0]] + [n for n in ranked if n.id != reordered[0].id]
            else:
                ranked = reordered + ranked[len(shortlist):]
            trace.append({"stage": "choice", "candidate_ids": [n.id for n in shortlist],
                          "probabilities": output.probabilities, "requests": 1,
                          "input_tokens": output.input_tokens, "forward_passes": output.forward_passes,
                          "seconds": time.perf_counter() - began})
        if query.conversation_id and ranked and cfg.use_memory:
            page_id = corpus.page_id(ranked[0].id)
            if page_id:
                self.memory[query.conversation_id] = [page_id] + [p for p in memory if p != page_id]
        return RetrievalResult(query.id, [n.id for n in ranked], [scores[n.id] for n in ranked],
                               time.perf_counter() - start, trace)
