"""Exact-text Recall and domain breakdown for saved MultiDoc2Dial rankings."""
from __future__ import annotations

from .metrics import aggregate, retrieval_metrics

KS = (1, 5, 10, 20)
RECALL_KEYS = tuple(f'recall@{k}' for k in KS)
DOMAINS = {'dmv': 'DMV', 'ssa': 'SSA', 'va': 'VA', 'studentaid': 'Student Aid'}


def is_multidoc2dial(queries):
    return bool(queries) and all(query.metadata.get('dataset') == 'multidoc2dial' for query in queries)


def text_groups(corpus, *, lowercase=False):
    """Map passage IDs to the first passage ID with the same normalized text."""
    representatives = {}
    groups = {}
    for passage in corpus.passages:
        key = ' '.join(passage.text.split())
        if lowercase:
            key = key.lower()
        groups[passage.id] = representatives.setdefault(key, passage.id)
    return groups


def group_relevance(relevant, groups):
    """Keep the strongest grade when multiple gold IDs share one group."""
    grouped = {}
    for passage_id, grade in relevant.items():
        key = groups[passage_id]
        grouped[key] = max(grade, grouped.get(key, grade))
    return grouped


def _corpus_domain(corpus, node_id):
    node = corpus.nodes.get(node_id)
    while node is not None:
        if node.kind == 'group' and node.id.startswith('domain:'):
            return DOMAINS.get(node.id.removeprefix('domain:'))
        node = corpus.nodes.get(node.parent_id)
    return None


def _query_domain(query, corpus):
    pages = query.metadata.get('gold_page_ids') or ()
    domains = {_corpus_domain(corpus, node_id) for node_id in pages}
    domains.discard(None)
    if not domains:
        domains = {_corpus_domain(corpus, passage_id) for passage_id, grade in query.relevant_ids.items() if grade > 0}
        domains.discard(None)
    return next(iter(domains)) if len(domains) == 1 else 'Unassigned'


def _quality(rows):
    identified, text = zip(*rows, strict=True) if rows else ((), ())
    return {'queries': len(rows), 'labeled_queries': sum(bool(row) for row in identified),
            'id_quality': aggregate(identified), 'quality': aggregate(text)}


class TextRecall:
    """Accumulate ID and exact-text scores for the same saved result IDs."""

    def __init__(self, corpus, *, ranking_limit=None):
        self.corpus = corpus
        self.ranking_limit = ranking_limit
        self.groups = text_groups(corpus)
        self.rows = {}
        self.max_saved_ids = 0

    def add(self, query, ranking, id_metrics=None):
        if query.id in self.rows:
            raise ValueError(f'Duplicate query ID in text Recall: {query.id}')
        if id_metrics is None:
            id_metrics = retrieval_metrics(ranking, query.relevant_ids)
        try:
            grouped_ranking = list(dict.fromkeys(self.groups[passage_id] for passage_id in ranking))
            grouped_gold = {self.groups[passage_id] for passage_id, grade in query.relevant_ids.items() if grade > 0}
        except KeyError as exc:
            raise ValueError(f'Unknown passage ID for {query.id}: {exc.args[0]}') from exc
        quality = {f'recall@{k}': sum(key in grouped_gold for key in grouped_ranking[:k]) / len(grouped_gold)
                   for k in KS} if grouped_gold else {}
        id_quality = {key: id_metrics[key] for key in RECALL_KEYS} if id_metrics else {}
        self.rows[query.id] = (_query_domain(query, self.corpus), id_quality, quality)
        self.max_saved_ids = max(self.max_saved_ids, len(ranking))
        return quality

    def summary(self):
        domains = {name: [] for name in DOMAINS.values()}
        rows = []
        # Throughput completion order varies. Aggregate in the same order as serial.
        for query_id in sorted(self.rows):
            domain, identified, text = self.rows[query_id]
            rows.append((identified, text))
            domains.setdefault(domain, []).append((identified, text))
        return {**_quality(rows), 'domains': {domain: _quality(values) for domain, values in domains.items()},
                'ranking_limit': self.ranking_limit, 'max_saved_ids': self.max_saved_ids,
                'limited_to_saved_ranking': True}
