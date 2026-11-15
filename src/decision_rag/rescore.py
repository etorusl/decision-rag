"""Add text Recall to a saved MultiDoc2Dial evaluation using its original artifacts."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from .evaluation import file_hash
from .metrics import retrieval_metrics
from .schema import Corpus, Query, load_queries, read_jsonl
from .text_recall import TextRecall, is_multidoc2dial


def score_saved_rankings(corpus: Corpus, gold_queries: Iterable[Query], rankings: Iterable[dict], *,
                         ranking_limit: int | None = None, expected_count: int | None = None) -> dict:
    gold = {}
    for query in gold_queries:
        if query.id in gold:
            raise ValueError(f'Duplicate gold query ID: {query.id}')
        gold[query.id] = query
    if not is_multidoc2dial(list(gold.values())):
        raise ValueError('Text Recall rescoring requires prepared MultiDoc2Dial gold queries')

    scorer = TextRecall(corpus, ranking_limit=ranking_limit)
    seen = set()
    for row in rankings:
        query_id = row.get('query_id')
        if not isinstance(query_id, str) or query_id not in gold:
            raise ValueError(f'Saved result has no gold query: {query_id!r}')
        if query_id in seen:
            raise ValueError(f'Duplicate saved query ID: {query_id}')
        ranking = row.get('ids')
        if not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking):
            raise ValueError(f'Saved query {query_id} must have an ordered list of passage IDs')
        query = gold[query_id]
        scorer.add(query, ranking, retrieval_metrics(ranking, query.relevant_ids))
        seen.add(query_id)
    if not seen:
        raise ValueError('No saved rankings to rescore')
    if expected_count is not None and len(seen) != expected_count:
        raise ValueError(f'Run has {len(seen)} saved queries, expected {expected_count} from summary.json')
    return scorer.summary()


def _check_input(summary, role, recorded_argument, file):
    inputs = summary.get('inputs', {})
    if not inputs:
        return
    expected = inputs.get(str(recorded_argument))
    if expected is None:
        names = {f'{role}.jsonl', file.name}
        if recorded_argument:
            names.add(Path(recorded_argument).name)
        candidates = {digest for path, digest in inputs.items() if Path(path).name in names}
        if len(candidates) != 1:
            raise ValueError(f'Cannot identify the recorded {role} input hash')
        expected = candidates.pop()
    if file_hash(file) != expected:
        raise ValueError(f'{file} does not match the recorded {role} hash in summary.json')


def rescore_run(run_dir: str | Path, corpus_file: str | Path, gold_file: str | Path) -> dict:
    run_dir, corpus_file, gold_file = Path(run_dir), Path(corpus_file), Path(gold_file)
    source = run_dir / 'summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    count = summary.get('latency', {}).get('queries')
    if type(count) is not int or count < 1 or not isinstance(summary.get('quality'), dict):
        raise ValueError(f'Expected a completed evaluation summary in {source}')
    config = summary.get('config', {})
    args = config.get('args') or config
    _check_input(summary, 'corpus', args.get('corpus'), corpus_file)
    _check_input(summary, 'queries', args.get('queries'), gold_file)
    top_k = config.get('retrieval', {}).get('top_k', args.get('top_k'))
    ranking_limit = top_k if type(top_k) is int and top_k > 0 else None
    text_recall = score_saved_rankings(
        Corpus.load(corpus_file), load_queries(gold_file), read_jsonl(run_dir / 'queries.jsonl'),
        ranking_limit=ranking_limit, expected_count=count,
    )

    output = run_dir / 'summary.text-recall.json'
    temporary = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=run_dir, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({**summary, 'text_recall': text_recall}, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'summary': str(output), 'text_recall': text_recall}
