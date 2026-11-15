import pytest

from decision_rag.evaluation import evaluate
from decision_rag.retrieval import RetrievalResult
from decision_rag.schema import Corpus, Node, Query, read_jsonl


class SavedRankingRetriever:
    top_k = 20

    def __init__(self, corpus):
        self.corpus = corpus

    def reset(self):
        pass

    def retrieve(self, query):
        ids = ['467', '466', 'foil-1', 'foil-2', 'foil-3', 'second'] if query.id == 'ssa' else ['case']
        return RetrievalResult(query.id, ids, [1.0] * len(ids), 0.0, [])


@pytest.mark.parametrize('execution_mode', ['latency', 'throughput'])
def test_both_recall_series_use_same_saved_ids_and_group_only_exact_text(tmp_path, execution_mode):
    corpus = Corpus([
        Node('root', 'Corpus', 'root'),
        Node('domain:ssa', 'SSA', 'group', 'root'),
        Node('domain:dmv', 'DMV', 'group', 'root'),
        Node('domain:va', 'VA', 'group', 'root'),
        Node('page:ssa', 'SSA page', 'page', 'domain:ssa'),
        Node('page:dmv', 'DMV page', 'page', 'domain:dmv'),
        Node('page:va', 'VA page', 'page', 'domain:va'),
        Node('470', 'Same evidence', 'passage', 'page:ssa'),
        Node('468', 'Same  evidence', 'passage', 'page:ssa'),
        Node('467', ' Same\n evidence  ', 'passage', 'page:ssa'),
        Node('466', 'Same\tevidence', 'passage', 'page:dmv'),
        Node('second', 'Different evidence', 'passage', 'page:ssa'),
        Node('foil-1', 'Irrelevant one', 'passage', 'page:ssa'),
        Node('foil-2', 'Irrelevant two', 'passage', 'page:ssa'),
        Node('foil-3', 'Irrelevant three', 'passage', 'page:ssa'),
        Node('dmv-gold', 'Evidence', 'passage', 'page:dmv'),
        Node('case', 'evidence', 'passage', 'page:dmv'),
        Node('va-passage', 'A VA passage', 'passage', 'page:va'),
    ])
    metadata = {'dataset': 'multidoc2dial'}
    queries = [
        Query('ssa', 'First labeled query', {'470': 1, '468': 1, 'second': 1}, metadata=metadata),
        Query('dmv', 'Second labeled query', {'dmv-gold': 1}, metadata=metadata),
        Query('unlabeled', 'No evidence', metadata={**metadata, 'gold_page_ids': ['page:va']}),
    ]
    retriever = SavedRankingRetriever(corpus)
    concurrent = execution_mode == 'throughput'
    summary = evaluate(retriever, queries, tmp_path, execution_mode=execution_mode,
                       query_concurrency=2 if concurrent else 1, worker_factory=lambda: retriever,
                       dedup_text=True, progress='off')
    text_recall = summary['text_recall']
    assert summary['quality']['recall@1'] == text_recall['id_quality']['recall@1'] == 0.0
    assert summary['quality']['recall@5'] == 0.0
    assert summary['quality']['recall@10'] == pytest.approx(1 / 6)
    assert text_recall['quality']['recall@1'] == 0.25
    assert text_recall['quality']['recall@5'] == 0.5
    assert text_recall['ranking_limit'] == 20
    assert text_recall['max_saved_ids'] == 6
    assert text_recall['limited_to_saved_ranking']
    domains = text_recall['domains']
    assert list(domains) == ['DMV', 'SSA', 'VA', 'Student Aid']
    assert domains['SSA']['queries'] == domains['SSA']['labeled_queries'] == 1
    assert domains['SSA']['quality']['recall@1'] == 0.5
    assert domains['SSA']['quality']['recall@5'] == 1.0
    assert domains['DMV']['quality']['recall@1'] == 0.0
    assert domains['VA'] == {'queries': 1, 'labeled_queries': 0, 'id_quality': {}, 'quality': {}}
    assert domains['Student Aid']['queries'] == 0
    ranked, _, unlabeled = read_jsonl(tmp_path / 'queries.jsonl')
    assert ranked['ids'] == ['467', '466', 'foil-1', 'foil-2', 'foil-3', 'second']
    assert ranked['metrics']['recall@1'] == ranked['metrics']['recall@5'] == 0.0
    assert ranked['text_recall'] == domains['SSA']['quality']
    assert not unlabeled['text_recall']
