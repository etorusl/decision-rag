import pytest

from decision_rag.metrics import paired_bootstrap, retrieval_metrics


def test_hit_is_not_recall_and_duplicate_ids_cannot_count_twice():
    m = retrieval_metrics(['a','a','x','b'],{'a':1,'b':1},ks=[1,3])
    assert m['hit@1'] == 1
    assert m['recall@1'] == .5
    assert m['recall@3'] == 1
    assert 0 < m['ndcg@3'] < 1


def test_unlabeled_is_not_zero_quality():
    assert retrieval_metrics(['x'],{}) == {}


def test_bootstrap_is_paired_and_deterministic():
    a=[{'query_id':'a','metrics':{'hit@1':1}}]
    b=[{'query_id':'a','metrics':{'hit@1':0}}]
    assert paired_bootstrap(a,b,'hit@1',20)['ci95'] == [1,1]
    with pytest.raises(ValueError):
        paired_bootstrap(a,[])


def test_seeded_bootstrap_is_independent_of_concurrent_completion_order():
    a = [{'query_id': str(i), 'conversation_id': f'dialog-{i // 2}',
          'metrics': {'recall@10': i / 12}} for i in range(12)]
    b = [{**row, 'metrics': {'recall@10': .5}} for row in a]
    expected = paired_bootstrap(a, b, samples=101, seed=7)
    actual = paired_bootstrap(a[::-1], b[::2] + b[1::2], samples=101, seed=7)
    assert actual == expected
