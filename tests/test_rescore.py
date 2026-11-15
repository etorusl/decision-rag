import json
from dataclasses import asdict

import pytest

from decision_rag.cli import main
from decision_rag.evaluation import file_hash
from decision_rag.rescore import rescore_run
from decision_rag.schema import Corpus, Node, Query, write_jsonl


@pytest.fixture
def saved_run(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    gold = tmp_path / 'gold.jsonl'
    Corpus([
        Node('root', '', 'root'), Node('domain:ssa', 'SSA', 'group', 'root'),
        Node('page', '', 'page', 'domain:ssa'), Node('470', 'Apply now.', 'passage', 'page'),
        Node('467', ' Apply\t now. ', 'passage', 'page'),
    ]).save(corpus)
    write_jsonl(gold, [asdict(Query('q', 'How?', {'470': 1}, metadata={'dataset': 'multidoc2dial'}))])
    run = tmp_path / 'saved-run'
    ranking = run / 'queries.jsonl'
    write_jsonl(ranking, [{'query_id': 'q', 'ids': ['467'], 'metrics': {'recall@1': 0.0}}])
    summary = {'latency': {'queries': 1}, 'quality': {'recall@1': 0.0},
               'config': {'args': {'top_k': 20, 'corpus': 'old/corpus.jsonl', 'queries': 'old/queries.jsonl'}},
               'inputs': {'old/corpus.jsonl': file_hash(corpus), 'old/queries.jsonl': file_hash(gold)}}
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return run, corpus, gold


def test_rescore_cli_adds_sidecar_using_saved_ids_and_preserves_old_run(saved_run, capsys):
    run, corpus, gold = saved_run
    original = {path: path.read_bytes() for path in run.iterdir()}
    main(['rescore-text', str(run), '--corpus', str(corpus), '--gold', str(gold)])
    result, = json.loads(capsys.readouterr().out)
    improved = json.loads((run / 'summary.text-recall.json').read_text(encoding='utf-8'))
    assert improved['quality']['recall@1'] == 0.0
    assert result['text_recall'] == improved['text_recall']
    text = improved['text_recall']
    assert text['quality']['recall@1'] == 1.0
    assert text['id_quality']['recall@1'] == 0.0
    assert text['ranking_limit'] == 20
    assert text['max_saved_ids'] == 1
    assert text['limited_to_saved_ranking'] is True
    assert text['domains']['SSA']['queries'] == 1
    assert {path: path.read_bytes() for path in original} == original


def test_rescore_checks_gold_hash_before_writing(saved_run):
    run, corpus, gold = saved_run
    gold.write_text(gold.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='recorded queries hash'):
        rescore_run(run, corpus, gold)
    assert not (run / 'summary.text-recall.json').exists()
