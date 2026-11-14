import math
from types import SimpleNamespace

import numpy as np
import pytest

from decision_rag.baselines import BM25Retriever, DenseRetriever
from decision_rag.schema import Corpus, Node, Query


def small_corpus():
    return Corpus([Node('root','','root'), Node('page','','page','root'),
                   Node('a','apple apple','passage','page',order=0),
                   Node('b','banana','passage','page',order=1)])


def test_bm25_precomputed_postings_match_formula():
    result = BM25Retriever(small_corpus()).retrieve(Query('q','apple'))
    expected = math.log(1 + 1.5/1.5) * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 2/1.5))
    assert result.ids == ['a','b']
    assert result.scores == pytest.approx([expected,0])


def test_dense_precision_cache_and_query_instruction(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    st = pytest.importorskip('sentence_transformers')
    calls = []
    class Encoder:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
        def __getitem__(self, index):
            return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash='revision')))
        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return np.array([[1.,0.] if 'apple' in text else [0.,1.] for text in texts])
    monkeypatch.setattr(st, 'SentenceTransformer', Encoder)
    dense = DenseRetriever(small_corpus(), device='cuda', index_dir=tmp_path)
    assert calls[0]['model_kwargs']['torch_dtype'] == torch.bfloat16
    assert dense.retrieve(Query('q','apple')).ids[0] == 'a'
    assert calls[-1]['prompt'].startswith('Instruct:')
    with pytest.raises(ValueError, match='mismatch'):
        DenseRetriever(small_corpus(),device='cuda',dtype='float32',index_dir=tmp_path)
