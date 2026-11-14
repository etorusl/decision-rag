import pytest

from decision_rag.backends.base import DecisionOutput
from decision_rag.retrieval import DecisionRetriever, RetrievalConfig
from decision_rag.schema import Corpus, Node, Query


class Scripted:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.contexts = []
    def count_tokens(self, text):
        return len(text.split())
    def binary(self, context, questions):
        self.contexts.append(context)
        return DecisionOutput(next(self.outputs))
    def choice(self, context, question, options):
        self.contexts.append(context)
        return DecisionOutput(next(self.outputs))


def corpus():
    return Corpus([Node('r','','root'), Node('a','Alpha','page','r',order=0),
                   Node('b','Beta','page','r',order=1),
                   Node('a1','first','passage','a',order=0), Node('a2','second','passage','a',order=1),
                   Node('b1','third','passage','b',order=0)])


def test_pruning_and_choice_shortlist_only():
    b = Scripted([[.9,.1], [.8,.7], [.2,.8]])
    retriever = DecisionRetriever(corpus(), b, RetrievalConfig(budgets=[1], top_k=2, choice_k=2))
    out = retriever.retrieve(Query('q','question'))
    assert out.ids == ['a2','a1']
    assert 'third' not in b.contexts[1]
    assert '[Paragraph 1] first' in b.contexts[2]
    assert '0.8' not in b.contexts[2]


def test_choice_preserves_tail_and_promotes_only_winner():
    b = Scripted([[.9,.8],[.9,.8,.7],[.1,.2,.7]])
    retriever = DecisionRetriever(corpus(), b, RetrievalConfig(budgets=[2],top_k=3,choice_k=3,choice_mode='top1'))
    assert retriever.retrieve(Query('q','q')).ids == ['b1','a1','a2']


def test_nonuniform_tree_does_not_drop_early_leaves():
    c = Corpus([Node('r','','root'),Node('p','p','passage','r'),Node('g','g','page','r'),
                Node('q','q','passage','g')])
    retriever = DecisionRetriever(c,Scripted([[.8],[.7,.9]]),RetrievalConfig())
    assert set(retriever.retrieve(Query('x','x')).ids) == {'p','q'}


def test_invalid_probability_rejected():
    with pytest.raises(ValueError, match='invalid probabilities'):
        DecisionRetriever(corpus(),Scripted([[float('nan'),.5]]),RetrievalConfig()).retrieve(Query('q','q'))


def test_missing_parent_and_cycles_rejected():
    with pytest.raises(ValueError, match='Missing parent'):
        Corpus([Node('r','','root'),Node('a','','passage','missing')])
    with pytest.raises(ValueError):
        Corpus([Node('r','','root'),Node('x','','page','y'),Node('y','','page','x')])


def test_oversize_never_silently_truncated():
    with pytest.raises(ValueError, match='exceeds'):
        DecisionRetriever(corpus(),Scripted([]),RetrievalConfig(max_request_tokens=2)).retrieve(Query('q','q'))


def test_md2d_own_memory_only_and_reset():
    c=Corpus([Node('r','','root'),Node('c1','center1','group','r',order=0),Node('c2','center2','group','r',order=1),
              Node('a','a','page','c1'),Node('b','b','page','c2'),Node('p','p','passage','a'),Node('q','q','passage','b')])
    b=Scripted([[.9,.1],[.8],[.9], [.1,.9],[.2,.8],[.3,.8]])
    retriever=DecisionRetriever(c,b,RetrievalConfig.for_dataset('multidoc2dial',page_budget=1))
    first=retriever.retrieve(Query('1','q',{'q':1},conversation_id='conv',turn_id=1))
    assert first.ids == ['p']  # gold doesn't affect state
    assert retriever.memory['conv'] == ['a']
    retriever.retrieve(Query('2','q',conversation_id='conv',turn_id=2))
    assert set(retriever.memory['conv']) == {'a','b'}
    assert 'center1' in b.contexts[3]
    retriever.reset()
    assert not retriever.memory


def test_nq_requires_clustered_corpus():
    with pytest.raises(ValueError, match="topic groups"):
        DecisionRetriever(corpus(), Scripted([]), RetrievalConfig.for_dataset("nq"))
