"""Tiny synthetic fixture for testing pipeline plumbing, never a quality benchmark."""
import re
from dataclasses import asdict

from .backends.base import DecisionOutput
from .schema import Corpus, Node, Query, write_jsonl


class SmokeBackend:
    name = 'smoke-keyword-fixture-NOT-a-model'
    def count_tokens(self, text):
        return len(text.split())
    def binary(self, context, questions):
        # Explicit synthetic fixture: query about film Memento, lexical cue only.
        blocks = re.findall(r'\[(?:Page|Paragraph|Passage|Topic|Section|Candidate) \d+\] ([\s\S]*?)(?=\n\n\[|$)', context)
        values = [.9 if 'memento' in block.lower() else .1 for block in blocks]
        return DecisionOutput(values[:len(questions)], self.count_tokens(context), 0)
    def choice(self, context, question, options):
        values = [len(options)-i for i in range(len(options))]
        total = sum(values)
        return DecisionOutput([x/total for x in values], self.count_tokens(context), 0)


def write_demo(path):
    from pathlib import Path
    path = Path(path)
    Corpus([
        Node('root','Library','root'),
        Node('film','Memento film','page','root','Memento',0),
        Node('sport','Football','page','root','Football',1),
        Node('p1','Memento follows Leonard Shelby.','passage','film',order=0),
        Node('p2','Memento was directed by Christopher Nolan.','passage','film',order=1),
        Node('p3','Football is played by two teams.','passage','sport',order=0),
    ]).save(path/'corpus.jsonl')
    write_jsonl(path/'queries.jsonl', [asdict(Query('q1','Who is the protagonist of Memento?', {'p1':1}))])
