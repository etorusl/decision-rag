"""Portable corpus/query format; IDs are always stable strings."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Node:
    id: str
    text: str
    kind: str  # root, group, page, section, passage
    parent_id: str | None = None
    title: str = ""
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant_ids: dict[str, float] = field(default_factory=dict)
    conversation_id: str | None = None
    turn_id: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def context(self, last_turns: int = 6) -> str:
        # History excludes the current user utterance, which is appended exactly once.
        turns = [*self.history, {"role": "user", "content": self.text}][-last_turns:]
        if self.history:
            return "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        return self.text


class Corpus:
    def __init__(self, nodes: Iterable[Node]):
        rows = list(nodes)
        self.nodes = {n.id: n for n in rows}
        if len(rows) != len(self.nodes):
            raise ValueError("Duplicate corpus IDs")
        roots = [n for n in rows if n.parent_id is None]
        if len(roots) != 1 or roots[0].kind != "root":
            raise ValueError("Corpus must have exactly one root node")
        self.root = roots[0]
        self._children: dict[str, list[Node]] = {n.id: [] for n in rows}
        for node in rows:
            if node is self.root:
                continue
            if node.parent_id not in self.nodes:
                raise ValueError(f"Missing parent for {node.id}: {node.parent_id}")
            self._children[node.parent_id].append(node)
        for children in self._children.values():
            children.sort(key=lambda n: (n.order, n.id))
        visited: set[str] = set()
        pending = [self.root]
        while pending:
            node = pending.pop()
            if node.id in visited:
                raise ValueError("Corpus cycle")
            visited.add(node.id)
            children = self._children[node.id]
            if node.kind == "passage" and children:
                raise ValueError(f"Passage cannot have children: {node.id}")
            if node.kind != "passage" and not children:
                raise ValueError(f"Non-passage leaf: {node.id}")
            pending.extend(children)
        if len(visited) != len(rows):
            raise ValueError("Disconnected corpus or cycle")
        self.passages = [n for n in rows if n.kind == "passage"]
        self.pages = [n for n in rows if n.kind == "page"]
        self._page: dict[str, str] = {}
        for passage in self.passages:
            cur = passage
            while cur.parent_id is not None and cur.kind != "page":
                cur = self.nodes[cur.parent_id]
            if cur.kind == "page":
                self._page[passage.id] = cur.id

    def children(self, node_id: str) -> list[Node]:
        return list(self._children[node_id])

    def descendants(self, node_id: str, kind: str = "passage") -> list[Node]:
        result, pending = [], list(reversed(self._children[node_id]))
        while pending:
            node = pending.pop()
            if node.kind == kind:
                result.append(node)
            pending.extend(reversed(self._children[node.id]))
        return result

    def page_id(self, passage_id: str) -> str | None:
        return self._page.get(passage_id)

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        return cls(Node(**row) for row in read_jsonl(path))

    def save(self, path: str | Path) -> None:
        write_jsonl(path, (asdict(n) for n in self.nodes.values()))


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: str | Path) -> list[Query]:
    return [Query(**row) for row in read_jsonl(path)]
