"""Backend contract: return probabilities in question/option order."""
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class DecisionOutput:
    probabilities: list[float]
    input_tokens: int = 0
    forward_passes: int | None = 1
    usage: dict = field(default_factory=dict)


class DecisionBackend(Protocol):
    name: str

    def count_tokens(self, text: str) -> int: ...
    def binary(self, context: str, questions: Sequence[str]) -> DecisionOutput: ...
    def choice(self, context: str, question: str, options: Sequence[str]) -> DecisionOutput: ...


def sum_known(values):
    """Unknown engine counters must remain unknown, never become invented zeroes."""
    values = list(values)
    return None if any(value is None for value in values) else sum(values)
