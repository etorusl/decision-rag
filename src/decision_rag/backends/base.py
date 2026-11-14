"""Backend contract: return probabilities in question/option order."""
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class DecisionOutput:
    probabilities: list[float]
    input_tokens: int = 0
    forward_passes: int = 1


class DecisionBackend(Protocol):
    name: str

    def count_tokens(self, text: str) -> int: ...
    def binary(self, context: str, questions: Sequence[str]) -> DecisionOutput: ...
    def choice(self, context: str, question: str, options: Sequence[str]) -> DecisionOutput: ...
