"""Abstract LLM provider interface.

Every provider exposes one method: `complete(system, user, ...)` returning the
generated text. Caching, retries and metadata logging happen one layer up
(in `tasks/`), not here, so providers stay thin and testable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] | None = None


class LLMProvider(ABC):
    """Subclass and set `name` (str) for registration."""

    name: str = ""

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        """Return generated text. Raise on hard failure (caller will retry)."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} model={self.model!r}>"
