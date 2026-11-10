"""LLM provider registry.

Add a new provider by subclassing `LLMProvider` and registering it here.
"""
from __future__ import annotations

from .base import LLMProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {}


def register(cls: type[LLMProvider]) -> type[LLMProvider]:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Provider {cls!r} must set a `name` class attribute.")
    _PROVIDERS[name] = cls
    return cls


def get_provider(name: str, **kwargs) -> LLMProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    return sorted(_PROVIDERS)


# Try to import each bundled provider. Missing optional deps are silently
# skipped — that provider just won't be in the registry. This means a fresh
# `pip install indicqe-ape` works without the full extras stack, and a CLI
# user only sees providers they can actually run.
def _try_import(modname: str) -> None:
    try:
        __import__(f"{__name__}.{modname}", fromlist=[modname])
    except ImportError:
        pass


for _m in (
    "openai_provider",
    "anthropic_provider",
    "google_provider",
    "cohere_provider",
    "hf_provider",
    "vllm_provider",
):
    _try_import(_m)
del _try_import, _m
