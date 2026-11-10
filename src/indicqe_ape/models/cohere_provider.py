"""Cohere Command / Aya provider.

Supported models (pass via `--model` or the `model=` kwarg):

- ``command-r-plus`` (default) - Cohere's flagship Command R+ model.
- ``command-r``                - Smaller, faster Command R.
- ``c4ai-aya-expanse-8b``      - Aya Expanse 8B, multilingual (CC-BY-NC).
- ``c4ai-aya-expanse-32b``     - Aya Expanse 32B, multilingual (CC-BY-NC).

For QE / APE on Indic language pairs, the **Aya Expanse** family is the
recommended choice: it is explicitly multilingual (covers many Indic
languages) whereas the Command R series is English-centric. Note that the
Aya weights on Cohere are released under CC-BY-NC, so non-commercial use
only.

Authentication: reads ``COHERE_API_KEY`` from the environment.

API: uses the Cohere Python SDK v5+ ``ClientV2.chat`` endpoint, which
takes OpenAI-style ``messages=[{"role": "system"|"user", "content": ...}]``
and returns ``response.message.content[0].text``. If the installed
``cohere`` SDK version differs, you may need to adjust the response-
parsing line below.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from . import register
from .base import GenerationConfig, LLMProvider


@register
class CohereProvider(LLMProvider):
    name = "cohere"

    def __init__(self, model: str = "command-r-plus", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("`pip install cohere>=5.13` to use CohereProvider.") from e
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is not set in the environment.")
        # ClientV2 is the v2 chat client (cohere SDK 5.x). If you are pinned
        # to an older SDK, fall back to `cohere.Client(...)` and the v1
        # `chat()` shape (preamble= + message=).
        self._client = cohere.ClientV2(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            p=cfg.top_p,
            stop_sequences=list(cfg.stop) if cfg.stop else None,
        )
        # Cohere v2 chat returns response.message.content as a list of
        # content blocks; the first block's `.text` is the assistant reply.
        # If your installed `cohere` SDK exposes a different shape (e.g. a
        # top-level `.text` attribute), adjust the line below accordingly.
        try:
            return resp.message.content[0].text or ""
        except (AttributeError, IndexError, TypeError):  # pragma: no cover
            # Defensive fallback for SDK shape drift.
            return getattr(resp, "text", "") or ""
