"""Anthropic / Claude provider (claude-opus-4-x, claude-sonnet-4-x, ...)."""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from . import register
from .base import GenerationConfig, LLMProvider


@register
class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("`pip install anthropic` to use AnthropicProvider.") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop_sequences=list(cfg.stop) if cfg.stop else None,
        )
        # Concatenate text blocks (Claude responses can be a list of content blocks)
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
