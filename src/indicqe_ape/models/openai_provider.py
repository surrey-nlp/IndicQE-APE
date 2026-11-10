"""OpenAI / GPT-class provider.

Handles both classic chat models (gpt-4o, gpt-4.1, ...) and the **reasoning** family
(gpt-5.x, o1/o3/o4). Reasoning models reject `temperature`/`top_p`, use
`max_completion_tokens` instead of `max_tokens`, and bill hidden reasoning tokens as
output — so for a scoring task we pin `reasoning_effort=none` (override with
`IQE_REASONING_EFFORT`). The integer-score parser is unchanged downstream.
"""
from __future__ import annotations

import os

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import register
from .base import GenerationConfig, LLMProvider


def _is_reasoning(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4")) or m.startswith("gpt-6")


@register
class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", *, seed: int | None = 0, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("`pip install openai` to use OpenAIProvider.") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self._client = OpenAI(api_key=api_key)
        self._seed = seed
        self._reasoning = _is_reasoning(model)
        self._effort = os.environ.get("IQE_REASONING_EFFORT", "none")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),  # broad; SDK-specific filtering can be added
        reraise=True,
    )
    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self._reasoning:
            # reasoning models: no temperature/top_p; reasoning tokens count as output, so
            # keep effort minimal for scoring and give a small completion budget for the digits.
            kwargs["max_completion_tokens"] = max(cfg.max_tokens, 16)
            kwargs["reasoning_effort"] = self._effort
        else:
            kwargs["max_tokens"] = cfg.max_tokens
            kwargs["temperature"] = cfg.temperature
            kwargs["top_p"] = cfg.top_p
            if cfg.stop:
                kwargs["stop"] = list(cfg.stop)
        if self._seed is not None:
            kwargs["seed"] = self._seed
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
