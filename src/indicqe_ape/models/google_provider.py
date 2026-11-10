"""Google Gemini provider (gemini-1.5-pro, gemini-2.0-flash, ...)."""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from . import register
from .base import GenerationConfig, LLMProvider


@register
class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self, model: str = "gemini-1.5-pro", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("`pip install google-generativeai` to use GoogleProvider.") from e
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set in the environment.")
        genai.configure(api_key=api_key)
        self._genai = genai

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        gen_cfg = self._genai.types.GenerationConfig(
            max_output_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop_sequences=list(cfg.stop) if cfg.stop else None,
        )
        model = self._genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(user, generation_config=gen_cfg)
        return (getattr(resp, "text", None) or "").strip()
