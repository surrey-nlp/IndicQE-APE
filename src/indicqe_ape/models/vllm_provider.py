"""vLLM provider — for models the transformers `hf` provider can't load cleanly:
multimodal `image-text-to-text` checkpoints (Qwen3.6, gemma-4, command-a-plus) run
text-only here, and quantized formats (AWQ / GPTQ / compressed-tensors / w4a4) are
auto-detected. Much faster than `hf` for thousand-example sweeps.

Config via kwargs or env:
  VLLM_TP=<n>        tensor-parallel size (e.g. 2x 48 GB L40S -> VLLM_TP=2 fits ~31B bf16)
  VLLM_QUANT=<q>     force quantization (e.g. compressed-tensors, awq, gptq); else auto
  VLLM_MAX_LEN=<n>   max_model_len (default 4096)
  VLLM_GMU=<f>       gpu_memory_utilization (default 0.90)

Usage: indicqe-ape qe --provider vllm --model Qwen/Qwen3.6-27B --pair enhi --subset challenge ...
Requires `pip install vllm` in the env (only imported on first use, so the registry
still loads on machines without vLLM).
"""
from __future__ import annotations

import os

from . import register
from .base import GenerationConfig, LLMProvider


@register
class VLLMProvider(LLMProvider):
    name = "vllm"

    def __init__(self, model: str, *, dtype: str = "bfloat16",
                 tensor_parallel_size: int | None = None, quantization: str | None = None,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.90,
                 trust_remote_code: bool = True, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._dtype = dtype
        self._tp = int(tensor_parallel_size or os.environ.get("VLLM_TP", "1"))
        self._quant = quantization or os.environ.get("VLLM_QUANT") or None
        self._max_len = int(os.environ.get("VLLM_MAX_LEN", max_model_len))
        self._gmu = float(os.environ.get("VLLM_GMU", gpu_memory_utilization))
        self._trust = trust_remote_code
        # enforce_eager disables torch.compile/inductor + CUDA-graph capture. Default ON because
        # some cluster compute nodes have no nvcc/CUDA_HOME, so the compile path dies with
        # "Could not find nvcc and default cuda_home='/usr/local/cuda'". Set VLLM_EAGER=0 to re-enable
        # compilation on a node that has the CUDA toolkit.
        self._eager = os.environ.get("VLLM_EAGER", "1") not in ("0", "false", "False", "")
        self._llm = None
        self._tok = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM  # type: ignore[import-not-found]
        self._llm = LLM(model=self.model, dtype=self._dtype,
                        tensor_parallel_size=self._tp, quantization=self._quant,
                        max_model_len=self._max_len, gpu_memory_utilization=self._gmu,
                        trust_remote_code=self._trust, enforce_eager=self._eager)
        self._tok = self._llm.get_tokenizer()

    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        self._ensure_loaded()
        from vllm import SamplingParams  # type: ignore[import-not-found]
        cfg = config or GenerationConfig()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        # IQE_NO_THINK=1 disables chain-of-thought for hybrid reasoning models (e.g. sarvam-m) so they
        # emit the score directly instead of a long trace that blows the token budget / costs 28s/item.
        # Models whose template doesn't take the kwarg (e.g. gemma3 sarvam-translate) fall back cleanly.
        ct_kw = {}
        if os.environ.get("IQE_NO_THINK", "") in ("1", "true", "True"):
            ct_kw["enable_thinking"] = False
        try:
            prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **ct_kw)
        except TypeError:
            prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Fallback must match the code default of 512, NOT 8. An 8-token budget truncates any
        # model that emits a word before the score, which is what destroyed the four-shot parse
        # rates (BharatGPT en-ta 0.22, tiny-aya si-en 0.31). This path is normally unreachable
        # because GenerationConfig always carries max_tokens, but the magic number is the same
        # one that caused the fault, so it does not stay in the tree.
        sp = SamplingParams(temperature=getattr(cfg, "temperature", 0.0),
                            max_tokens=getattr(cfg, "max_tokens", None) or 512)
        out = self._llm.generate([prompt], sp, use_tqdm=False)
        return out[0].outputs[0].text.strip()
