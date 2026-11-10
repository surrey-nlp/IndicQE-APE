"""Open-weight HuggingFace provider (Llama, Qwen, IndicTrans2, sarvam, ...).

Loads the model lazily and runs single-sample generation. For multi-thousand-
example evaluations on multi-GPU hardware, batch via vLLM or TGI in a separate
script — this keeps the canonical code path simple.

Device handling
---------------

- ``device="auto"`` (default): pick the best available device:
    * CUDA → ``device_map="auto"`` (multi-GPU friendly via ``accelerate``)
    * Apple Silicon (MPS) → load on ``"mps"`` explicitly
    * CPU → load on ``"cpu"``
- ``device="cuda"`` / ``"mps"`` / ``"cpu"``: force the choice.

4-bit quantisation (``load_in_4bit=True``) requires CUDA + ``bitsandbytes``;
on MPS or CPU it raises a clear error rather than silently falling back.

dtype safety
------------

- bfloat16 on CUDA Ampere+ and on MPS macOS 14+ (most Apple Silicon).
- Older MPS / CPU prefer float16 or float32. If you see NaN losses or
  "MPS does not support bfloat16" errors, pass ``dtype="float16"``.
"""
from __future__ import annotations

import os
import platform
from typing import Any

from . import register
from .base import GenerationConfig, LLMProvider


def _pick_device(device: str) -> str:
    """Resolve device='auto' to the best concrete choice on this host."""
    if device != "auto":
        return device
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_dtype(requested: str, device: str) -> str:
    """Coerce a torch dtype string to one this device actually supports."""
    if device == "cpu":
        # CPU path is mostly fp32 in practice; bf16/fp16 work but are rarely fast.
        return "float32"
    if device == "mps" and requested == "bfloat16":
        # Older MPS builds (macOS 13 / older PyTorch) lack bf16. fp16 is the safe default.
        return "float16"
    if device == "cuda" and requested == "bfloat16":
        # bf16 needs Ampere (sm_80) or newer. On Turing (sm_75: RTX 2080 Ti, RTX 5000, Quadro
        # RTX 8000) torch accepts bf16 but emulates it, which is slow and errors in some kernels.
        # Some clusters still have large 2080ti and rtx5000 partitions, and they are often the only
        # idle capacity, so silently degrading there is worse than switching to fp16.
        try:
            import torch
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major < 8:
                    return "float16"
        except Exception:  # noqa: BLE001 - never let a probe break model loading
            pass
    return requested


@register
class HFProvider(LLMProvider):
    name = "hf"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        # Resolve aliases (e.g. "qwen2.5-7b") via the curated registry.
        try:
            from .indic_models import resolve as _resolve_alias
            self.model = _resolve_alias(self.model)
        except Exception:  # pragma: no cover
            pass
        self.device = _pick_device(device)
        self.dtype = _safe_dtype(dtype, self.device)
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

        if self.load_in_4bit and self.device != "cuda":
            raise RuntimeError(
                "load_in_4bit=True requires CUDA + bitsandbytes; "
                f"current device is {self.device!r}. Drop the 4-bit flag, "
                "or run on a CUDA host."
            )

        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "`pip install transformers torch accelerate` to use HFProvider."
            ) from e

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, token=token, trust_remote_code=self.trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "token": token,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.dtype, torch.float16),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = getattr(torch, self.dtype, torch.float32)
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
            elif self.device == "mps":
                # accelerate's auto map does not handle MPS well; load to CPU then move.
                load_kwargs["device_map"] = {"": "mps"}
            else:
                load_kwargs["device_map"] = {"": "cpu"}

        self._model = AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Sanity log — only print on first load, not subsequent .complete() calls.
        sysinfo = f"{platform.machine()}/{platform.system()}"
        print(f"[hf_provider] loaded {self.model} on {self.device} dtype={self.dtype} "
              f"4bit={self.load_in_4bit} host={sysinfo}", flush=True)

    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        self._ensure_loaded()
        import torch  # type: ignore[import-not-found]

        cfg = config or GenerationConfig()
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        # Some chat templates lack a system role — fall back to prepending to user.
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # pragma: no cover
            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{system}\n\n{user}"}],
                tokenize=False, add_generation_prompt=True,
            )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        do_sample = cfg.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
