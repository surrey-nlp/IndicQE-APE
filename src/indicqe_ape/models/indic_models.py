"""Curated registry of open-weight Indic / multilingual LLMs for QE & APE.

This module is import-light by design: it does **not** import ``transformers``
or ``torch``. It is a pure-Python lookup table consumed by the CLI / configs
to resolve short aliases into HuggingFace model paths and recommended load
settings. The actual model loading happens in
:mod:`indicqe_ape.models.hf_provider`, which uses ``AutoModelForCausalLM``
under the hood — every entry here is therefore expected to be a causal-LM
runnable through that provider.

Usage
-----

From the CLI / configs::

    indicqe-ape qe --provider hf --model qwen2.5-7b ...
    # or, equivalently, pass the full HF path:
    indicqe-ape qe --provider hf --model Qwen/Qwen2.5-7B-Instruct ...

From Python::

    from indicqe_ape.models.indic_models import (
        INDIC_MODEL_REGISTRY, resolve, recommend_for_laptop,
    )

    cfg = INDIC_MODEL_REGISTRY["qwen2.5-7b"]
    hf_path = resolve("qwen2.5-7b")            # -> "Qwen/Qwen2.5-7B-Instruct"
    hf_path = resolve("Qwen/Qwen2.5-7B-Instruct")  # -> passes through

    laptop_alias = recommend_for_laptop()      # -> "qwen2.5-1.5b"

Registry schema
---------------

Each value in :data:`INDIC_MODEL_REGISTRY` is a ``dict`` with:

- ``hf_path`` (str): canonical HuggingFace repo id
- ``params_b`` (float): parameter count in billions
- ``dtype`` (str): recommended torch dtype, ``"bfloat16"`` or ``"float16"``
- ``quantize`` (str | None): suggested ``HFProvider.load_in_4bit`` setting —
  ``"4bit"`` if 4-bit quant is recommended for typical 24 GB GPUs, else ``None``
- ``chat_template`` (str): which chat template the tokenizer applies natively
  (``"native"`` means the model ships its own template via ``apply_chat_template``)
- ``licence`` (str): short licence tag from the model card
- ``languages`` (list[str]): ISO codes the model card claims support for
- ``notes`` (str): one-line summary, including any caveats

Sizes are taken from HuggingFace ``safetensors`` total-params metadata where
available; rounded sensibly otherwise.
"""
from __future__ import annotations

from typing import Final


INDIC_MODEL_REGISTRY: Final[dict[str, dict]] = {
    # ---------- Laptop tier (<= 2B params) ----------
    "qwen2.5-0.5b": {
        "hf_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "apache-2.0",
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "Tiny, runs on CPU / Apple Silicon MPS. Sanity-check tier.",
    },
    "qwen2.5-1.5b": {
        "hf_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "apache-2.0",
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "Best <2B multilingual baseline. Fits in <4 GB on MPS/CUDA.",
    },
    "llama-3.2-1b": {
        "hf_path": "meta-llama/Llama-3.2-1B-Instruct",
        "params_b": 1.2,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "llama3",
        "licence": "llama3.2",
        "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
        "notes": "Gated. Officially supports Hindi but no other Indic languages.",
    },
    "sarvam-1": {
        "hf_path": "sarvamai/sarvam-1",
        "params_b": 2.5,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "other",
        "languages": [
            "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te",
        ],
        "notes": "Llama-arch base trained on 11 Indic languages + English.",
    },

    # ---------- Mid tier (3-9B, 7-9B for single A5000 24GB) ----------
    "qwen2.5-3b": {
        "hf_path": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.1,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "qwen-research",  # "other" non-commercial Qwen licence
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "Strong multilingual; non-commercial licence — check before paper use.",
    },
    "llama-3.2-3b": {
        "hf_path": "meta-llama/Llama-3.2-3B-Instruct",
        "params_b": 3.2,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "llama3",
        "licence": "llama3.2",
        "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
        "notes": "Gated. Hindi-only on the Indic side.",
    },
    "airavata": {
        "hf_path": "ai4bharat/Airavata",
        "params_b": 6.9,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "llama2",
        "languages": ["en", "hi"],
        "notes": "AI4Bharat Hindi-tuned Llama-2. Gated. en-hi only.",
    },
    "qwen2.5-7b": {
        "hf_path": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7.6,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "apache-2.0",
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "Apache-2.0, fits comfortably bf16 on a single 24 GB GPU.",
    },
    "llama-3.1-8b": {
        "hf_path": "meta-llama/Llama-3.1-8B-Instruct",
        "params_b": 8.0,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "llama3",
        "licence": "llama3.1",
        "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
        "notes": "Gated. Hindi only among Indic. Strong general baseline.",
    },
    "llama-3-8b": {
        "hf_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "params_b": 8.0,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "llama3",
        "licence": "llama3",
        "languages": ["en"],
        "notes": "English-tuned baseline; useful as no-Indic-pretraining control.",
    },
    "aya-23-8b": {
        "hf_path": "CohereLabs/aya-23-8B",
        "params_b": 8.0,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "cc-by-nc-4.0",
        "languages": [
            "en", "fr", "de", "es", "it", "pt", "ja", "ko", "zh", "ar", "el",
            "fa", "pl", "id", "cs", "he", "hi", "nl", "ro", "ru", "tr", "uk", "vi",
        ],
        "notes": (
            "Cohere Aya 23. 23-language multilingual; only Hindi among Indic. "
            "Non-commercial licence."
        ),
    },
    "gemma-2-9b": {
        "hf_path": "google/gemma-2-9b-it",
        "params_b": 9.2,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "gemma",
        "languages": ["multilingual"],
        "notes": "Gated. Strong general multilingual; en->Indic decent zero-shot.",
    },
    "navarasa-7b": {
        "hf_path": "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",
        "params_b": 8.5,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "gemma",
        "languages": [
            "te", "en", "ta", "ml", "mr", "hi", "kn", "sd", "ne", "ur",
            "as", "gu", "bn", "pa", "or",
        ],
        "notes": "Gemma-7B fine-tuned on 15 Indic languages — broadest Indic coverage in this tier.",
    },

    # ---------- Workstation multi-GPU / quantized tier (>=14B) ----------
    "qwen2.5-14b": {
        "hf_path": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14.0,
        "dtype": "bfloat16",
        "quantize": None,
        "chat_template": "native",
        "licence": "apache-2.0",
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "Fits bf16 across 2x24 GB with device_map='auto'. Apache-2.0.",
    },
    "sarvam-m-24b": {
        "hf_path": "sarvamai/sarvam-m",
        "params_b": 24.0,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "native",
        "licence": "apache-2.0",
        "languages": [
            "en", "bn", "hi", "kn", "gu", "mr", "ml", "or", "pa", "ta", "te",
        ],
        "notes": (
            "Mistral-Small-3.1 24B fine-tuned for 10 Indian languages — "
            "the strongest Indic-tuned chat model that fits on a single A5000 in 4-bit."
        ),
    },
    "gemma-2-27b": {
        "hf_path": "google/gemma-2-27b-it",
        "params_b": 27.2,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "native",
        "licence": "gemma",
        "languages": ["multilingual"],
        "notes": "Gated. Use 4-bit on 2x24 GB or bf16 across both GPUs.",
    },
    "qwen2.5-32b": {
        "hf_path": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32.5,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "native",
        "licence": "apache-2.0",
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "4-bit on a single 24 GB GPU; bf16 needs both. Apache-2.0.",
    },
    "aya-23-35b": {
        "hf_path": "CohereLabs/aya-23-35B",
        "params_b": 35.0,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "native",
        "licence": "cc-by-nc-4.0",
        "languages": [
            "en", "fr", "de", "es", "it", "pt", "ja", "ko", "zh", "ar", "el",
            "fa", "pl", "id", "cs", "he", "hi", "nl", "ro", "ru", "tr", "uk", "vi",
        ],
        "notes": "Aya 23 35B. Non-commercial. 4-bit fits across 2x24 GB.",
    },
    "llama-3-70b": {
        "hf_path": "meta-llama/Meta-Llama-3-70B-Instruct",
        "params_b": 70.6,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "llama3",
        "licence": "llama3",
        "languages": ["en"],
        "notes": "Gated. 4-bit only on 2x24 GB. English-tuned but useful as upper-bound baseline.",
    },
    "llama-3.1-70b": {
        "hf_path": "meta-llama/Llama-3.1-70B-Instruct",
        "params_b": 70.6,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "llama3",
        "licence": "llama3.1",
        "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
        "notes": "Gated. 4-bit on 2x24 GB. Hindi-aware on the Indic side.",
    },
    "qwen2.5-72b": {
        "hf_path": "Qwen/Qwen2.5-72B-Instruct",
        "params_b": 72.7,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "native",
        "licence": "qwen",  # "other" — Qwen licence
        "languages": ["en", "zh", "hi", "bn", "ta", "te", "mr", "ur", "ar"],
        "notes": "4-bit just fits 2x24 GB. Best open multilingual ceiling we list.",
    },
    "nemotron-70b": {
        "hf_path": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "params_b": 70.6,
        "dtype": "bfloat16",
        "quantize": "4bit",
        "chat_template": "llama3",
        "licence": "llama3.1",
        "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
        "notes": "NVIDIA RLHF-tuned Llama-3.1-70B. Strong instruction-following baseline.",
    },
}


def resolve(alias_or_path: str) -> str:
    """Resolve a registry alias to a HuggingFace repo id.

    If ``alias_or_path`` is a key in :data:`INDIC_MODEL_REGISTRY`, returns its
    ``hf_path``. Otherwise returns the input unchanged so that callers can
    pass full HF paths (``"org/model"``) straight through.
    """
    entry = INDIC_MODEL_REGISTRY.get(alias_or_path)
    if entry is not None:
        return entry["hf_path"]
    return alias_or_path


def recommend_for_laptop() -> str:
    """Recommended laptop-runnable alias.

    ``Qwen2.5-1.5B-Instruct`` is the current pick: ~3 GB in bf16, runs on
    Apple Silicon MPS, Apache-2.0, and has the strongest multilingual
    baseline among <2B causal LMs we've verified for Indic prompts.
    """
    return "qwen2.5-1.5b"


__all__ = [
    "INDIC_MODEL_REGISTRY",
    "recommend_for_laptop",
    "resolve",
]
