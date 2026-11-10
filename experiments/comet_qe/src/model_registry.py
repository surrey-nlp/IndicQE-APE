"""
model_registry.py

Model metadata registry for the BEA 2026 Lexical Difficulty Prediction Framework.
Provides model information (hidden dimensions, layer counts, head counts) to enable
auto-configuration of the MLP regression head and validation of experiment configs.

Team SurreyCTS
"""

from typing import Any, Dict, List, Optional, Tuple
import warnings


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Base Encoders (768-dim) ──────────────────────────────────────────
    "xlm-roberta-base": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "type": "encoder",
        "description": "XLM-RoBERTa Base — strong multilingual baseline",
    },
    "microsoft/mdeberta-v3-base": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "type": "encoder",
        "description": "mDeBERTa-v3 Base — disentangled attention",
    },
    "microsoft/xlm-align-base": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "type": "encoder",
        "description": "XLM-Align Base — cross-lingual alignment",
    },
    "susnato/ernie-m-base_pytorch": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "type": "encoder",
        "description": "ERNIE-M Base — Baidu multilingual encoder",
    },
    # NOTE: comet-xlmr is XLM-R *large* (1024/24/16), not base. The extracted
    # models/comet-xlmr/config.json confirms this. The previous 768/12/12 entry was
    # wrong (see QE_ADAPTATION/01_correctness_audit.md C1) and mis-set COMET funnelling.
    "models/comet-xlmr": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "type": "encoder",
        "local": True,
        "description": "COMET-22 (wmt22-comet-da) XLM-R large encoder — ref-based MT QE pretrained",
    },
    "models/cometkiwi-infoxlm": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "type": "encoder",
        "local": True,
        "description": "CometKiwi (wmt22-cometkiwi-da) InfoXLM-large encoder — reference-free MT QE pretrained",
    },

    # ── Large Encoders (1024-dim) ────────────────────────────────────────
    "FacebookAI/xlm-roberta-large": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "type": "encoder",
        "description": "XLM-RoBERTa Large",
    },
    "microsoft/infoxlm-large": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "type": "encoder",
        "description": "InfoXLM Large — information-theoretic pretraining",
    },
    "susnato/ernie-m-large_pytorch": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "type": "encoder",
        "description": "ERNIE-M Large",
    },

    # ── Extra-Large Encoders ─────────────────────────────────────────────
    "google/rembert": {
        "hidden_size": 1152,
        "num_layers": 32,
        "num_heads": 18,
        "type": "encoder",
        "description": "RemBERT — resampled multilingual BERT (default backbone)",
    },
    "facebook/xlm-roberta-xl": {
        "hidden_size": 2560,
        "num_layers": 36,
        "num_heads": 32,
        "type": "encoder",
        "description": "XLM-RoBERTa XL (3.5B params)",
    },
}

# Short aliases for convenience in configs
MODEL_ALIASES: Dict[str, str] = {
    "xlmr": "xlm-roberta-base",
    "xlmr-base": "xlm-roberta-base",
    "xlm-roberta-base": "xlm-roberta-base",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "xlmr-xl": "facebook/xlm-roberta-xl",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "mdeberta-base": "microsoft/mdeberta-v3-base",
    "rembert": "google/rembert",
    "infoxlm-base": "microsoft/infoxlm-base",
    "infoxlm-large": "microsoft/infoxlm-large",
    "xlm-align": "microsoft/xlm-align-base",
    "ernie-m-base": "susnato/ernie-m-base_pytorch",
    "ernie-m-large": "susnato/ernie-m-large_pytorch",
    "comet-xlmr": "models/comet-xlmr",
    "comet-22": "models/comet-xlmr",
    "cometkiwi": "models/cometkiwi-infoxlm",
    "cometkiwi-da": "models/cometkiwi-infoxlm",
    "cometkiwi-infoxlm": "models/cometkiwi-infoxlm",
}


def normalize_model_name(name: str) -> str:
    """Resolve short aliases to full HuggingFace model identifiers."""
    return MODEL_ALIASES.get(name, name)


def get_model_info(name: str) -> Optional[Dict[str, Any]]:
    """Look up model metadata from the registry.

    Returns None if the model is not registered (unknown models are still
    supported — they just won't get auto-configuration).
    """
    resolved = normalize_model_name(name)
    return MODEL_REGISTRY.get(resolved)


def get_default_funneling_layers(num_layers: int, num_funneling_layers: int = 4) -> List[int]:
    """Compute evenly-spaced layer indices for semantic funneling.

    Always includes the last layer (-1). Remaining layers are spread
    across the second half of the network.

    Args:
        num_layers: Total number of transformer layers.
        num_funneling_layers: How many layers to pool from (default 4).

    Returns:
        List of layer indices into `hidden_states` (1-indexed, since
        hidden_states[0] is the embedding layer).
    """
    if num_funneling_layers <= 1:
        return [-1]

    # Spread across the upper half of layers
    half = num_layers // 2
    step = half // (num_funneling_layers - 1)
    indices = [half + i * step for i in range(num_funneling_layers - 1)]
    indices.append(-1)  # Always include the last layer
    return indices


def get_entropy_layer_indices(funneling_layers: List[int], num_layers: int) -> List[int]:
    """Convert funneling layer indices (1-indexed into hidden_states) to
    attention layer indices (0-indexed into attentions tuple).

    hidden_states[0] = embedding, hidden_states[1] = layer 1 output, etc.
    attentions[0] = layer 1 attention, attentions[1] = layer 2 attention, etc.
    So: attention_index = hidden_state_index - 1 (for positive indices).
    """
    indices = []
    for idx in funneling_layers:
        if idx == -1:
            indices.append(num_layers - 1)  # Last attention layer
        else:
            indices.append(idx - 1)  # Convert 1-indexed HS to 0-indexed attn
    return indices


def get_recommended_mlp_dims(hidden_size: int) -> Tuple[int, ...]:
    """Return recommended MLP layer dimensions based on the backbone's hidden size.

    The MLP progressively narrows from a projection of the concatenated features
    down to a single regression output.
    """
    # Scalable MLP sizing
    if hidden_size >= 2560:  # e.g., XLM-R XL
        return (2048, 1024, 256, 1)
    elif hidden_size >= 1152:  # e.g., RemBERT
        return (1024, 512, 128, 1)
    elif hidden_size >= 1024:  # e.g., XLM-R Large
        return (1024, 512, 128, 1)
    else:  # e.g., XLM-R Base (768)
        return (768, 384, 128, 1)


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Run pre-flight checks on an experiment config.

    Returns a list of warning strings. Empty list = all good.
    """
    warns: List[str] = []
    model_name = normalize_model_name(cfg.get("model_name", "google/rembert"))
    info = get_model_info(model_name)

    if info is None:
        warns.append(
            f"Model '{model_name}' not in registry. MLP auto-sizing unavailable; "
            f"using transformer's config.hidden_size at runtime."
        )
        return warns

    hidden = info["hidden_size"]
    num_layers = info["num_layers"]

    # Check funneling layer validity
    funneling_layers = cfg.get("funneling_layers")
    if funneling_layers:
        for idx in funneling_layers:
            if idx != -1 and (idx < 1 or idx > num_layers):
                warns.append(
                    f"Funneling layer {idx} is out of range for {model_name} "
                    f"which has {num_layers} layers. Valid range: 1-{num_layers} or -1."
                )

    # Check deep MLP with small models
    if cfg.get("deep_mlp", False) and hidden < 768:
        warns.append(
            f"deep_mlp=True with hidden_size={hidden}. "
            f"The MLP head may be larger than the backbone. Consider disabling deep_mlp."
        )

    # Check multi-layer features with shallow models
    if cfg.get("multi_layer_funneling", False) and num_layers < 12:
        warns.append(
            f"multi_layer_funneling=True with only {num_layers} layers. "
            f"Funneling works best with deeper models (≥12 layers)."
        )

    return warns


def print_model_table():
    """Print a formatted table of all registered models."""
    print(f"\n{'Model':<45} {'Hidden':>7} {'Layers':>7} {'Heads':>6} {'Type':>8}")
    print("─" * 80)
    for name, info in MODEL_REGISTRY.items():
        print(
            f"{name:<45} {info['hidden_size']:>7} "
            f"{info['num_layers']:>7} {info['num_heads']:>6} "
            f"{info['type']:>8}"
        )
    print()


if __name__ == "__main__":
    print_model_table()
