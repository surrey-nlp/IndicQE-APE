#!/usr/bin/env python3
"""
extract_comet_encoder.py

Downloads a COMET checkpoint directly and extracts its XLM-R encoder weights
into a standard HuggingFace-compatible format. Uses raw PyTorch loading to
avoid dependency issues with the COMET library.

Usage:
    python scripts/extract_comet_encoder.py

Then use in a config:
    model_name: models/comet-xlmr
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)

import argparse
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--filename", default="checkpoints/model.ckpt")
    parser.add_argument("--output_dir", default="models/comet-xlmr")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token for gated models")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download the checkpoint (or use cached via hf_hub)
    print(f"Downloading/Locating COMET checkpoint for {args.repo_id} ...")
    try:
        ckpt_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            token=args.hf_token,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nNote: For gated models like wmt23-cometkiwi-da-xl, you must accept the license on HuggingFace and provide a token with --hf_token YOUR_TOKEN")
        sys.exit(1)

    # 2. Load checkpoint and inspect encoder keys
    print("Loading checkpoint ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # COMET stores encoder weights with prefix "encoder.model."
    # We need to map them to HF's XLMRobertaForSequenceClassification format
    encoder_prefix = "encoder.model."
    encoder_keys = [k for k in state_dict if k.startswith(encoder_prefix)]
    print(f"Found {len(encoder_keys)} encoder keys (prefix: '{encoder_prefix}')")

    if not encoder_keys:
        # Try alternative prefix patterns
        for prefix in ["encoder.", "model.encoder.", "model."]:
            encoder_keys = [k for k in state_dict if k.startswith(prefix)]
            if encoder_keys:
                encoder_prefix = prefix
                print(f"Found {len(encoder_keys)} keys with prefix '{prefix}'")
                break

    if not encoder_keys:
        print("ERROR: Could not find encoder keys in checkpoint.")
        print("Top-level keys:", [k.split(".")[0] for k in list(state_dict.keys())[:20]])
        sys.exit(1)

    # 3. Strip prefix and build clean state dict
    # HF XLMRobertaForSequenceClassification expects keys like:
    #   roberta.embeddings.word_embeddings.weight
    #   roberta.encoder.layer.0.attention...
    clean_state = {}
    for key in encoder_keys:
        new_key = key[len(encoder_prefix):]
        # Ensure keys start with "roberta." for XLMRobertaForSequenceClassification
        if not new_key.startswith("roberta."):
            new_key = "roberta." + new_key
        clean_state[new_key] = state_dict[key]

    # 4. Detect config from weights
    embed_weight = clean_state.get("roberta.embeddings.word_embeddings.weight")
    if embed_weight is not None:
        vocab_size, hidden_size = embed_weight.shape
        print(f"Detected: vocab_size={vocab_size}, hidden_size={hidden_size}")
    else:
        print("WARNING: Could not detect hidden_size from embeddings. Using XLM-R Large defaults.")
        hidden_size = 1024

    # 5. Create HF model with regression head
    if hidden_size == 1024:
        base_model = "xlm-roberta-large"
    elif hidden_size == 2560:
        base_model = "facebook/xlm-roberta-xl"
    else:
        base_model = "xlm-roberta-base"

    print(f"Creating XLMRobertaForSequenceClassification from {base_model} config ...")
    config = XLMRobertaConfig.from_pretrained(base_model)
    config.num_labels = 1
    config.problem_type = "regression"
    hf_model = XLMRobertaForSequenceClassification(config)

    # 6. Load encoder weights (classifier head stays randomly initialized)
    missing, unexpected = hf_model.load_state_dict(clean_state, strict=False)
    print(f"\nWeight transfer:")
    print(f"  Loaded: {len(clean_state)} tensors")
    print(f"  Missing (classifier head, expected): {missing}")
    print(f"  Unexpected: {len(unexpected)} keys")

    # 7. Save model + tokenizer
    hf_model.save_pretrained(str(OUTPUT_DIR))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Clean up the large checkpoint (hf_hub_download returns a str path)
    try:
        Path(ckpt_path).unlink()
    except OSError as e:
        print(f"(note: could not remove cached checkpoint {ckpt_path}: {e})")

    print(f"\n✅ Done! HuggingFace model saved to: {OUTPUT_DIR}")
    print(f"Use in config as:  model_name: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
