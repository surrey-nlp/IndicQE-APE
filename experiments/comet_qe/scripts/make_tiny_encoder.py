#!/usr/bin/env python3
"""Build a tiny random XLM-R encoder (real tokenizer, 2 layers, hidden 64) for
fully-offline end-to-end smoke tests of the QE pipeline. NOT for real training —
weights are random. The real runs use an extracted COMET/CometKiwi encoder.
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(tokenizer_src: str = "models/comet-xlmr", out: str = "models/tiny-xlmr") -> None:
    tok_src = REPO_ROOT / tokenizer_src
    out_dir = REPO_ROOT / out
    tok = AutoTokenizer.from_pretrained(str(tok_src))
    cfg = XLMRobertaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=514,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    XLMRobertaModel(cfg).save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"tiny encoder -> {out_dir} "
          f"(vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers})")


if __name__ == "__main__":
    main(*sys.argv[1:])
