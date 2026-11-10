"""Unit tests for train-time plumbing that the QE path depends on.

The custom data collator must carry the QE feature columns through to the model
with the right dtypes (float scalar for length_ratio, long index for the
language-pair embedding), just as it already does for the BEA features.
"""
import torch
from transformers import AutoTokenizer

from train import CustomDataCollator

# The local COMET encoder dir ships a tokenizer (no weights needed for this test).
TOKENIZER_DIR = "models/comet-xlmr"


def _collator():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return CustomDataCollator(tokenizer=tok)


def test_collator_passes_qe_features_with_correct_dtypes():
    feats = [
        {"input_ids": [0, 5, 6, 2], "attention_mask": [1, 1, 1, 1],
         "target_mask": [0, 1, 1, 0], "source_mask": [0, 1, 0, 0],
         "length_ratio_scaled": 0.3, "lp_indices": 2, "labels": 0.5},
        {"input_ids": [0, 7, 2], "attention_mask": [1, 1, 1],
         "target_mask": [0, 1, 0], "source_mask": [0, 1, 0],
         "length_ratio_scaled": -0.1, "lp_indices": 1, "labels": -0.2},
    ]
    batch = _collator()(feats)
    assert batch["length_ratio_scaled"].dtype == torch.float
    assert batch["length_ratio_scaled"].shape == (2,)
    assert batch["lp_indices"].dtype == torch.long
    assert batch["lp_indices"].tolist() == [2, 1]
    # masks still padded to the max input length in the batch
    assert batch["target_mask"].shape == batch["input_ids"].shape
