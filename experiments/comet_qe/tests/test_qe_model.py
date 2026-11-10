"""Wiring tests for the QE additions to BEA2026HybridRegressor.

Uses a tiny randomly-initialised XLM-R encoder saved to a temp dir, so no model
download is needed. The key property under test: the two new QE features
(`use_length_ratio`, `use_lp`) each add exactly the right number of dimensions to
`mlp_input_dim`, and a forward pass runs (a dim mismatch would throw at the head).
"""
import torch
from transformers import AutoConfig, XLMRobertaConfig, XLMRobertaModel

HIDDEN = 32


def _tiny_encoder(tmp_path):
    cfg = XLMRobertaConfig(
        vocab_size=100, hidden_size=HIDDEN, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
    )
    XLMRobertaModel(cfg).save_pretrained(tmp_path)
    return str(tmp_path)


def _hf_config(path, **flags):
    cfg = AutoConfig.from_pretrained(path)
    defaults = dict(
        use_attention=True,
        use_target_source_alignment=True,
        use_target_source_entropy=True,
        use_length_ratio=True,
        use_lp=True,
        num_language_pairs=5,
        multi_layer_funneling=False,
        deep_mlp=False,
        use_levenshtein=False,
    )
    defaults.update(flags)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _batch(B=2, S=8, num_lp=5):
    ids = torch.randint(5, 100, (B, S))
    ids[:, 0] = 0
    tmask = torch.zeros(B, S, dtype=torch.long); tmask[:, 4:7] = 1  # mt span
    smask = torch.zeros(B, S, dtype=torch.long); smask[:, 1:4] = 1  # src span
    return dict(
        input_ids=ids,
        attention_mask=torch.ones(B, S, dtype=torch.long),
        target_mask=tmask, source_mask=smask,
        context_mask=torch.zeros(B, S, dtype=torch.long),
        length_ratio_scaled=torch.randn(B),
        lp_indices=torch.randint(0, num_lp, (B,)),
        labels=torch.randn(B),
    )


def test_length_ratio_and_lp_add_dims_and_forward_runs(tmp_path):
    from hybrid_model import BEA2026HybridRegressor
    path = _tiny_encoder(tmp_path)
    model = BEA2026HybridRegressor(path, config=_hf_config(path))
    # CLS + pooled + TSA(1) + TSE(1) + length_ratio(1) + lp(lp_dim)
    assert model.mlp_input_dim == HIDDEN + HIDDEN + 1 + 1 + 1 + model.lp_dim
    out = model(**_batch(num_lp=5))
    assert out.logits.shape == (2, 1)
    assert torch.isfinite(out.loss)


def test_qe_features_off_gives_baseline_dims(tmp_path):
    from hybrid_model import BEA2026HybridRegressor
    path = _tiny_encoder(tmp_path)
    cfg = _hf_config(
        path, use_length_ratio=False, use_lp=False,
        use_target_source_alignment=False, use_target_source_entropy=False,
    )
    model = BEA2026HybridRegressor(path, config=cfg)
    assert model.mlp_input_dim == HIDDEN + HIDDEN  # CLS + pooled only
    out = model(**_batch())
    assert out.logits.shape == (2, 1)
