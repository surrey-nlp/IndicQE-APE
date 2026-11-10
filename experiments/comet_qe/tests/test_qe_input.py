"""Tests for the reference-free QE input construction in `data_utils.py`.

QE input is a single joined string `source </s> mt` with char-offset spans:
the **source** span and the **target = mt** span (the hypothesis being judged).
There is no "context" segment, so its offsets are -1. Getting these offsets
exactly right matters: `tokenize_function` turns them into the token masks that
drive funnelling (pool over mt) and TSA/TSE (mt->src adequacy).
"""
import numpy as np
import pandas as pd

from data_utils import (
    build_qe_input_with_offsets,
    add_qe_text_column,
    prepare_qe_features,
    qe_dataframe_to_dataset,
)


def _toy_qe_row(**extra):
    base = {
        "text": ["a </s> b"], "target_start": [6], "target_end": [7],
        "context_start": [-1], "context_end": [-1],
        "source_start": [0], "source_end": [1], "target": [0.5],
    }
    base.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(base)


def test_qe_input_src_first_offsets_roundtrip():
    text, ts, te, cs, ce, ss, se = build_qe_input_with_offsets(
        "the cat", "der Kater", sep_token="</s>", order="src_first"
    )
    assert text == "the cat </s> der Kater"
    assert text[ss:se] == "the cat"    # source span
    assert text[ts:te] == "der Kater"  # target span = mt (the judged text)
    assert (cs, ce) == (-1, -1)        # no context segment in QE


def test_qe_input_mt_first_offsets_roundtrip():
    text, ts, te, cs, ce, ss, se = build_qe_input_with_offsets(
        "the cat", "der Kater", sep_token="</s>", order="mt_first"
    )
    assert text == "der Kater </s> the cat"
    assert text[ts:te] == "der Kater"  # target span = mt, now first
    assert text[ss:se] == "the cat"    # source span, now second


def test_add_qe_text_column_populates_spans():
    df = pd.DataFrame({"source": ["a b c"], "mt": ["x y"]})
    out = add_qe_text_column(df, sep_token="</s>")
    row = out.iloc[0]
    assert row["text"] == "a b c </s> x y"
    assert row["text"][row["source_start"]:row["source_end"]] == "a b c"
    assert row["text"][row["target_start"]:row["target_end"]] == "x y"
    assert row["context_start"] == -1 and row["context_end"] == -1


def test_prepare_qe_features_length_ratio_and_train_scaling():
    train = pd.DataFrame({"source": ["a b", "a b c d"], "mt": ["w x y z", "w x"]})
    dev = pd.DataFrame({"source": ["a b"], "mt": ["w x y z"]})
    prepare_qe_features(train, dev, None, length_unit="token")
    # length_ratio = |mt tokens| / |src tokens|
    assert np.isclose(train["length_ratio"].iloc[0], 4 / 2)   # 2.0
    assert np.isclose(train["length_ratio"].iloc[1], 2 / 4)   # 0.5
    # scaler is fit on train (mean of [2.0, 0.5] = 1.25); dev uses train stats
    assert "length_ratio_scaled" in dev.columns
    assert np.isclose(train["length_ratio_scaled"].mean(), 0.0, atol=1e-9)


def test_qe_dataframe_to_dataset_minimal_columns():
    ds = qe_dataframe_to_dataset(_toy_qe_row(), label_column="target")
    assert "labels" in ds.column_names
    assert "target" not in ds.column_names  # renamed to labels
    # BEA-only feature columns must NOT be required or carried
    for c in ["lev_score_scaled", "cefr_indices", "lid_indices", "subword_ratio"]:
        assert c not in ds.column_names


def test_qe_dataframe_to_dataset_optional_features():
    df = _toy_qe_row(length_ratio_scaled=0.1, lp_indices=2)
    ds = qe_dataframe_to_dataset(
        df, label_column="target", use_length_ratio=True, use_lp=True
    )
    assert "length_ratio_scaled" in ds.column_names
    assert "lp_indices" in ds.column_names

