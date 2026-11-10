"""Tests for the reference-free QE data adapter (`src/qe_data.py`).

The adapter turns the IndicQE `qe-da` parquet into the CSV the BEA training
pipeline consumes, with a per-pair z-standardised regression target (decision
D2: the official `da_z_mean` is too patchy per pair to use directly).
"""
import numpy as np
import pandas as pd

from qe_data import (
    fit_perpair_scaler,
    apply_perpair_scaler,
    inverse_transform,
    build_qe_dataframe,
    save_scaler,
    load_scaler,
)


def _toy_train():
    # Pair en-hi centred ~80, pair en-ml centred ~20 — deliberately different
    # locations and scales so a single global scaler would NOT give unit std.
    return pd.DataFrame({
        "uid": [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)],
        "language_pair": ["en-hi"] * 4 + ["en-ml"] * 4,
        "da_mean": [70.0, 90.0, 75.0, 85.0, 10.0, 30.0, 15.0, 25.0],
    })


def test_perpair_target_is_zero_mean_unit_std_on_train():
    train = _toy_train()
    scaler = fit_perpair_scaler(train, label="da_mean", group="language_pair")
    out = apply_perpair_scaler(
        train, scaler, label="da_mean", group="language_pair", out="target"
    )
    for pair, g in out.groupby("language_pair"):
        assert abs(g["target"].mean()) < 1e-9, f"{pair} target mean not ~0"
        # StandardScaler uses population std (ddof=0); target should be unit std.
        assert abs(g["target"].std(ddof=0) - 1.0) < 1e-9, f"{pair} target std not ~1"


def test_inverse_transform_recovers_da_mean():
    train = _toy_train()
    scaler = fit_perpair_scaler(train)
    out = apply_perpair_scaler(train, scaler, out="target")
    recovered = [
        inverse_transform(row["target"], row["language_pair"], scaler)
        for _, row in out.iterrows()
    ]
    assert np.allclose(recovered, train["da_mean"].to_numpy(), atol=1e-9)


def _toy_frames():
    tr = _toy_train()
    tr["source"] = ["s"] * len(tr)
    tr["mt"] = ["m"] * len(tr)
    tr["da_z_mean"] = np.nan
    dev = pd.DataFrame({
        "uid": ["d0", "d1"], "language_pair": ["en-hi", "en-ml"],
        "da_mean": [80.0, 20.0], "source": ["s", "s"], "mt": ["m", "m"],
        "da_z_mean": [np.nan, np.nan],
    })
    test = pd.DataFrame({
        "uid": ["t0", "t1"], "language_pair": ["en-hi", "en-ml"],
        "da_mean": [60.0, 40.0], "source": ["s", "s"], "mt": ["m", "m"],
        "da_z_mean": [np.nan, np.nan],
    })
    return {"train": tr, "validation": dev, "test": test}


def test_build_qe_dataframe_schema_and_split_mapping():
    combined, scaler = build_qe_dataframe(_toy_frames())
    assert list(combined.columns) == [
        "item_id", "subset", "source", "mt",
        "language_pair", "da_mean", "da_z_mean", "target",
    ]
    # validation -> dev; train/test kept verbatim
    assert set(combined["subset"]) == {"train", "dev", "test"}
    # Scaler is fit on TRAIN only: train en-hi = [70,90,75,85], mean 80 -> target mean 0
    tr_hi = combined[(combined.subset == "train") & (combined.language_pair == "en-hi")]
    assert abs(tr_hi["target"].mean()) < 1e-9
    # A dev row whose da_mean equals the train pair-mean maps to target 0 (uses train stats)
    dev_hi = combined[(combined.subset == "dev") & (combined.language_pair == "en-hi")]
    assert abs(dev_hi["target"].iloc[0]) < 1e-9


def test_build_qe_dataframe_pair_filter():
    combined, scaler = build_qe_dataframe(_toy_frames(), pairs=["en-hi"])
    assert set(combined["language_pair"]) == {"en-hi"}
    assert "en-ml" not in set(scaler)


def test_scaler_json_roundtrip(tmp_path):
    scaler = fit_perpair_scaler(_toy_train())
    path = tmp_path / "scaler.json"
    save_scaler(scaler, path)
    assert load_scaler(path) == scaler
