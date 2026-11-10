from __future__ import annotations

"""
qe_data.py

Reference-free MT Quality Estimation data adapter for the IndicQE-APE project.

Turns the IndicQE `qe-da` parquet splits into the CSV shape the BEA hybrid
training pipeline consumes, with a **per-pair z-standardised** regression target.

Why per-pair z (decision D2, QE_ADAPTATION/08_decision_log.md): the official
`da_z_mean` is present for wildly different fractions of each pair (en-hi ~50%
null, en-gu ~1% null) and is entirely absent for the gated en-ml pair, so it
cannot be the unified target. Standardising `da_mean` within each language pair
(mean/std fit on the train split) gives every pair a consistent, scale-matched
target so the Huber delta is meaningful across pairs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# The parquet split names -> the canonical split labels the BEA pipeline expects
# (`DataConfig` defaults dev_split="dev", test_split="test").
DEFAULT_SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}

# The exact columns (and order) the training CSV carries.
OUTPUT_COLUMNS = [
    "item_id", "subset", "source", "mt",
    "language_pair", "da_mean", "da_z_mean", "target",
]


def fit_perpair_scaler(
    df: pd.DataFrame,
    label: str = "da_mean",
    group: str = "language_pair",
) -> dict:
    """Fit a per-pair standardiser on `label`, returning {pair: {mean, std}}.

    Uses population std (ddof=0) to match sklearn's StandardScaler. A degenerate
    zero-variance pair is given std=1.0 so the transform is a pure re-centring
    rather than a divide-by-zero.
    """
    stats: dict = {}
    for pair, g in df.groupby(group):
        vals = g[label].dropna().to_numpy(dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        stats[pair] = {"mean": mean, "std": std if std > 0 else 1.0}
    return stats


def apply_perpair_scaler(
    df: pd.DataFrame,
    scaler: dict,
    label: str = "da_mean",
    group: str = "language_pair",
    out: str = "target",
) -> pd.DataFrame:
    """Add a standardised `out` column: (label - pair_mean) / pair_std.

    Rows whose pair is unseen in the scaler, or whose `label` is null, get NaN.
    """
    result = df.copy()

    def _z(row):
        st = scaler.get(row[group])
        value = row[label]
        if st is None or pd.isna(value):
            return np.nan
        return (float(value) - st["mean"]) / st["std"]

    result[out] = result.apply(_z, axis=1)
    return result


def inverse_transform(values, pair: str, scaler: dict):
    """Map standardised target values back to the raw `da_mean` scale for `pair`.

    Lets us report RMSE/MAE on the interpretable 0-100 DA scale even though the
    model trains and selects on the z-scale.
    """
    st = scaler[pair]
    return np.asarray(values, dtype=float) * st["std"] + st["mean"]


def build_qe_dataframe(
    frames: dict,
    pairs=None,
    item_id: str = "uid",
    label: str = "da_mean",
    group: str = "language_pair",
    split_map: dict | None = None,
):
    """Assemble the QE training frame from parquet split frames.

    Args:
        frames: {"train": df, "validation": df, "test": df} (missing/empty splits
            are skipped). The per-pair scaler is fit on the "train" frame only.
        pairs: optional whitelist of language pairs; others are dropped.
        item_id: source id column, renamed to "item_id" in the output.

    Returns:
        (combined_df, scaler) where combined_df has exactly OUTPUT_COLUMNS and the
        scaler is the per-pair {mean, std} dict fit on train.
    """
    split_map = split_map or DEFAULT_SPLIT_MAP

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        return df[df[group].isin(pairs)].copy() if pairs else df.copy()

    frames = {k: _filter(v) for k, v in frames.items()}
    scaler = fit_perpair_scaler(frames["train"], label=label, group=group)

    parts = []
    for split, df in frames.items():
        if df.empty:
            continue
        d = apply_perpair_scaler(df, scaler, label=label, group=group, out="target")
        d = d.rename(columns={item_id: "item_id"})
        d["subset"] = split_map.get(split, split)
        if "da_z_mean" not in d.columns:
            d["da_z_mean"] = np.nan
        parts.append(d[OUTPUT_COLUMNS])

    combined = pd.concat(parts, ignore_index=True)
    return combined, scaler


def save_scaler(scaler: dict, path) -> None:
    """Persist the per-pair scaler as JSON (for inverse-transform at eval time)."""
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(path) -> dict:
    """Load a per-pair scaler previously written by `save_scaler`."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
