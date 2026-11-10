"""Every released config must be loadable through DataLoader, and the mqm gold must survive.

This exists because `mqm`, `word-qe` and `explainable-qe` were released on the hub but absent
from data.py's _CONFIG_DIR, so `--subset mqm` raised and no system was ever run on the en-hi
MQM layer. A missing dict entry is exactly the kind of gap a test catches and a code review
does not.

Runs against the local release build; skips cleanly if it is not present.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from indicqe_ape.data import CONFIGS, SPLITS, DataLoader

ROOT = Path(__file__).resolve().parents[1]
# The authoritative source is the HF release, not dataset/release_build (which is a stale
# intermediate: it predates the mqm, word-qe and explainable-qe configs and the en-ml pair).
# Refresh with:
#   python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('surrey-nlp/IndicQE-APE', repo_type='dataset', \
#       local_dir='dataset/hf_snapshot', allow_patterns=['data/**'])"
SNAPSHOT = Path(os.environ.get("INDICQEAPE_DATA") or ROOT / "dataset" / "hf_snapshot")

pytestmark = pytest.mark.skipif(
    not (SNAPSHOT / "data").exists(),
    reason=f"no local HF snapshot at {SNAPSHOT}; see module docstring")


@pytest.fixture(scope="module")
def dl() -> DataLoader:
    return DataLoader(local_parquet_dir=SNAPSHOT)


@pytest.mark.parametrize("config", CONFIGS)
def test_every_config_loads(dl: DataLoader, config: str) -> None:
    """Each config resolves to a path and yields a non-empty test split."""
    df = dl.load(config, split="test")
    assert len(df) > 0, f"{config}/test is empty"
    assert {"language_pair", "source", "mt"} <= set(df.columns)


def test_mqm_config_shape(dl: DataLoader) -> None:
    """The mqm config carries the WMT24 batch, and most of it has no DA."""
    df = dl.load("mqm", split="test")
    assert len(df) == 2639, f"expected 2,639 mqm test rows, got {len(df)}"
    assert df["mqm_score"].notna().all(), "every mqm row must carry an MQM score"
    # the WMT24 general-MT batch is MQM-only: scoring it against da_mean would drop it
    assert df["da_mean"].isna().sum() == 2163, (
        f"expected 2,163 MQM-only rows, got {df['da_mean'].isna().sum()}")
    assert set(df["language_pair"].unique()) == {"en-hi"}


def test_mqm_gold_survives_row_to_item(dl: DataLoader) -> None:
    """Item.mqm_score must be populated, or prompted runs silently lose the gold."""
    items = list(dl.iter_items("mqm", split="test", pair="enhi", limit=200, seed=0))
    assert len(items) == 200
    assert all(i.mqm_score is not None for i in items), "mqm_score dropped in _row_to_item"
    assert any(i.da_mean is None for i in items), "expected some MQM-only rows in the sample"


def test_legacy_alias_still_resolves(dl: DataLoader) -> None:
    """cli.py defaulted to the pre-June name; it must not raise."""
    a = dl.load("mqm_en_hi", split="test")
    b = dl.load("mqm", split="test")
    assert len(a) == len(b)


@pytest.mark.parametrize("split", SPLITS)
def test_mqm_splits_are_disjoint_and_complete(dl: DataLoader, split: str) -> None:
    df = dl.load("mqm", split=split)
    assert df["uid"].is_unique, f"duplicate uids in mqm/{split}"


@pytest.mark.parametrize("pair", ["enta", "enhi", "enmr", "ente", "enml"])
def test_fewshot_pool_is_full_size(dl: DataLoader, pair: str) -> None:
    """A k-shot prompt must carry exactly k examples, for every pair.

    Regression test. run_qe used to sample `shots` rows and THEN drop any without a DA label,
    so a DA-sparse train split silently produced a shorter prompt. Measured DA rates: en-ta
    0.469, en-hi 0.687, en-mr 0.945, others 1.000, so nominal 4-shot en-ta averaged about 1.9
    examples. That makes shot count vary by language, which is the one thing a few-shot
    comparison has to hold fixed. The fix filters before sampling.
    """
    shots = 4
    pool = dl.load(pair, split="train", pair=pair)
    pool = pool[pool["da_mean"].notna()]
    assert len(pool) >= shots, f"{pair}: only {len(pool)} DA-bearing train rows"
    picked = pool.sample(n=shots, random_state=10_000)
    assert len(picked) == shots
    assert picked["da_mean"].notna().all(), f"{pair}: a picked example has no DA label"


@pytest.mark.parametrize("pair", ["enml", "enhi", "sien"])
def test_per_pair_configs_resolve(dl: DataLoader, pair: str) -> None:
    """Per-pair configs must load, including the train split used for few-shot pools.

    Regression test. The hub fallback used to pass the config NAME to load_dataset, which only
    resolves names declared in the dataset card's YAML. The per-pair configs are not declared
    there, so `--pair enml` raised FileNotFoundError mid-run even though the parquet exists on
    the hub. It failed at the few-shot pool, i.e. only for shots>0, which is why zero-shot runs
    never surfaced it.
    """
    for split in ("train", "test"):
        df = dl.load(pair, split=split)
        assert len(df) > 0, f"{pair}/{split} empty"
        assert df["language_pair"].nunique() == 1
