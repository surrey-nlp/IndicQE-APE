"""Data loading for IndicQE-APE (unified release).

Retargeted (8 Jun 2026) from the old 4-subset model to the unified
`surrey-nlp/IndicQE-APE` dataset: configurable views (all / full / challenge / qe-da /
ape / per-pair), curated test-first splits (train / validation / test), and the new
schema (source, mt, post_edit, da_mean, route, is_curated, uid).

Download-once pattern: point `INDICQEAPE_DATA` (or `local_parquet_dir`) at the local
snapshot produced by `scripts/download_data.py`; everything then reads offline. If no
local snapshot is set, it falls back to `load_dataset` from the public Hub repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from .languages import all_pair_codes, to_hyphen

HF_REPO = "surrey-nlp/IndicQE-APE"            # canonical unified release (public)
HF_REPO_DEV = "sourabhd13/ape-qe-combined-data"  # legacy dev mirror (old schema)

# configurable views over the unified release.
# `mqm`, `word-qe` and `explainable-qe` are released task configs that this loader could not
# previously reach: they existed on the hub but were absent from _CONFIG_DIR, so `--subset mqm`
# raised. That is why no prompted or trained system had ever been run on the en-hi MQM layer.
CONFIGS = ("all", "full", "challenge", "qe-da", "ape", "mqm", "word-qe", "explainable-qe")
SUBSETS = CONFIGS                              # back-compat alias (cli imports SUBSETS)
SPLITS = ("train", "validation", "test")

# accept the old subset names; map them onto the new configs
_ALIAS = {"qe_only": "qe-da", "ape_only": "ape", "ape_qe_combined": "ape",
          "qe-da": "qe-da", "ape": "ape", "all": "all", "full": "full",
          "challenge": "challenge",
          # pre-June schema name for the en-hi MQM set; cli.py still defaulted to it
          "mqm_en_hi": "mqm", "mqm": "mqm",
          "word-qe": "word-qe", "explainable-qe": "explainable-qe"}

# config -> relative dir under the snapshot's data/ tree (mirrors the README YAML)
_CONFIG_DIR = {"all": "data/all", "full": "data/full", "challenge": "data/challenge",
               "qe-da": "data/tasks/qe-da", "ape": "data/tasks/ape",
               "mqm": "data/tasks/mqm", "word-qe": "data/tasks/word-qe",
               "explainable-qe": "data/tasks/explainable-qe"}
_CONFIG_DIR.update({c: f"data/pairs/{to_hyphen(c)}" for c in all_pair_codes()})


@dataclass
class Item:
    """A single evaluation item, normalised across configs (fields may be None)."""
    id: str
    pair: str                       # hyphenated, e.g. "en-hi"
    source: str
    mt: str
    pe: str | None = None           # human post-edit (alias of post_edit; APE target)
    da_scores: list[float] | None = None
    da_mean: float | None = None
    mqm_score: float | None = None  # en-hi MQM layer (0-100, higher = better), `mqm` config only
    domain: str | None = None
    route: str | None = None        # challenge difficulty axis (A1.. / easy_control / uncurated)
    is_curated: bool | None = None  # False for X->En MLQE-PE passthrough
    uid: str | None = None
    split: str | None = None


def _resolve_config(name: str) -> str:
    if name in _CONFIG_DIR:
        return name
    if name in _ALIAS:
        return _ALIAS[name]
    raise ValueError(f"Unknown config/subset {name!r}. Choose from {sorted(_CONFIG_DIR)} "
                     f"(or legacy {sorted(_ALIAS)}).")


class DataLoader:
    """Load and filter the unified IndicQE-APE configs from a local snapshot (or the Hub)."""

    def __init__(self, repo: str = HF_REPO, cache_dir: str | Path | None = None,
                 local_parquet_dir: str | Path | None = None) -> None:
        self.repo = repo
        self.cache_dir = Path(cache_dir) if cache_dir else None
        snap = local_parquet_dir or os.environ.get("INDICQEAPE_DATA")
        self.snapshot_dir = Path(snap) if snap else None
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------ load
    def load(self, config: str, split: str = "test", pair: str | None = None) -> pd.DataFrame:
        """Load a config/split (optionally filtered by pair). Cached in-memory.

        `pair` may be a 4-char ('enhi') or hyphenated ('en-hi') code.
        """
        cfg = _resolve_config(config)
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}. Choose from {SPLITS}.")
        key = (cfg, split)
        df = self._cache.get(key)
        if df is None:
            df = self._read(cfg, split)
            self._cache[key] = df
        if pair is not None:
            hp = pair if "-" in pair else to_hyphen(pair)
            df = df[df["language_pair"] == hp].reset_index(drop=True)
        return df

    def iter_items(self, config: str, split: str = "test", pair: str | None = None,
                   limit: int | None = None, seed: int = 0) -> Iterator[Item]:
        """Stream items, optionally subsampled deterministically."""
        df = self.load(config, split=split, pair=pair)
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        for row in df.itertuples(index=False):
            yield self._row_to_item(row)

    def stats(self) -> pd.DataFrame:
        """Row counts per config × split × pair (handy for the paper table)."""
        rows = []
        for cfg in CONFIGS:
            for split in (("test",) if cfg == "challenge" else SPLITS):
                try:
                    df = self.load(cfg, split=split)
                except (FileNotFoundError, ValueError):
                    continue
                for pair, n in df["language_pair"].value_counts().items():
                    rows.append({"config": cfg, "split": split, "pair": pair, "n": int(n)})
        return pd.DataFrame(rows).sort_values(["config", "split", "pair"]).reset_index(drop=True)

    # ------------------------------------------------------------------ I/O
    def _read(self, cfg: str, split: str) -> pd.DataFrame:
        # 1) local snapshot (download-once)
        if self.snapshot_dir is not None:
            path = self.snapshot_dir / _CONFIG_DIR[cfg] / f"{split}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self._validate(df)
                return df
        # 2) the Hub, by direct parquet path.
        #
        # This deliberately does NOT use load_dataset(repo, cfg). load_dataset resolves `cfg`
        # against the config names declared in the dataset card's YAML, which covers the task
        # configs but not the per-pair ones, so `--pair enml` raised FileNotFoundError even though
        # data/pairs/en-ml/train.parquet exists on the hub. Fetching the parquet path from
        # _CONFIG_DIR is the same resolution the local-snapshot branch above uses, so both branches
        # now agree and every released config is reachable without a card change.
        errors = []
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
            rel = f"{_CONFIG_DIR[cfg]}/{split}.parquet"
            p = hf_hub_download(self.repo, rel, repo_type="dataset",
                                cache_dir=str(self.cache_dir) if self.cache_dir else None)
            df = pd.read_parquet(p)
            self._validate(df)
            return df
        except Exception as e:  # noqa: BLE001
            errors.append(f"parquet path {_CONFIG_DIR.get(cfg)}/{split}.parquet: {e!r}")

        # 3) last resort: the datasets loader, for anything the card declares but we do not map.
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            ds = load_dataset(self.repo, cfg, split=split,
                              cache_dir=str(self.cache_dir) if self.cache_dir else None)
            df = ds.to_pandas()
            self._validate(df)
            return df
        except Exception as e:  # noqa: BLE001
            errors.append(f"load_dataset({cfg!r}): {e!r}")

        raise FileNotFoundError(
            f"Config {cfg!r} split {split!r} not found locally or on {self.repo!r}. "
            f"Tried: " + " | ".join(errors) +
            f". For offline use run scripts/download_data.py once and set INDICQEAPE_DATA.")

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"language_pair", "source", "mt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)}; got {sorted(df.columns)}.")

    @staticmethod
    def _row_to_item(row) -> Item:
        d = row._asdict() if hasattr(row, "_asdict") else dict(row)

        def s(x):
            if x is None:
                return None
            try:
                if pd.isna(x):
                    return None
            except (TypeError, ValueError):
                pass
            return x

        def lst(x):
            v = s(x)
            return list(v) if v is not None else None

        da = s(d.get("da_mean"))
        mqm = s(d.get("mqm_score"))
        pe = s(d.get("post_edit")) if "post_edit" in d else s(d.get("pe"))
        return Item(
            id=str(d.get("uid") or d.get("id") or ""),
            pair=str(d.get("language_pair", "")),
            source=str(d.get("source") or d.get("original") or ""),
            mt=str(d.get("mt") or d.get("translation") or ""),
            pe=pe,
            da_scores=lst(d.get("da_scores") if "da_scores" in d else d.get("scores")),
            da_mean=float(da) if da is not None else None,
            mqm_score=float(mqm) if mqm is not None else None,
            domain=s(d.get("domain")),
            route=s(d.get("route")),
            is_curated=bool(d["is_curated"]) if d.get("is_curated") is not None else None,
            uid=s(d.get("uid")),
            split=s(d.get("split")),
        )
