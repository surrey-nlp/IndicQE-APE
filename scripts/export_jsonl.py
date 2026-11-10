#!/usr/bin/env python3
"""Dump a config/pair/split from the local snapshot to flat JSONL for file-based tools
(mt-metrix, the ALOPE repo scripts). Uses the package DataLoader (offline).

  python scripts/export_jsonl.py --config challenge --pair enhi --require-da --out x.jsonl
  python scripts/export_jsonl.py --config enhi --split train --require-da --out tr.jsonl
"""
from __future__ import annotations
import argparse
import os
from indicqe_ape.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--pair")
ap.add_argument("--require-da", action="store_true")
ap.add_argument("--require-pe", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = DataLoader().load(a.config, split=a.split, pair=a.pair)
if a.require_da:
    df = df[df["da_mean"].notna()]
if a.require_pe and "post_edit" in df.columns:
    df = df[df["post_edit"].notna()]
cols = [c for c in ["uid", "language_pair", "source", "mt", "da_mean", "post_edit",
                    "route", "is_curated"] if c in df.columns]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df[cols].to_json(a.out, orient="records", lines=True, force_ascii=False)
print(f"wrote {a.out} ({len(df)} rows)")
