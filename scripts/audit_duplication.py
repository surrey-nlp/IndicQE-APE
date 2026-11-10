#!/usr/bin/env python3
"""Forensic duplication + provenance audit of surrey-nlp/IndicQE-APE. READ-ONLY.

Hypothesis under test
---------------------
A QE/APE instance is identified by its content key K = (language_pair, source, mt).
Named configs (qe-da, ape, word-qe, explainable-qe, per-pair, challenge) are meant to be
VIEWS of one underlying table: the same instance appears in several configs sharing ONE uid.

The failure mode we are hunting: explanations (or any late-added field) were attached to
existing instances by APPENDING NEW ROWS WITH FRESH UIDS instead of writing a column onto the
existing rows. That would (a) duplicate the same content under two uids, (b) inflate counts,
(c) leak content across train/test, and (d) explain why explainable-qe spans four pairs and
19k rows when human explanations exist only for en-ml.

Nothing is written to the Hub. This only reads and reports.

Checks
------
 1. uid semantics: within the master `full` config, is K <-> uid a bijection?
 2. intra-config duplication: for every config, duplicate uid / duplicate K /
    duplicate (K, post_edit).
 3. cross-config uid consistency: does any content key K map to >1 uid across configs?
 4. explainable-qe deep dive: are its non-en-ml uids a subset of `full`'s uids for those
    pairs (=views, good) or novel (=duplication)? Is any K duplicated inside it?
 5. en-ml deep dive: is the en-ml uid set identical across every config that contains en-ml?
    Any en-ml content under two uids?
 6. provenance: does any original_id map to >1 uid?

Usage:  python3 scripts/audit_duplication.py
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "surrey-nlp/IndicQE-APE"

TASK_CONFIGS = ["all", "full", "tasks/qe-da", "tasks/ape", "tasks/word-qe",
                "tasks/explainable-qe"]
PAIRS = ["en-hi", "en-mr", "en-gu", "en-ta", "en-te", "en-ml", "et-en", "ne-en", "si-en"]
COLS = ["uid", "language_pair", "source", "mt", "post_edit", "original_id"]


def h(s) -> str:
    return hashlib.md5(("" if s is None else str(s)).encode("utf-8")).hexdigest()[:16]


def load(cfg: str, splits=("train", "validation", "test")) -> pd.DataFrame:
    frames = []
    for sp in splits:
        try:
            p = hf_hub_download(REPO, f"data/{cfg}/{sp}.parquet", repo_type="dataset")
        except Exception:
            continue
        df = pd.read_parquet(p)
        keep = [c for c in COLS if c in df.columns]
        df = df[keep].copy()
        df["_split"] = sp
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    if len(out):
        out["K"] = [h(a) + "|" + h(b) + "|" + str(c)
                    for a, b, c in zip(out["source"], out["mt"], out["language_pair"])]
    return out


def main() -> None:
    print("=" * 78)
    print("LOADING CONFIGS (read-only)")
    print("=" * 78)
    data = {}
    for cfg in TASK_CONFIGS:
        df = load(cfg, ("train", "validation", "test") if cfg != "challenge" else ("test",))
        data[cfg] = df
        pairs = df["language_pair"].nunique() if len(df) else 0
        print(f"  {cfg:24s} rows={len(df):7d}  pairs={pairs}  uniq_uid={df['uid'].nunique() if len(df) else 0}")
    chal = load("challenge", ("test",))
    data["challenge"] = chal
    print(f"  {'challenge':24s} rows={len(chal):7d}  pairs={chal['language_pair'].nunique()}  uniq_uid={chal['uid'].nunique()}")

    full = data["full"]

    # ---- 1. uid semantics on the master config -------------------------------------
    print("\n" + "=" * 78)
    print("1. uid <-> content-key bijection in `full` (the master, 9 pairs)")
    print("=" * 78)
    k_per_uid = full.groupby("uid")["K"].nunique()
    uid_per_k = full.groupby("K")["uid"].nunique()
    print(f"  rows={len(full)}  distinct uid={full['uid'].nunique()}  distinct K={full['K'].nunique()}")
    print(f"  uids mapping to >1 content key : {int((k_per_uid > 1).sum())}")
    print(f"  content keys mapping to >1 uid : {int((uid_per_k > 1).sum())}   <-- >0 means DUPLICATION")
    bij = (k_per_uid.max() == 1 and uid_per_k.max() == 1
           and full["uid"].nunique() == len(full))
    print(f"  VERDICT: uid is {'a clean bijection with content (good)' if bij else 'NOT a clean key (investigate)'}")
    if (uid_per_k > 1).any():
        ex = uid_per_k[uid_per_k > 1].head(5)
        print("  example content keys with multiple uids:")
        for k in ex.index:
            us = full.loc[full.K == k, ["uid", "language_pair", "original_id"]].drop_duplicates()
            print(f"    K={k}  uids={list(us.uid)}  pair={list(us.language_pair)[:1]}  orig={list(us.original_id)}")

    # ---- 2. intra-config duplication ----------------------------------------------
    print("\n" + "=" * 78)
    print("2. intra-config duplication")
    print("=" * 78)
    print(f"  {'config':24s} {'dup_uid':>8s} {'dup_K':>8s} {'dup_K+PE':>9s}")
    for cfg, df in data.items():
        if not len(df):
            continue
        dup_uid = int(df["uid"].duplicated().sum())
        dup_k = int(df["K"].duplicated().sum())
        if "post_edit" in df.columns:
            kpe = df["K"] + "|" + df["post_edit"].astype(str).map(h)
            dup_kpe = int(kpe.duplicated().sum())
        else:
            dup_kpe = -1
        flag = "  <-- DUPES" if (dup_uid or dup_k) else ""
        print(f"  {cfg:24s} {dup_uid:8d} {dup_k:8d} {dup_kpe:9d}{flag}")

    # ---- 3. cross-config uid consistency ------------------------------------------
    print("\n" + "=" * 78)
    print("3. cross-config: does one content key ever get different uids in different configs?")
    print("=" * 78)
    k_to_uids = defaultdict(set)
    for cfg, df in data.items():
        if cfg in ("all", "full"):
            continue  # these are supersets; compare the task/pair VIEWS against each other
        for k, u in zip(df["K"], df["uid"]):
            k_to_uids[k].add(u)
    multi = {k: v for k, v in k_to_uids.items() if len(v) > 1}
    print(f"  content keys seen across task configs : {len(k_to_uids)}")
    print(f"  keys with >1 uid across configs       : {len(multi)}   <-- >0 means the 'new rows not new column' bug")
    for k, us in list(multi.items())[:8]:
        rows = full.loc[full.K == k, ["uid", "language_pair", "original_id"]].drop_duplicates()
        print(f"    K={k}  uids={sorted(us)}  {rows.to_dict('records')}")

    # ---- 4. explainable-qe deep dive ----------------------------------------------
    print("\n" + "=" * 78)
    print("4. explainable-qe: views of `full`, or novel rows?")
    print("=" * 78)
    xq = data["tasks/explainable-qe"]
    print(f"  explainable-qe rows={len(xq)}  by pair: {xq['language_pair'].value_counts().to_dict()}")
    full_uid_by_pair = {p: set(full.loc[full.language_pair == p, "uid"]) for p in PAIRS}
    full_K_by_pair = {p: set(full.loc[full.language_pair == p, "K"]) for p in PAIRS}
    for p in ["en-hi", "en-mr", "en-ta", "en-ml"]:
        sub = xq[xq.language_pair == p]
        if not len(sub):
            continue
        uid_in_full = len(set(sub.uid) & full_uid_by_pair[p])
        uid_novel = len(set(sub.uid) - full_uid_by_pair[p])
        k_in_full = len(set(sub.K) & full_K_by_pair[p])
        k_novel = len(set(sub.K) - full_K_by_pair[p])
        dup_k_here = int(sub["K"].duplicated().sum())
        print(f"  {p}: rows={len(sub):5d} | uid in full={uid_in_full:5d} novel={uid_novel:4d} "
              f"| K in full={k_in_full:5d} novel_content={k_novel:4d} | dup_K within={dup_k_here}")
    print("  Reading: uid_novel>0 with K_in_full>0 == same CONTENT re-issued under a NEW uid (duplication).")
    print("           uid all in full, novel_content 0 == clean views (good).")

    # ---- 5. en-ml deep dive --------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. en-ml uid-set consistency across every config that contains it")
    print("=" * 78)
    enml_sets = {}
    for cfg, df in data.items():
        s = set(df.loc[df.language_pair == "en-ml", "uid"])
        if s:
            enml_sets[cfg] = s
            k = set(df.loc[df.language_pair == "en-ml", "K"])
            print(f"  {cfg:24s} en-ml uids={len(s):5d}  distinct content K={len(k):5d}  "
                  f"{'(uid==K, no dup)' if len(s)==len(k) else 'MISMATCH uid!=K'}")
    ref = enml_sets.get("full", set())
    print(f"\n  reference (full) en-ml uids: {len(ref)}")
    for cfg, s in enml_sets.items():
        extra = s - ref
        print(f"  {cfg:24s} uids not in full: {len(extra)}  {'<-- NOVEL en-ml uids' if extra else ''}")

    # ---- 6. provenance -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. provenance: original_id mapping")
    print("=" * 78)
    f2 = full.dropna(subset=["original_id"])
    oid_uid = f2.groupby("original_id")["uid"].nunique()
    oid_pair = f2.groupby("original_id")["language_pair"].nunique()
    print(f"  rows with original_id: {len(f2)}  distinct original_id: {f2['original_id'].nunique()}")
    print(f"  original_id mapping to >1 uid   : {int((oid_uid > 1).sum())}")
    print(f"  original_id mapping to >1 pair  : {int((oid_pair > 1).sum())}  (cross-pair id reuse is fine if uids differ)")
    if (oid_uid > 1).any():
        for oid in oid_uid[oid_uid > 1].head(5).index:
            r = f2.loc[f2.original_id == oid, ["uid", "language_pair", "K"]].drop_duplicates()
            print(f"    original_id={oid}: {r.to_dict('records')}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    dup_full = int((uid_per_k > 1).sum())
    print(f"  content keys with multiple uids in `full`     : {dup_full}")
    print(f"  keys with multiple uids across task configs   : {len(multi)}")
    print(f"  en-ml uids outside `full` in any config       : "
          f"{sum(len(s - ref) for s in enml_sets.values())}")
    verdict = "NO DUPLICATION DETECTED" if (dup_full == 0 and len(multi) == 0) else "DUPLICATION PRESENT — SEE ABOVE"
    print(f"\n  >>> {verdict}")


if __name__ == "__main__":
    main()
