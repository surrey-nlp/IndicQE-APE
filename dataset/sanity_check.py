#!/usr/bin/env python3
"""Standalone data-sanity check for the IndicQE-APE release.

Self-contained: depends only on pandas / numpy / pyarrow (+ huggingface_hub and
datasets ONLY for the optional `--repo` / `--external` modes). It does NOT import
the build scripts, so it can ship inside the HuggingFace repo next to load_test.py
and verify the *published* data on its own.

It re-derives and checks everything we care about, computable from the public
columns alone:

  IDENTITY     uid = sha1(norm(source)|norm(mt))[:16] recomputed for every row;
               uid unique per config; cadence_id format + uniqueness.
  DA INTEGRITY da_mean in [0,100] or null; da_scores all numeric (no char
               corruption); da_mean == mean(da_scores); n_annotators == len(scores);
               da_std == std(scores).
  CURATION     is_curated (X->En = False); challenge per curated pair >= 1000;
               route vocabulary; has_da/has_pe flags consistent.
  CROSS-CONFIG challenge==full test; all subset of full; all has no X->En;
               qe-da == rows with DA; ape == rows with a post-edit; pairs partition full.
  GATING       no RL columns in public configs; no en-ml in public.
  SPLITS       uid disjoint across train/val/test; source-text disjoint train vs test.
  EXTERNAL     (--external) exact-match source/mt text + da_mean vs the public
               source C (surrey-nlp/Low-resource-QE-DA-dataset).

Usage
-----
  python sanity_check.py                       # auto-find ./data or ../release_build/hf_repo/data
  python sanity_check.py --data-dir path/to/data
  python sanity_check.py --repo surrey-nlp/IndicQE-APE   # snapshot-download then check
  python sanity_check.py --external            # also exact-match against source C (needs network)

Exit code 0 iff there are no HARD failures.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- matcher (must
# byte-match dataset/build_unified.py:norm_match / uid_of)
_WS = re.compile(r"\s+")
_QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'",
              "–": "-", "—": "-"}


def norm_match(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("<EOS>", "").replace("<eos>", "")
    for a, b in _QUOTE_MAP.items():
        s = s.replace(a, b)
    return _WS.sub("", s)


def uid_of(src: object, mt: object) -> str:
    return hashlib.sha1((norm_match(src) + "|" + norm_match(mt)).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------- report state
FAILS: list[str] = []
WARNS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS {msg}")
    else:
        FAILS.append(msg); print(f"  FAIL {msg}")


def warn(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        WARNS.append(msg); print(f"  WARN {msg}")


def parse_scores(v) -> list | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    try:
        j = json.loads(v)
        return list(j) if isinstance(j, (list, tuple)) else None
    except Exception:
        return None


def as_floats(seq) -> list[float] | None:
    if not seq:
        return None
    try:
        return [float(x) for x in seq]
    except (TypeError, ValueError):
        return None


# config -> dir under data/ (mirrors the HF layout)
CORE = {"all": "all", "full": "full", "challenge": "challenge",
        "qe-da": "tasks/qe-da", "ape": "tasks/ape"}
PAIRS = ["en-hi", "en-mr", "en-ta", "en-te", "en-gu", "et-en", "ne-en", "si-en"]
XEN = {"et-en", "ne-en", "si-en"}
CURATED = {"en-hi", "en-mr", "en-ta", "en-te", "en-gu"}
RL_COLS = {"wtag", "tqr", "synthetic_description", "error_categories", "mt_tokens",
           "has_wtag", "has_tqr", "has_synth_expl"}
ROUTES = {"A1_disagreement", "A2_edit_effort", "A3_word_error", "A4_contradiction",
          "easy_control", "pair_floor", "distributional", "co_selected", "uncurated"}
CADENCE_RE = re.compile(r"^[A-Z0-9]+-[A-Z]+-\d{6}$")


def load_cfg(data_dir: str, cfg: str) -> pd.DataFrame | None:
    sub = CORE.get(cfg, f"pairs/{cfg}")
    parts = []
    for split in ["train", "validation", "test"]:
        p = os.path.join(data_dir, sub, f"{split}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p); d["split"] = split; parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


# ----------------------------------------------------------------- checks
def check_identity(data_dir):
    print("\n========== IDENTITY (uid / cadence) ==========")
    for cfg in ["all", "full"]:
        df = load_cfg(data_dir, cfg)
        if df is None:
            continue
        recomputed = np.array([uid_of(s, m) for s, m in zip(df["source"], df["mt"])])
        mis = int((df["uid"].values != recomputed).sum())
        check(mis == 0, f"[{cfg}] uid = sha1(norm(source)|norm(mt)) for all rows (mismatches={mis}, n={len(df):,})")
        check(int(df["uid"].duplicated().sum()) == 0, f"[{cfg}] uid unique within config")
        if "cadence_id" in df:
            bad = int((~df["cadence_id"].astype(str).map(lambda x: bool(CADENCE_RE.match(x)))).sum())
            check(bad == 0, f"[{cfg}] cadence_id matches {{SRC}}-{{PAIR}}-{{seq6}} (bad={bad})")
            check(bool(df["cadence_id"].is_unique), f"[{cfg}] cadence_id unique")


def check_da(data_dir):
    print("\n========== DA INTEGRITY ==========")
    df = load_cfg(data_dir, "full")
    if df is None:
        return
    da = pd.to_numeric(df["da_mean"], errors="coerce")
    check(int(((da < 0) | (da > 100)).sum()) == 0, "da_mean in [0,100] or null")

    corrupt = mean_off = nann_off = std_off = null_with_scores = 0
    z_corrupt = z_off = 0
    n_scored = n_zscored = 0
    has_z = "da_z_scores" in df.columns
    for _, r in df.iterrows():
        sc = parse_scores(r.get("da_scores"))
        dm = r.get("da_mean")
        if sc is not None:
            fl = as_floats(sc)
            if fl is None:
                corrupt += 1
            else:
                n_scored += 1
                if pd.isna(dm):
                    null_with_scores += 1          # the en-mr domain bug class
                elif abs(float(np.mean(fl)) - float(dm)) > 0.01:
                    mean_off += 1
                na = r.get("n_annotators")
                if pd.notna(na) and int(na) != len(fl):
                    nann_off += 1
                ds = r.get("da_std")
                if pd.notna(ds) and len(fl) > 1 and abs(float(np.std(fl)) - float(ds)) > 0.01:
                    std_off += 1
        if has_z:
            zs = parse_scores(r.get("da_z_scores"))
            if zs is not None:
                zfl = as_floats(zs)
                if zfl is None:
                    z_corrupt += 1
                else:
                    n_zscored += 1
                    zm = r.get("da_z_mean")
                    if pd.notna(zm) and abs(float(np.mean(zfl)) - float(zm)) > 0.01:
                        z_off += 1
    check(corrupt == 0, f"da_scores all numeric — no char corruption (corrupt lists={corrupt})")
    check(null_with_scores == 0, f"no row has da_scores present but da_mean null (violations={null_with_scores})")
    check(mean_off == 0, f"da_mean == mean(da_scores) within 0.01 (scored rows={n_scored:,}, off={mean_off})")
    check(nann_off == 0, f"n_annotators == len(da_scores) (off={nann_off})")
    warn(std_off == 0, f"da_std == std(da_scores) within 0.01 (off={std_off})")
    if has_z:
        check(z_corrupt == 0, f"da_z_scores all numeric (corrupt lists={z_corrupt})")
        check(z_off == 0, f"da_z_mean == mean(da_z_scores) within 0.01 (z-scored rows={n_zscored:,}, off={z_off})")
    # flag consistency
    check(bool((df["has_da"].astype(bool) == da.notna()).all()), "has_da flag == da_mean present")
    # text cleanliness: the C X->En corruption was the annotator-score array appended to
    # the MT after a TAB ("...gift\".\t[46, 20, 50, ...]"). Match that exact signature
    # (tab + bracketed integer list) — not legitimate brackets/tabs in normal text.
    emb = re.compile(r"\t\s*\[\s*\d")
    bad_txt = int(df["mt"].astype(str).map(lambda s: bool(emb.search(s))).sum()
                  + df["source"].astype(str).map(lambda s: bool(emb.search(s))).sum())
    check(bad_txt == 0, f"no tab-appended score array in source|mt text (violations={bad_txt})")
    # X->En DA coverage (recovered from wmt20_mlqe_task1)
    ch = load_cfg(data_dir, "challenge")
    if ch is not None:
        xt = ch[ch["language_pair"].isin(XEN)]
        cov = pd.to_numeric(xt["da_mean"], errors="coerce").notna().mean() if len(xt) else 1.0
        check(cov > 0.99, f"X->En challenge rows have da_mean (coverage={cov:.1%})")


def check_curation(data_dir):
    print("\n========== CURATION (is_curated / floor / routes) ==========")
    ch = load_cfg(data_dir, "challenge")
    if ch is None:
        return
    xen = ch[ch["language_pair"].isin(XEN)]
    check(bool((~xen["is_curated"]).all()), f"X->En challenge rows all is_curated=False (n={len(xen):,})")
    enx = ch[~ch["language_pair"].isin(XEN)]
    check(bool(enx["is_curated"].all()), f"En->Indic challenge rows all is_curated=True (n={len(enx):,})")
    per = ch.groupby("language_pair").size()
    print("  challenge test per pair:", per.to_dict())
    floored = {p: int(n) for p, n in per.items() if p in CURATED}
    check(all(n >= 1000 for n in floored.values()), f"every curated pair >=1000 challenge rows (min={min(floored.values())})")
    bad_routes = set(ch["route"].unique()) - ROUTES
    check(not bad_routes, f"route vocabulary valid (unexpected={bad_routes})")


def check_cross_config(data_dir):
    print("\n========== CROSS-CONFIG INVARIANTS ==========")
    full = load_cfg(data_dir, "full"); allc = load_cfg(data_dir, "all")
    ch = load_cfg(data_dir, "challenge"); qe = load_cfg(data_dir, "qe-da"); ape = load_cfg(data_dir, "ape")
    if full is None or allc is None:
        return
    full_test = set(full[full.split == "test"]["uid"])
    if ch is not None:
        check(set(ch["uid"]) == full_test, "challenge test uid set == full test uid set")
    check(set(allc["uid"]).issubset(set(full["uid"])), "all ⊆ full (uid)")
    check(int((allc["language_pair"].isin(XEN)).sum()) == 0, "all carries no X->En rows")
    if qe is not None:
        check(set(qe["uid"]) == set(allc[allc["has_da"].astype(bool)]["uid"]), "qe-da uid set == all rows with DA")
    if ape is not None:
        check(set(ape["uid"]) == set(allc[allc["has_pe"].astype(bool)]["uid"]), "ape uid set == all rows with a post-edit")
    # per-pair partition of full
    pair_uns, ok_pairs = set(), True
    for pr in PAIRS:
        d = load_cfg(data_dir, pr)
        if d is None:
            continue
        if not (d["language_pair"] == pr).all():
            ok_pairs = False
        pair_uns |= set(d["uid"])
    check(ok_pairs, "each per-pair config holds only its own language_pair")
    if pair_uns:
        check(pair_uns == set(full["uid"]), "union of per-pair configs == full (uid)")


def check_gating(data_dir):
    print("\n========== GATING (no RL columns / no en-ml in public) ==========")
    for cfg in ["all", "full", "qe-da", "ape", "challenge"]:
        df = load_cfg(data_dir, cfg)
        if df is None:
            continue
        leaked = RL_COLS & set(df.columns)
        check(not leaked, f"[{cfg}] no RL columns present ({sorted(leaked)})")
        check(int((df["language_pair"] == "en-ml").sum()) == 0, f"[{cfg}] no en-ml rows")
    check(not os.path.isdir(os.path.join(data_dir, "pairs", "en-ml")), "no public en-ml pair dir")


def check_splits(data_dir):
    print("\n========== SPLITS / LEAKAGE ==========")
    for cfg in ["all", "qe-da", "ape"]:
        df = load_cfg(data_dir, cfg)
        if df is None or df["split"].nunique() < 2:
            continue
        d = {sp: g for sp, g in df.groupby("split")}
        uids = {sp: set(g["uid"]) for sp, g in d.items()}
        for a in uids:
            for b in uids:
                if a < b:
                    check(len(uids[a] & uids[b]) == 0, f"[{cfg}] uid disjoint {a}∩{b}")
        if "train" in d and "test" in d:
            st = set(d["train"]["source"].map(norm_match)); se = set(d["test"]["source"].map(norm_match))
            warn(len(st & se) == 0, f"[{cfg}] source-text disjoint train∩test (shared={len(st & se)})")


def check_external(data_dir):
    """Exact-match vs source C, uid-group-aware.

    C contains near-duplicate instances (same normalised content -> same uid, but
    different surface text / DA). The content-hash merge collapses those to one row.
    So fidelity is: the release's surface text must be ONE OF C's surfaces for that
    uid, and its da_mean must be ONE OF C's means (within tol) OR a self-consistency
    correction (== mean of the release's own da_scores). Comparing row-for-row would
    falsely flag the collapsed duplicates.
    """
    print("\n========== EXTERNAL: exact-match vs source C (uid-group-aware) ==========")
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        WARNS.append("external: huggingface_hub not installed"); print("  WARN huggingface_hub missing — skip"); return
    full = load_cfg(data_dir, "full")
    if full is None:
        return

    # release uid -> (source, mt, da_mean, mean(da_scores))
    rel = {}
    for r in full.itertuples(index=False):
        fl = as_floats(parse_scores(getattr(r, "da_scores", None)))
        rel[r.uid] = (r.source, r.mt, r.da_mean, (float(np.mean(fl)) if fl else None))

    # source C grouped by uid: surfaces + means
    repo = "surrey-nlp/Low-resource-QE-DA-dataset"
    codes = ["enhi", "enmr", "enta", "engu", "ente", "eten", "neen", "sien"]
    csurf: dict[str, set] = {}
    cmeans: dict[str, list] = {}
    for code in codes:
        for sp in ["train", "dev", "test"]:
            try:
                d = pd.read_parquet(hf_hub_download(repo, f"parquet/{code}/{sp}.parquet", repo_type="dataset"))
            except Exception:
                continue
            for _, x in d.iterrows():
                u = uid_of(x["original"], x["translation"])
                csurf.setdefault(u, set()).add((str(x["original"]), str(x["translation"])))
                if pd.notna(x.get("mean")):
                    cmeans.setdefault(u, []).append(float(x["mean"]))

    seen = text_bad = da_bad = corrected = 0
    for u, surfs in csurf.items():
        if u not in rel:
            continue
        seen += 1
        rs, rm, rdm, rms = rel[u]
        if (str(rs), str(rm)) not in surfs:
            text_bad += 1
        means = cmeans.get(u, [])
        if pd.notna(rdm) and means:
            in_c = any(abs(float(rdm) - m) <= 0.01 for m in means)
            is_corr = (rms is not None and abs(float(rdm) - rms) <= 0.01)
            if not in_c and not is_corr:
                da_bad += 1
            elif not in_c and is_corr:
                corrected += 1
    check(seen > 0, f"matched {seen:,} unique source-C instances into the release")
    check(text_bad == 0, f"release source/mt is one of C's surface forms for the uid (violations={text_bad})")
    check(da_bad == 0, f"release da_mean is a C value or a self-consistency correction (unexplained={da_bad})")
    print(f"  note: {corrected} C-instance da_mean(s) are self-consistency corrections (mean!=mean(scores) in C)")


def find_data_dir(arg: str | None) -> str:
    if arg:
        return arg
    for c in ["data", "release_build/hf_repo/data", "dataset/release_build/hf_repo/data",
              os.path.join(os.path.dirname(__file__), "release_build/hf_repo/data")]:
        if os.path.isdir(c):
            return c
    sys.exit("Could not find a data/ directory — pass --data-dir or --repo.")


def main() -> int:
    ap = argparse.ArgumentParser(description="IndicQE-APE data sanity check")
    ap.add_argument("--data-dir", help="path to the data/ directory (default: auto-detect)")
    ap.add_argument("--repo", help="snapshot-download this HF dataset's data/** and check it")
    ap.add_argument("--external", action="store_true", help="also exact-match vs source C (needs network)")
    a = ap.parse_args()

    if a.repo:
        from huggingface_hub import snapshot_download
        path = snapshot_download(a.repo, repo_type="dataset", allow_patterns=["data/**"])
        data_dir = os.path.join(path, "data")
    else:
        data_dir = find_data_dir(a.data_dir)
    print(f"data dir: {data_dir}")

    check_identity(data_dir)
    check_da(data_dir)
    check_curation(data_dir)
    check_cross_config(data_dir)
    check_gating(data_dir)
    check_splits(data_dir)
    if a.external:
        check_external(data_dir)

    print("\n========== SUMMARY ==========")
    print(f"HARD failures: {len(FAILS)}")
    for f in FAILS:
        print(f"  FAIL {f}")
    print(f"WARNINGS: {len(WARNS)}")
    for w in WARNS:
        print(f"  WARN {w}")
    print("\nRESULT:", "PASS ✓" if not FAILS else "FAIL ✗")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
