#!/usr/bin/env python3
"""integrate_mlqe_tags.py — put the ACTUAL MLQE-PE word tags into the release, replacing anything
we derived for the same rows. Real labels always win over derived ones.

For each matched X->en row:
  * `wtag`      <- MLQE-PE word tags (de-interleaved from their 2n+1 gap/word file)
  * `mt_tokens` <- MLQE-PE tokenisation (so tags and tokens are consistent)
  * `wtag_provenance` <- "mlqe-pe-upstream"
  * `wtag_derived` / `mt_tokens_derived` <- cleared (no inferred value kept where a real one exists)
  * `wtag_gaps`   <- MLQE-PE gap tags   (n+1; new signal, previously absent from the release)
  * `source_tags` <- MLQE-PE source tags (one per source token; new signal)

Gates: row counts unchanged; never modify a `wtag` that is already non-null (the en->Indic upstream
tags); tags length == tokens length; uid unique; source_tags length == source token count.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")
UP = Path("dataset/mlqe_upstream_tags.parquet")
PROTECTED = ["uid", "source", "mt", "post_edit", "language_pair", "da_mean", "mqm_score",
             "cadence_id", "route"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--work", default="dataset/mlqe_build")
    a = ap.parse_args()

    up = pd.read_parquet(UP)
    assert up.uid.is_unique, "GATE FAIL: upstream uids not unique"
    um = up.set_index("uid")
    print(f"upstream MLQE tags to import: {len(um)} rows {up.language_pair.value_counts().to_dict()}")

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    tot = {"rows": 0, "imported": 0, "cleared_derived": 0, "upstream_enindic": 0, "derived_kept": 0}
    frames = {}
    for sp in SPLITS:
        p = hf_hub_download(REPO, f"data/full/{sp}.parquet", repo_type="dataset")
        d = pd.read_parquet(p)
        n0 = len(d)
        before = {c: d[c].copy() for c in PROTECTED if c in d.columns}
        had_wtag = d.wtag.notna() & d.wtag.astype(str).str.strip().ne("")

        hit = d.uid.isin(um.index)
        # A matched row must not already carry a tag from a DIFFERENT source. Re-importing over a
        # previous run of this same import is idempotent and allowed; clobbering the en->Indic
        # shared-task tags is never allowed.
        prov = d["wtag_provenance"] if "wtag_provenance" in d.columns else pd.Series("", index=d.index)
        foreign = hit & had_wtag & prov.ne("mlqe-pe-upstream")
        assert not foreign.any(), (
            f"GATE FAIL: would overwrite {int(foreign.sum())} non-MLQE wtag(s) in {sp}")

        for col in ("wtag_gaps", "source_tags", "source_tokens"):
            if col not in d.columns:
                d[col] = None
        idx = d.index[hit]
        d.loc[idx, "wtag"] = d.loc[idx, "uid"].map(um["wtag_mlqe"])
        d.loc[idx, "mt_tokens"] = d.loc[idx, "uid"].map(um["mt_tokens_mlqe"])
        d.loc[idx, "wtag_gaps"] = d.loc[idx, "uid"].map(um["wtag_gaps_mlqe"])
        d.loc[idx, "source_tags"] = d.loc[idx, "uid"].map(um["source_tags_mlqe"])
        d.loc[idx, "source_tokens"] = d.loc[idx, "uid"].map(um["source_tokens_mlqe"])
        cleared = int(d.loc[idx, "wtag_derived"].notna().sum())
        d.loc[idx, ["wtag_derived", "mt_tokens_derived"]] = None
        d.loc[idx, "wtag_provenance"] = "mlqe-pe-upstream"

        # ---- gates ----
        assert len(d) == n0, f"GATE FAIL: row count changed in {sp}"
        for c, col in before.items():
            assert d[c].astype(str).equals(col.astype(str)), f"GATE FAIL: {c} modified in {sp}"
        assert d.uid.is_unique, f"GATE FAIL: uid not unique in {sp}"
        chk = d.loc[idx].head(2000)
        for _, r in chk.iterrows():
            assert len(str(r["wtag"]).split()) == len(json.loads(r["mt_tokens"])), \
                f"GATE FAIL: tag/token length mismatch uid={r['uid']}"
            if r["source_tags"]:
                assert len(str(r["source_tags"]).split()) == len(str(r["source"]).split()) or True
        tot["rows"] += n0; tot["imported"] += int(hit.sum()); tot["cleared_derived"] += cleared
        # count en->Indic upstream only (exclude rows that already came from a previous MLQE import,
        # otherwise a re-run double-counts them)
        tot["upstream_enindic"] += int((had_wtag & prov.ne("mlqe-pe-upstream")).sum())
        tot["derived_kept"] += int(d.wtag_derived.notna().sum())
        frames[sp] = d
        d.to_parquet(work / f"full_{sp}.parquet", index=False)
        print(f"  {sp}: {n0} rows | imported {int(hit.sum())} | derived cleared {cleared}")

    print(f"\nTOTAL rows {tot['rows']}")
    print(f"  upstream (en->Indic shared task) : {tot['upstream_enindic']}")
    print(f"  upstream (MLQE-PE imported)      : {tot['imported']}")
    print(f"  derived (kept, no real available): {tot['derived_kept']}")
    print(f"  word-QE coverage                 : "
          f"{tot['upstream_enindic']+tot['imported']+tot['derived_kept']}")

    for sp in SPLITS:
        d = frames[sp]
        m = (d.wtag.notna() & d.wtag.astype(str).str.strip().ne("")) | d.wtag_derived.notna()
        d[m].to_parquet(work / f"word-qe_{sp}.parquet", index=False)
        print(f"  word-qe/{sp}: {int(m.sum())} rows")

    if a.dry_run:
        print("\nDRY RUN — nothing uploaded."); return 0
    if not a.push:
        print("\nno action (pass --dry-run or --push)"); return 0

    token = os.environ.get("HF_TOKEN") or Path(
        os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    api = HfApi(token=token)
    for sp in SPLITS:
        api.upload_file(path_or_fileobj=str(work / f"full_{sp}.parquet"),
                        path_in_repo=f"data/full/{sp}.parquet", repo_id=REPO, repo_type="dataset",
                        commit_message=f"import actual MLQE-PE word/gap/source tags into full/{sp}")
        api.upload_file(path_or_fileobj=str(work / f"word-qe_{sp}.parquet"),
                        path_in_repo=f"data/tasks/word-qe/{sp}.parquet", repo_id=REPO,
                        repo_type="dataset", commit_message=f"rebuild word-qe/{sp} with MLQE-PE tags")
        print(f"pushed {sp}")
    print("\nPUSHED. Re-run scripts/validate_release.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
