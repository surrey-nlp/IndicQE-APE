#!/usr/bin/env python3
"""integrate_word_tags.py — push derived word-level tags into the HF release.

Follows the gated pattern used for the DeepSeek en-ml and en-hi MQM integrations:
join on `uid`, verify byte-identity of every protected column, assert the uid bijection, never
overwrite an existing gold value, and re-gate after the push.

What changes:
  * `data/full/{split}.parquet` gains `wtag_derived`, `mt_tokens_derived`, `wtag_provenance`.
      - `wtag_provenance` = "upstream" where the shipped `wtag` is non-null,
        "derived-ter-indicnlp" where we filled it, "" otherwise.
      - the existing `wtag` / `mt_tokens` columns are NOT modified.
  * `data/tasks/word-qe/{split}.parquet` is rebuilt as the union of rows carrying EITHER an
    upstream or a derived tag, so the task config reflects the expanded coverage.

Usage:  python3 scripts/integrate_word_tags.py --dry-run
        python3 scripts/integrate_word_tags.py --push
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")
DERIVED = Path("dataset/derived_word_tags.parquet")
PROTECTED = ["uid", "source", "mt", "post_edit", "language_pair", "da_mean", "wtag",
             "mt_tokens", "mqm_score", "cadence_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--work", default="dataset/wordqe_build")
    a = ap.parse_args()

    dv = pd.read_parquet(DERIVED)
    print(f"derived tags: {len(dv)} rows, {dv.uid.nunique()} unique uids")
    assert dv.uid.is_unique, "GATE FAIL: derived uids not unique"
    dmap = dv.set_index("uid")

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    totals = {"filled": 0, "upstream": 0, "rows": 0}
    frames = {}
    for sp in SPLITS:
        p = hf_hub_download(REPO, f"data/full/{sp}.parquet", repo_type="dataset")
        d = pd.read_parquet(p)
        before = {c: d[c].copy() for c in PROTECTED if c in d.columns}
        n0 = len(d)

        has_up = d.wtag.notna() & d.wtag.astype(str).str.strip().ne("")
        hit = d.uid.isin(dmap.index) & ~has_up
        d["wtag_derived"] = d.uid.map(dmap["wtag_derived"]).where(hit, None)
        d["mt_tokens_derived"] = d.uid.map(dmap["mt_tokens_derived"]).where(hit, None)
        d["wtag_provenance"] = ""
        d.loc[has_up, "wtag_provenance"] = "upstream"
        d.loc[hit, "wtag_provenance"] = "derived-ter-indicnlp"

        # ---- gates ----
        assert len(d) == n0, f"GATE FAIL: row count changed in {sp}"
        for c, col in before.items():
            assert d[c].astype(str).equals(col.astype(str)), f"GATE FAIL: {c} modified in {sp}"
        assert not (has_up & hit).any(), f"GATE FAIL: overwrote upstream tag in {sp}"
        assert d.uid.is_unique, f"GATE FAIL: uid not unique in {sp}"
        totals["rows"] += n0
        totals["filled"] += int(hit.sum()); totals["upstream"] += int(has_up.sum())
        frames[sp] = d
        d.to_parquet(work / f"full_{sp}.parquet", index=False)
        print(f"  {sp}: {n0} rows | upstream {int(has_up.sum())} | derived-filled {int(hit.sum())}")

    print(f"\nTOTAL rows {totals['rows']} | upstream tags {totals['upstream']} | "
          f"derived {totals['filled']} | word-QE coverage now "
          f"{totals['upstream']+totals['filled']}")

    # rebuild word-qe task config = rows with either tag
    for sp in SPLITS:
        d = frames[sp]
        m = (d.wtag.notna() & d.wtag.astype(str).str.strip().ne("")) | d.wtag_derived.notna()
        sub = d[m].copy()
        sub.to_parquet(work / f"word-qe_{sp}.parquet", index=False)
        print(f"  word-qe/{sp}: {len(sub)} rows")

    if a.dry_run:
        print("\nDRY RUN — nothing uploaded. Files staged in", work)
        return 0
    if not a.push:
        print("\nno action (pass --dry-run or --push)")
        return 0

    token = os.environ.get("HF_TOKEN") or Path(
        os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    api = HfApi(token=token)
    for sp in SPLITS:
        api.upload_file(path_or_fileobj=str(work / f"full_{sp}.parquet"),
                        path_in_repo=f"data/full/{sp}.parquet",
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"add derived word-QE tags to full/{sp}")
        api.upload_file(path_or_fileobj=str(work / f"word-qe_{sp}.parquet"),
                        path_in_repo=f"data/tasks/word-qe/{sp}.parquet",
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"rebuild word-qe/{sp} with derived tags")
        print(f"pushed {sp}")
    print("\nPUSHED. Re-run scripts/validate_release.py to re-gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
