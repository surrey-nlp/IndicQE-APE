#!/usr/bin/env python3
"""fill_inferable_nulls.py — fill only the nulls whose value is *known*, never guessed.

Each fill below is a fact already implied by another column, not an imputation. Anything requiring
a model, an assumption about human judgement, or data we never had is deliberately left null
(see results/analysis/null_audit.md).

1. `domain_confidence` = 1.0 where `domain_provenance == "annotated"`.
   A gold human label is certain by definition; the column previously held a classifier probability
   for inferred rows only, so a filter like `domain_confidence > 0.8` silently DROPPED every gold
   label. The annotated/inferred distinction is not lost: `domain_provenance` remains explicit.

2. `route` = "uncurated" where null.
   The X->en rows already carry the literal string "uncurated" for exactly this meaning, while
   en->Indic non-test rows carried null. One meaning, two encodings — harmonised.

3. `mqm_spans_complete` (NEW boolean, MQM rows only).
   `mqm_spans` is an empty list in two very different situations: 160 segments score 100 and truly
   have no errors, while 1,618 score < 100 — errors exist but their offsets were never recorded
   upstream (only 33.9% of the raw EAMT records carry offsets). Same encoding, opposite meaning.
   We cannot invent the spans, but we can say which empty list is which.

4. `has_*` flags on the 2,163 MQM-only rows, recomputed from the columns they summarise.

Deliberately NOT filled: `da_z_mean` (WMT z is per-annotator and we lack annotator identity — a
per-pair z is a different statistic and would belong in a differently named column), `domain_raw`
for inferred rows (there was no original string), `reference` (reserved by design), `mqm_spans`
themselves, and any pair's missing human annotation.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")
PROTECTED = ["uid", "source", "mt", "post_edit", "language_pair", "da_mean", "da_scores",
             "wtag", "wtag_derived", "mqm_score", "mqm_spans", "domain", "domain_provenance",
             "domain_raw", "cadence_id", "sources"]


def ne(s):
    return s.notna() & s.astype(str).str.strip().ne("")


def fix(d: pd.DataFrame, is_mqm_cfg: bool) -> dict:
    st = {}
    # 1. domain_confidence for gold labels
    ann = d["domain_provenance"].eq("annotated")
    tofill = ann & d["domain_confidence"].isna()
    st["domain_confidence"] = int(tofill.sum())
    d.loc[tofill, "domain_confidence"] = 1.0

    # 2. route harmonisation
    r_null = d["route"].isna() | d["route"].astype(str).str.strip().eq("")
    st["route"] = int(r_null.sum())
    d.loc[r_null, "route"] = "uncurated"

    # 3. mqm_spans_complete
    if "mqm_score" in d.columns:
        has_mqm = d["mqm_score"].notna()
        nspans = d["mqm_spans"].map(lambda x: 0 if x is None else len(x))
        complete = (d["mqm_score"] == 100) | (nspans > 0)
        d["mqm_spans_complete"] = pd.NA
        d.loc[has_mqm, "mqm_spans_complete"] = complete[has_mqm]
        st["mqm_spans_complete_set"] = int(has_mqm.sum())
        st["mqm_spans_incomplete"] = int((has_mqm & ~complete).sum())

    # 4. has_* flags
    src = {"has_da": lambda x: x["da_mean"].notna(),
           "has_pe": lambda x: ne(x["post_edit"]),
           "has_domain": lambda x: ne(x["domain"]),
           "has_mqm": lambda x: x["mqm_score"].notna(),
           "has_wtag": lambda x: ne(x["wtag"]) | (x["wtag_derived"].notna()
                                                  if "wtag_derived" in x else False),
           "has_tqr": lambda x: ne(x["tqr"]) if "tqr" in x else False,
           "has_synth_expl": lambda x: ne(x["synthetic_description"]) if "synthetic_description" in x else False}
    filled = 0
    for col, fn in src.items():
        if col not in d.columns:
            continue
        miss = d[col].isna()
        if int(miss.sum()) == 0:
            continue
        d.loc[miss, col] = fn(d)[miss]
        filled += int(miss.sum())
    st["has_flags"] = filled
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--work", default="dataset/nullfill_build")
    a = ap.parse_args()

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    targets = [("full", f"data/full/{{}}.parquet"), ("tasks/mqm", "data/tasks/mqm/{}.parquet")]
    staged = []
    for cfg, tmpl in targets:
        for sp in SPLITS:
            path = tmpl.format(sp)
            try:
                p = hf_hub_download(REPO, path, repo_type="dataset")
            except Exception:
                continue
            d = pd.read_parquet(p)
            n0 = len(d)
            before = {c: d[c].copy() for c in PROTECTED if c in d.columns}
            st = fix(d, cfg == "tasks/mqm")
            assert len(d) == n0, f"GATE FAIL rows changed {cfg}/{sp}"
            for c, col in before.items():
                assert d[c].astype(str).equals(col.astype(str)), f"GATE FAIL {c} modified {cfg}/{sp}"
            out = work / f"{cfg.replace('/', '_')}_{sp}.parquet"
            d.to_parquet(out, index=False)
            staged.append((path, out))
            print(f"  {cfg}/{sp}: {n0} rows | " +
                  " ".join(f"{k}={v}" for k, v in st.items() if v))

    if a.dry_run or not a.push:
        print("\nDRY RUN — nothing uploaded." if a.dry_run else "\nno action (--dry-run/--push)")
        return 0
    token = os.environ.get("HF_TOKEN") or Path(
        os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    api = HfApi(token=token)
    for path, out in staged:
        api.upload_file(path_or_fileobj=str(out), path_in_repo=path, repo_id=REPO,
                        repo_type="dataset",
                        commit_message=f"fill known-value nulls (domain_confidence, route, flags) in {path}")
        print(f"pushed {path}")
    print("\nPUSHED. Re-run scripts/validate_release.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
