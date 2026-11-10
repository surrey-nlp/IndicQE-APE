#!/usr/bin/env python3
"""Integrate the DeepSeek v4-Pro en-ml run into the explainable-qe config on HuggingFace.

Integration rules:
  * suffixed parallel columns, never overwrite
  * the six validation gates, run before the push and again after
  * the 363-row decision: ADD them (see below)

The 363 decision, and why it is a repair rather than a growth
-------------------------------------------------------------
`data/all` en-ml carries 5,000 synthetic descriptions; `explainable-qe` carries only
4,637. The 363 rows that the v4-Pro run found in `all` but not in `explainable-qe` are
exactly that gap: every one of them has a non-empty `synthetic_description`, a
`post_edit`, `has_synth_expl=True` and `explanation_model='deepseek'`, and the column
sets of the two configs are identical. They were dropped by the config build, not
withheld by design. Adding them takes explainable-qe 19,281 -> 19,644 and en-ml
9,637 -> 10,000.

Because this is a deliberate membership change, gate 2 is asserted against the NEW
expected count (19,644) and the old count is recorded, so the change can never happen
silently.

Usage
-----
    export HF_TOKEN=...                     # write access to surrey-nlp/IndicQE-APE
    python3 scripts/integrate_deepseek_v4pro.py --dry-run     # build + gate, no push
    python3 scripts/integrate_deepseek_v4pro.py --push        # build + gate + push + re-gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")
SRC_JSONL = Path("deepseek-enml/enml_deepseek_v4pro.jsonl")
WORK = Path("/tmp/iqe_v4pro")

# Counts that must hold, verified against
# the live repo on 2026-07-20.
N_NEW_ROWS = 10_000
N_XQ_BEFORE = 19_281
N_XQ_AFTER = 19_644          # 19,281 + 363, the deliberate membership change
N_363 = 363
N_ENML_BEFORE = 9_637
N_ENML_AFTER = 10_000
N_DESC_BEFORE = 14_281       # synthetic_description non-empty, all four pairs
N_DESC_AFTER = 14_644        # + the 363 restored rows, each of which carries one

# The Gemini parallel is 5,000 en-ml rows in `data/all`, not the 4,637 recorded in an earlier
# count ledger. That figure was itself an artefact of the 363-row drop:
# explainable-qe lost 363 rows wholesale, and with them 363 DeepSeek descriptions AND
# 363 Gemini descriptions. In `data/all` the described set and the Gemini set are the
# same 5,000 uids exactly. Restoring the 363 therefore also restores the Gemini column
# to its true size.
N_GEMINI_BEFORE = 4_637
N_GEMINI_AFTER = 5_000

# The four suffixed columns, mapped from the source file. Never overwrite an existing column.
COLUMN_MAP = {
    "synthetic_description_deepseek_v4pro": "synthetic_description",
    "error_categories_deepseek_v4pro": "error_categories",
    "post_edit_deepseek_v4pro": "deepseek_post_edit",
    "synth_desc_reused_v4pro": "used_existing_synth_desc",
}

PROTECTED = [
    "synthetic_description",
    "synthetic_description_gemini",
    "error_categories",
    "error_categories_gemini",
    "explanation_model",
    "post_edit",
]


def nonempty(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def load_remote(config: str, tag: str) -> dict[str, pd.DataFrame]:
    """Download a config's three splits and cache them locally under `tag`."""
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for sp in SPLITS:
        path = hf_hub_download(REPO, f"data/{config}/{sp}.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        df.to_parquet(WORK / f"{config.replace('/', '_')}_{sp}_{tag}.parquet")
        out[sp] = df
    return out


def load_source() -> pd.DataFrame:
    rows = [json.loads(line) for line in SRC_JSONL.open() if line.strip()]
    df = pd.DataFrame(rows)
    assert len(df) == N_NEW_ROWS, f"source file has {len(df)} rows, expected {N_NEW_ROWS}"
    assert df["uid"].nunique() == N_NEW_ROWS, "source file has duplicate uids"
    assert (df["language_pair"] == "en-ml").all(), "source file is not en-ml only"
    for col in COLUMN_MAP.values():
        assert nonempty(df[col]).all(), f"source column {col} has empty values"
    return df


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(xq: dict[str, pd.DataFrame], allc: dict[str, pd.DataFrame], src: pd.DataFrame):
    """Return the new explainable-qe splits, plus a provenance record."""
    xq_all = pd.concat([d.assign(_split=sp) for sp, d in xq.items()], ignore_index=True)
    all_all = pd.concat([d.assign(_split=sp) for sp, d in allc.items()], ignore_index=True)

    assert len(xq_all) == N_XQ_BEFORE, f"explainable-qe is {len(xq_all)}, expected {N_XQ_BEFORE}"

    xq_ml_uids = set(xq_all.loc[xq_all.language_pair == "en-ml", "uid"])
    src_uids = set(src["uid"])
    missing = sorted(src_uids - xq_ml_uids)
    assert len(missing) == N_363, f"found {len(missing)} rows outside explainable-qe, expected {N_363}"

    # --- the 363: pull their full records from data/all, same column set ---
    add = all_all[all_all["uid"].isin(missing)].copy()
    assert len(add) == N_363, f"data/all yielded {len(add)} of the {N_363}"
    assert nonempty(add["synthetic_description"]).all(), "a 363 row has no description"
    assert set(add.columns) == set(xq_all.columns), "column mismatch between all and explainable-qe"
    add = add[xq_all.columns]

    grown = pd.concat([xq_all, add], ignore_index=True)
    assert len(grown) == N_XQ_AFTER, f"grown config is {len(grown)}, expected {N_XQ_AFTER}"
    assert grown["uid"].nunique() == N_XQ_AFTER, "adding the 363 created duplicate uids"

    # --- the four suffixed columns ---
    for new_col in COLUMN_MAP:
        assert new_col not in grown.columns, f"{new_col} already exists; refusing to overwrite"

    side = src[["uid"] + list(COLUMN_MAP.values())].rename(
        columns={v: k for k, v in COLUMN_MAP.items()}
    )
    assert side["uid"].nunique() == len(side), "source side-table has duplicate uids"

    before_len = len(grown)
    merged = grown.merge(side, on="uid", how="left", validate="one_to_one")
    assert len(merged) == before_len, "merge changed the row count"

    # every en-ml row must now carry the v4-pro columns; no other pair may
    ml_mask = merged.language_pair == "en-ml"
    assert ml_mask.sum() == N_ENML_AFTER, f"en-ml is {ml_mask.sum()}, expected {N_ENML_AFTER}"
    for col in COLUMN_MAP:
        assert nonempty(merged.loc[ml_mask, col]).all(), f"{col} empty on some en-ml row"
        assert merged.loc[~ml_mask, col].isna().all(), f"{col} written onto a non-en-ml row"

    # cast the reuse flag to a real boolean rather than leaving it object-typed
    merged["synth_desc_reused_v4pro"] = merged["synth_desc_reused_v4pro"].astype("boolean")

    out = {sp: merged[merged._split == sp].drop(columns="_split").reset_index(drop=True)
           for sp in SPLITS}

    prov = {
        "rows_before": N_XQ_BEFORE,
        "rows_after": len(merged),
        "added_363": len(add),
        "enml_before": N_ENML_BEFORE,
        "enml_after": int(ml_mask.sum()),
        "new_columns": sorted(COLUMN_MAP),
        "v4pro_fresh_descriptions": int(
            (merged.loc[ml_mask, "synth_desc_reused_v4pro"] == False).sum()),  # noqa: E712
        "v4pro_reused_descriptions": int(
            (merged.loc[ml_mask, "synth_desc_reused_v4pro"] == True).sum()),  # noqa: E712
        "split_sizes": {sp: len(d) for sp, d in out.items()},
    }
    return out, merged, prov


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gates(before: pd.DataFrame, after: pd.DataFrame, src: pd.DataFrame, label: str) -> bool:
    """The six validation gates, run before the push and again after."""
    print(f"\n--- validation gates ({label}) ---")
    ok = True

    def check(n, desc, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] gate {n}: {desc}" + (f"  ({detail})" if detail else ""))

    # 1 — join is 1:1
    ml = after[after.language_pair == "en-ml"]
    joined = nonempty(ml["synthetic_description_deepseek_v4pro"]).sum()
    check(1, "join on uid is 1:1, 10,000 in / 10,000 matched, no duplicates",
          joined == N_NEW_ROWS and after["uid"].nunique() == len(after),
          f"matched={joined}, unique_uids={after['uid'].nunique()}/{len(after)}")

    # 2 — row count, deliberately changed
    check(2, f"row count is the DELIBERATE {N_XQ_AFTER} (was {N_XQ_BEFORE}, +{N_363})",
          len(after) == N_XQ_AFTER, f"actual={len(after)}")

    SENTINEL = "\0<null>"

    def n_differ(x: pd.Series, y: pd.Series) -> int:
        return int((x.fillna(SENTINEL) != y.fillna(SENTINEL)).sum())

    # 3 — post_edit byte-identical on the pre-existing rows
    b = before.set_index("uid")["post_edit"]
    a = after.set_index("uid")["post_edit"].reindex(b.index)
    check(3, "post_edit byte-identical on all pre-existing rows",
          a.equals(b), f"{n_differ(a, b)} differ")

    # 4 — the two description columns byte-identical on the pre-existing rows
    same = True
    detail = []
    for col in ("synthetic_description", "synthetic_description_gemini"):
        bb = before.set_index("uid")[col]
        aa = after.set_index("uid")[col].reindex(bb.index)
        if not aa.equals(bb):
            same = False
            detail.append(f"{col}: {n_differ(aa, bb)} differ")
    check(4, "synthetic_description and *_gemini byte-identical on pre-existing rows",
          same, "; ".join(detail))

    # 5 — the other three pairs completely untouched
    untouched = True
    detail = []
    for pair in ("en-hi", "en-mr", "en-ta"):
        bb = before[before.language_pair == pair].sort_values("uid").reset_index(drop=True)
        aa = after[after.language_pair == pair].sort_values("uid").reset_index(drop=True)
        aa = aa[bb.columns]
        if not bb.equals(aa):
            untouched = False
            detail.append(f"{pair}: {len(bb)} vs {len(aa)}")
    check(5, "en-hi / en-mr / en-ta completely untouched", untouched, "; ".join(detail))

    # 6 — protected columns not overwritten, and counts land on the post-repair truth.
    # Both rise by exactly 363 because the restored rows carry one of each; the Gemini
    # set and the DeepSeek-described set are the same 5,000 uids in `data/all`.
    n_desc = int(nonempty(after["synthetic_description"]).sum())
    n_gem = int(nonempty(after["synthetic_description_gemini"]).sum())
    counts_ok = n_desc == N_DESC_AFTER and n_gem == N_GEMINI_AFTER
    check(6,
          f"protected columns intact (descriptions {N_DESC_BEFORE}+{N_363}={N_DESC_AFTER}, "
          f"gemini {N_GEMINI_BEFORE}+{N_363}={N_GEMINI_AFTER})",
          counts_ok, f"desc={n_desc}, gemini={n_gem}")

    print(f"--- {'ALL GATES PASS' if ok else 'GATES FAILED'} ({label}) ---")
    return ok


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="push to the Hub after gates pass")
    ap.add_argument("--dry-run", action="store_true", help="build and gate only (default)")
    args = ap.parse_args()

    print("downloading current state from the Hub ...")
    xq = load_remote("tasks/explainable-qe", "ORIG")
    allc = load_remote("all", "ORIG")
    src = load_source()

    before = pd.concat(xq.values(), ignore_index=True)
    out, after, prov = build(xq, allc, src)

    print("\nprovenance:")
    for k, v in prov.items():
        print(f"  {k}: {v}")

    if not gates(before, after, src, "pre-push, local build"):
        print("\nABORT — gates failed before any write.")
        return 1

    WORK.mkdir(parents=True, exist_ok=True)
    for sp, df in out.items():
        df.to_parquet(WORK / f"explainable-qe_{sp}_NEW.parquet", index=False)
    (WORK / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nbuilt artefacts in {WORK}")

    if not args.push:
        print("\ndry run — nothing pushed. Re-run with --push (and HF_TOKEN set) to publish.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nHF_TOKEN is not set — cannot push.")
        return 1

    api = HfApi(token=token)
    for sp in SPLITS:
        api.upload_file(
            path_or_fileobj=str(WORK / f"explainable-qe_{sp}_NEW.parquet"),
            path_in_repo=f"data/tasks/explainable-qe/{sp}.parquet",
            repo_id=REPO,
            repo_type="dataset",
            commit_message=(
                "explainable-qe: add DeepSeek v4-Pro en-ml parallel columns; "
                "restore the 363 described en-ml rows dropped by the config build "
                "(19,281 -> 19,644)"
            ),
        )
        print(f"  pushed {sp}")

    print("\nre-downloading to verify against the live repo ...")
    xq2 = load_remote("tasks/explainable-qe", "AFTER")
    live = pd.concat(xq2.values(), ignore_index=True)
    if not gates(before, live, src, "post-push, live repo"):
        print("\nPOST-PUSH GATES FAILED — investigate before relying on the release.")
        return 1
    print("\nintegration complete and verified against the live repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
