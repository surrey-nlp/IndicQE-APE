#!/usr/bin/env python3
"""Audit every column of every HuggingFace config of surrey-nlp/IndicQE-APE.

Reports, per config and split: row count, column list, null rate, distinct count and a sample
value. Flags dead columns (100% null), schema drift between configs, and value-range problems.

    PYTHONPATH=. .venv/bin/python scripts/hf_schema_audit.py > hf-schema-audit.md
"""
import collections
import sys

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "surrey-nlp/IndicQE-APE"


def load(path):
    return pd.read_parquet(hf_hub_download(REPO, path, repo_type="dataset"))


def brief(v, n=48):
    s = str(v).replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s


def main():
    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    configs = collections.defaultdict(list)
    for f in sorted(files):
        parts = f.split("/")
        cfg = "/".join(parts[1:-1]) if len(parts) > 2 else parts[1].replace(".parquet", "")
        configs[cfg].append(f)

    print("# HuggingFace schema audit — surrey-nlp/IndicQE-APE\n")
    print(f"{len(files)} parquet files across {len(configs)} configs.\n")

    schemas = {}
    dead = collections.defaultdict(list)

    for cfg in sorted(configs):
        print(f"\n## `{cfg}`\n")
        frames = []
        for f in configs[cfg]:
            split = f.split("/")[-1].replace(".parquet", "")
            try:
                d = load(f)
            except Exception as e:
                print(f"- **{split}: FAILED to load** — {e}")
                continue
            frames.append((split, d))
            print(f"- `{split}`: {len(d):,} rows × {len(d.columns)} cols")
        if not frames:
            continue
        d = pd.concat([x for _, x in frames], ignore_index=True)
        schemas[cfg] = list(d.columns)

        print(f"\n| column | dtype | non-null | null % | distinct | sample |")
        print("|---|---|---:|---:|---:|---|")
        for c in d.columns:
            nn = int(d[c].notna().sum())
            pct = 100 * (1 - nn / len(d))
            try:
                nd = int(d[c].nunique(dropna=True))
            except Exception:
                nd = -1
            samp = ""
            nzs = d[c].dropna()
            if len(nzs):
                samp = brief(nzs.iloc[0])
            flag = " **DEAD**" if nn == 0 else ""
            if nn == 0:
                dead[cfg].append(c)
            print(f"| `{c}`{flag} | {d[c].dtype} | {nn:,} | {pct:.1f}% | "
                  f"{nd if nd >= 0 else '?'} | {samp} |")

        # value sanity on the known numeric fields
        for c in ("da_mean", "da_z_mean", "mqm_score", "n_annotators", "tqr"):
            if c in d.columns and d[c].notna().any():
                v = pd.to_numeric(d[c], errors="coerce").dropna()
                if len(v):
                    print(f"\n  - `{c}` range: min {v.min():.3f} · max {v.max():.3f} · "
                          f"mean {v.mean():.3f} · n={len(v):,}")

    # ---- cross-config schema drift ----
    print("\n\n## Schema drift across configs\n")
    allcols = collections.Counter()
    for cfg, cols in schemas.items():
        for c in cols:
            allcols[c] += 1
    n = len(schemas)
    universal = sorted(c for c, k in allcols.items() if k == n)
    partial = sorted((c, k) for c, k in allcols.items() if k < n)
    print(f"- configs audited: {n}")
    print(f"- columns present in EVERY config ({len(universal)}): {', '.join(f'`{c}`' for c in universal)}")
    print(f"\n- columns present in only SOME configs:\n")
    print("| column | in n configs | missing from |")
    print("|---|---:|---|")
    for c, k in partial:
        miss = sorted(cfg for cfg, cols in schemas.items() if c not in cols)
        print(f"| `{c}` | {k}/{n} | {', '.join(miss[:6])}{' …' if len(miss) > 6 else ''} |")

    print("\n\n## Dead columns (100% null within a config)\n")
    if not dead:
        print("None.")
    for cfg, cols in sorted(dead.items()):
        print(f"- `{cfg}`: {', '.join(f'`{c}`' for c in cols)}")


if __name__ == "__main__":
    main()
