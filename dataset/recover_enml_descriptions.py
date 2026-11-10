#!/usr/bin/env python3
"""
recover_enml_descriptions.py — enrich the PUBLISHED surrey-nlp/IndicQE-APE en-ml data with the
DeepSeek + Gemini synthetic descriptions from the ALOPE-RL source, surgically (no full rebuild,
so nothing else in the release changes).

Why: the build carried only 2,960 of the 5,000 ALOPE DeepSeek descriptions and none of the Gemini
ones. This recovers the full ALOPE coverage:
  * synthetic_description / error_categories  (DeepSeek)  -> filled for all ~5,000 ALOPE en-ml rows
  * synthetic_description_gemini / error_categories_gemini (NEW) -> the 5,000 Gemini variants
The other ~5,000 en-ml rows (tech/sports/finance, not in ALOPE) are left untouched here — those get
DeepSeek-v4-pro descriptions later via generate_enml_deepseek.py, and NO Gemini variant.

Match key: HF `original_id`  <->  ALOPE `ID`  (verified 4,791 unique; falls back to `source`).

Usage:
  python3 recover_enml_descriptions.py \
      --alope ../ALOPE-RL-Dataset \
      --out   ./release_build/hf_enriched            # local enriched snapshot (then push)
  # dry run (no download/write): add --dry-run
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd

REPO = "surrey-nlp/IndicQE-APE"
DESC_COL = "Description of the translation"
EC_COL = "Identified error categories"


def _present(v):
    return v is not None and str(v).strip() not in ("", "None", "nan", "[]", "null")


def load_alope(alope_dir: str) -> pd.DataFrame:
    """Return one row per ALOPE en-ml SOURCE text with DeepSeek + Gemini description/categories.
    Keyed on `source` (the English text) because it matches all 5,000 HF rows exactly, whereas
    `ID` collides for ~209 rows."""
    sub = os.path.join(alope_dir, "En-Ml_data_with_TQR_and_Synthetic_explanations")
    def read(model):
        parts = [pd.read_excel(os.path.join(sub, f"EN-ML_{sp}_with_{model}_synthetic_data.xlsx"))
                 for sp in ("Train", "Test")]
        d = pd.concat(parts, ignore_index=True)
        d["Source"] = d["Source"].astype(str).str.strip()
        d = d[["Source", DESC_COL, EC_COL]].rename(
            columns={DESC_COL: f"desc_{model.lower()}", EC_COL: f"ec_{model.lower()}"})
        return d.drop_duplicates("Source", keep="first")
    ds = read("Deepseek")
    gm = read("Gemini")[["Source", "desc_gemini", "ec_gemini"]]
    m = ds.merge(gm, on="Source", how="outer")
    print(f"[alope] unique sources: {len(m)} | deepseek desc: {int(m['desc_deepseek'].map(_present).sum())} "
          f"| gemini desc: {int(m['desc_gemini'].map(_present).sum())}", file=sys.stderr)
    return m


def enrich_frame(df: pd.DataFrame, amap: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Enrich the en-ml rows of one parquet in place; return (df, n_enml_enriched)."""
    if "language_pair" not in df.columns:
        return df, 0
    is_ml = df["language_pair"].astype(str) == "en-ml"
    if not is_ml.any():
        return df, 0
    # ensure the two new columns exist (null elsewhere) so the config's schema stays consistent
    for c in ("synthetic_description_gemini", "error_categories_gemini"):
        if c not in df.columns:
            df[c] = None
    if "source" not in df.columns:
        return df, 0
    key = df.loc[is_ml, "source"].astype(str).str.strip()
    lut = amap.set_index("Source")
    hit = key.map(lambda k: k in lut.index)
    idx = df.index[is_ml][hit.values]
    for i in idx:
        k = str(df.at[i, "source"]).strip()
        row = lut.loc[k]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # DeepSeek -> primary columns (fill/refresh so all ALOPE rows are covered)
        if _present(row.get("desc_deepseek")):
            df.at[i, "synthetic_description"] = row["desc_deepseek"]
            if "error_categories" in df.columns and _present(row.get("ec_deepseek")):
                df.at[i, "error_categories"] = row["ec_deepseek"]
            if "explanation_model" in df.columns:
                df.at[i, "explanation_model"] = "deepseek"
        # Gemini -> new columns
        if _present(row.get("desc_gemini")):
            df.at[i, "synthetic_description_gemini"] = row["desc_gemini"]
            df.at[i, "error_categories_gemini"] = row.get("ec_gemini")
    if "has_synth_expl" in df.columns:
        df.loc[is_ml, "has_synth_expl"] = df.loc[is_ml, "synthetic_description"].map(_present)
    return df, int(len(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alope", required=True, help="path to ALOPE-RL-Dataset clone")
    ap.add_argument("--out", default="./release_build/hf_enriched")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    amap = load_alope(args.alope)
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")]
    print(f"[repo] {len(files)} parquet files", file=sys.stderr)

    total_ml = 0
    touched = []
    for f in files:
        p = hf_hub_download(args.repo, f, repo_type="dataset")
        df = pd.read_parquet(p)
        df2, n = enrich_frame(df, amap)
        if n:
            total_ml += n
            touched.append(f)
            outp = os.path.join(args.out, f)
            if not args.dry_run:
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                df2.to_parquet(outp, index=False)
            ds = int(df2.loc[df2["language_pair"].astype(str) == "en-ml", "synthetic_description"].map(_present).sum())
            gm = int(df2.loc[df2["language_pair"].astype(str) == "en-ml", "synthetic_description_gemini"].map(_present).sum())
            print(f"  {f:45s} en-ml={int((df2['language_pair'].astype(str)=='en-ml').sum()):5d} "
                  f"deepseek={ds:5d} gemini={gm:5d} matched={n}")
    print(f"\n[done] enriched {len(touched)} parquet files ({total_ml} en-ml row-matches). "
          f"{'DRY-RUN (nothing written)' if args.dry_run else 'written to '+args.out}")
    print("Touched configs:", sorted(set(t.split('/')[1] if t.startswith('data/') else t for t in touched)))


if __name__ == "__main__":
    main()
