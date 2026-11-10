#!/usr/bin/env python3
"""
reconfirm_domain_classifier.py — re-confirm the domain-classifier accuracy/F1 that the paper cites
(§benchmark: 0.824 acc / 0.786 macro-F1, 5-fold CV; LLM baseline 0.618/0.595) on the *finalised*
public release (surrey-nlp/IndicQE-APE, post en-ml + consolidation).

It reuses the production pipeline from normalize_domain.build_pipeline (word⊕char TF-IDF → LR) and
runs proper StratifiedKFold(5) over the annotated rows' English side → canonical domain, exactly the
quantity the paper reports. Prints per-fold accuracy/macro-F1 + mean±std, and echoes the LLM baseline
from results/analysis/domain_labelling_report.md for the comparison.

Usage:
  python3 reconfirm_domain_classifier.py                 # read the live Hub release
  python3 reconfirm_domain_classifier.py --data DIR      # read a local snapshot dir instead
  python3 reconfirm_domain_classifier.py --dedup none    # keep all annotated rows (default dedups
                                                          # identical English texts to avoid leakage)
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from normalize_domain import build_pipeline, english_side, _norm_one, CANONICAL_DOMAINS  # noqa: E402

REPO = "surrey-nlp/IndicQE-APE"


def load_all(source: str | None) -> pd.DataFrame:
    """Load the 'all' config (every instance) from a local snapshot dir or the Hub."""
    if source and os.path.isdir(source):
        files = (glob.glob(os.path.join(source, "data", "all", "*.parquet"))
                 or glob.glob(os.path.join(source, "**", "all", "*.parquet"), recursive=True))
    else:
        from huggingface_hub import HfApi, hf_hub_download
        names = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                 if f.startswith("data/all/") and f.endswith(".parquet")]
        files = [hf_hub_download(REPO, f, repo_type="dataset") for f in names]
    if not files:
        sys.exit("no 'all' parquet files found (check --data or Hub access).")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def five_fold(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe = build_pipeline()
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        a = accuracy_score(y[te], pred)
        f = f1_score(y[te], pred, average="macro", zero_division=0)
        accs.append(a); f1s.append(f)
        print(f"  fold {i}: acc={a:.3f}  macro-F1={f:.3f}", flush=True)
    return np.array(accs), np.array(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local snapshot dir (default: live Hub release)")
    ap.add_argument("--dedup", choices=["eng", "none"], default="eng",
                    help="'eng' (default) dedups identical English texts before CV; 'none' keeps all")
    ap.add_argument("--out", default=os.path.join(HERE, "domain_reconfirm.json"))
    args = ap.parse_args()

    df = load_all(args.data)
    canon = (df["domain"] if "domain" in df.columns else pd.Series([None] * len(df))).map(_norm_one)
    eng = pd.Series(english_side(df), index=df.index).fillna("").astype(str)

    # Prefer the shipped provenance flag; fall back to "has a canonical label".
    if "domain_provenance" in df.columns:
        annotated = df["domain_provenance"].astype(str).eq("annotated") & canon.notna() & (eng.str.len() > 0)
    else:
        annotated = canon.notna() & (eng.str.len() > 0)

    sub = pd.DataFrame({"eng": eng[annotated].values, "y": canon[annotated].values})
    n_raw = len(sub)
    if args.dedup == "eng":
        sub = sub.drop_duplicates("eng", keep="first")

    # Drop classes with <5 members so StratifiedKFold(5) is valid.
    vc = sub["y"].value_counts()
    keep = vc[vc >= 5].index
    sub = sub[sub["y"].isin(keep)]

    print(f"[data] annotated rows: {n_raw:,}  |  after dedup({args.dedup}): {len(sub):,}  "
          f"|  classes: {sorted(sub['y'].unique())}")
    print(f"[data] class counts: {sub['y'].value_counts().to_dict()}")
    print(f"[cv] production pipeline (word⊕char TF-IDF → LR C=8), StratifiedKFold(5):")

    accs, f1s = five_fold(sub["eng"].values, sub["y"].values, k=5, seed=0)
    res = {
        "release": REPO, "n_annotated_raw": n_raw, "n_used": int(len(sub)),
        "dedup": args.dedup, "k": 5,
        "accuracy_mean": round(float(accs.mean()), 3), "accuracy_std": round(float(accs.std()), 3),
        "macro_f1_mean": round(float(f1s.mean()), 3), "macro_f1_std": round(float(f1s.std()), 3),
        "acc_folds": [round(float(x), 3) for x in accs],
        "f1_folds": [round(float(x), 3) for x in f1s],
        "llm_baseline": {"model": "gpt-5.4-nano (convention+6-shot)", "accuracy": 0.618, "macro_f1": 0.595},
        "paper_cited": {"accuracy": 0.824, "accuracy_std": 0.003, "macro_f1": 0.786, "macro_f1_std": 0.002},
    }
    print(f"\n[RESULT] accuracy = {res['accuracy_mean']:.3f} ± {res['accuracy_std']:.3f}  |  "
          f"macro-F1 = {res['macro_f1_mean']:.3f} ± {res['macro_f1_std']:.3f}")
    print(f"[paper ] accuracy = 0.824 ± 0.003  |  macro-F1 = 0.786 ± 0.002   (LLM baseline 0.618/0.595)")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
