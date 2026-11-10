#!/usr/bin/env python3
"""run_zeroshot_comet.py — Strategy-Z zero-shot COMET baselines for IndicQE-APE.

Runs pretrained COMET QE metrics reference-free on a task config's segments, writes per-instance
scores, and reports per-pair correlation with the human label. This is the Z row of the money
table (06_ablation_matrix.md) AND the X->en MLQE repro sanity check that confirms the pipeline is
correct before we trust the Indic numbers.

Models (reference-free QE, {src, mt} only):
    Unbabel/wmt22-cometkiwi-da       565M InfoXLM   — open-ish (gated)
    Unbabel/wmt23-cometkiwi-da-xl    3.5B XLM-R-XL  — gated (needs 24GB+; A5000 fp16 ok for predict)
    Unbabel/XCOMET-XL                3.5B           — QE mode ({src, mt})

Label:
    qe-da / per-pair configs -> `da_mean`   (higher = better)
    mqm config               -> `mqm_score` (and `da_mean` on the paired rows)

Runs on a single 24 GB GPU (e.g. an A5000). GPU predict only; correlations are CPU.

    python scripts/run_zeroshot_comet.py --config qe-da --split test \
        --models Unbabel/wmt22-cometkiwi-da Unbabel/wmt23-cometkiwi-da-xl Unbabel/XCOMET-XL \
        --gpus 1 --batch-size 16 --out-dir results/zeroshot
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from huggingface_hub import hf_hub_download

REPO = "surrey-nlp/IndicQE-APE"
DEFAULT_MODELS = ["Unbabel/wmt22-cometkiwi-da",
                  "Unbabel/wmt23-cometkiwi-da-xl",
                  "Unbabel/XCOMET-XL"]


def load_config_split(config: str, split: str) -> pd.DataFrame:
    """Load one split of a task/pair config from the live HF release."""
    fn = f"data/tasks/{config}/{split}.parquet" if "-" not in config or config in (
        "qe-da", "ape", "word-qe", "explainable-qe", "mqm", "all", "full", "challenge"
    ) else f"data/pairs/{config}/{split}.parquet"
    p = hf_hub_download(REPO, fn, repo_type="dataset")
    return pd.read_parquet(p)


def score_comet(model_id: str, data: list[dict], gpus: int, batch_size: int):
    from comet import download_model, load_from_checkpoint
    token = None
    tf = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(tf):
        token = open(tf).read().strip()
    ckpt = download_model(model_id, saving_directory=None) if token is None else \
        download_model(model_id)
    model = load_from_checkpoint(ckpt)
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return list(out["scores"]) if isinstance(out, dict) else list(out.scores)


def corr(x, y):
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return None
    return {"n": int(m.sum()), "pearson": round(float(pearsonr(x[m], y[m])[0]), 4),
            "spearman": round(float(spearmanr(x[m], y[m])[0]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qe-da")
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--label", default=None, help="override label col (default: mqm_score for mqm, else da_mean)")
    ap.add_argument("--out-dir", default="results/zeroshot")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N rows")
    a = ap.parse_args()

    label = a.label or ("mqm_score" if a.config == "mqm" else "da_mean")
    print(f"loading {a.config}/{a.split} from HF ...")
    df = load_config_split(a.config, a.split)
    if a.limit:
        df = df.head(a.limit)
    df = df[df[label].notna()].reset_index(drop=True)
    print(f"  {len(df)} rows with {label}; pairs: {sorted(df.language_pair.unique())}")

    data = [{"src": s, "mt": m} for s, m in zip(df["source"], df["mt"])]
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for model_id in a.models:
        tag = model_id.split("/")[-1]
        print(f"\n=== {tag} ===")
        try:
            scores = score_comet(model_id, data, a.gpus, a.batch_size)
        except Exception as e:
            print(f"  FAILED: {e}")
            summary.append({"model": tag, "error": str(e)})
            continue
        df["_pred"] = scores
        # per-instance jsonl
        jp = out_dir / f"{a.config}-{a.split}-{tag}.jsonl"
        with open(jp, "w") as fh:
            for uid, pv in zip(df["uid"], scores):
                fh.write(json.dumps({"uid": uid, "pred_score": float(pv)}) + "\n")
        # correlations: overall + per pair (+ mqm_score when config==mqm and da present)
        overall = corr(df["_pred"], df[label])
        per_pair = {p: corr(g["_pred"], g[label]) for p, g in df.groupby("language_pair")}
        row = {"model": tag, "label": label, "overall": overall, "per_pair": per_pair}
        if a.config == "mqm" and "da_mean" in df:
            row["vs_da_mean"] = corr(df["_pred"], df["da_mean"])
        summary.append(row)
        print(f"  overall {overall}")
        for p, c in per_pair.items():
            print(f"    {p}: {c}")
        print(f"  scores -> {jp}")

    sp = out_dir / f"zeroshot-summary-{a.config}-{a.split}.json"
    json.dump(summary, open(sp, "w"), indent=2)
    print(f"\nsummary -> {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
