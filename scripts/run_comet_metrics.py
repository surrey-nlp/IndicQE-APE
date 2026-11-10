#!/usr/bin/env python3
"""Trained-metric reference-free QE with COMET (card-verified API).

Metrics (no Tower):
  cometkiwi-da  Unbabel/wmt22-cometkiwi-da     (reference-free QE)
  cometkiwi-xl  Unbabel/wmt23-cometkiwi-da-xl  (reference-free QE)
  xcomet-xl     Unbabel/XCOMET-XL             (error-span + score; referenceless QE mode)

All gated (cc-by-nc-sa-4.0) — accept the licences + `huggingface-cli login`, and pre-download on a
login node (compute runs offline). Needs a GPU. `pip install "unbabel-comet>=2.2.0"`.

Two modes:
  QE  (default): scores the challenge test per pair, correlates vs da_mean, writes per-item records
                 in the analyse_challenge format (route/is_curated/gold_da_mean/pred_score) to
                 <out>/<metric>.<pair>.jsonl so metrics sit alongside the LLM judges.
  APE (--ape DIR): scores generated post-edits {src=source, mt=hyp_pe} with cometkiwi-da; writes
                   a per-file comet summary to <out>/<basename>.comet.json.

Resumable: skips any (metric,pair) whose output already has a _summary. One failed metric is logged
and skipped (the rest continue).

  python scripts/run_comet_metrics.py --out $IQE_OUT/qe_metrics --gpus 1
  python scripts/run_comet_metrics.py --ape $IQE_OUT/ape_open --out $IQE_OUT/ape_metrics --gpus 1
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
from indicqe_ape.data import DataLoader
from indicqe_ape.metrics import correlation

METRICS = {
    "cometkiwi-da": "Unbabel/wmt22-cometkiwi-da",
    "cometkiwi-xl": "Unbabel/wmt23-cometkiwi-da-xl",
    "xcomet-xl":    "Unbabel/XCOMET-XL",
}
PAIRS = ["enhi", "enmr", "enta", "ente", "engu", "enml", "eten", "neen", "sien"]
HP = {"enhi": "en-hi", "enmr": "en-mr", "enta": "en-ta", "ente": "en-te",
      "engu": "en-gu", "eten": "et-en", "neen": "ne-en", "sien": "si-en"}


def _has_summary(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return any('"_summary"' in ln for ln in f)
    except Exception:
        return False


def _load(repo: str, paths: dict):
    """Offline-robust load: prefer a pre-resolved local checkpoint path (written at prefetch to
    $IQE_CACHE/comet_paths.json) because download_model() can bypass the HF cache under
    HF_HUB_OFFLINE=1. Falls back to download_model() when online."""
    from comet import download_model, load_from_checkpoint
    ckpt = paths.get(repo)
    if not ckpt or not os.path.exists(ckpt):
        ckpt = download_model(repo)
    print(f"  loading {repo}  <-  {ckpt}", file=sys.stderr)
    return load_from_checkpoint(ckpt)


def _load_paths(arg: str | None) -> dict:
    pj = arg or os.path.join(os.environ.get("IQE_CACHE", "."), "comet_paths.json")
    try:
        return json.load(open(pj)) if os.path.exists(pj) else {}
    except Exception:
        return {}


def run_qe(out_dir: str, metrics: list[str], pairs: list[str], gpus: int, bs: int, paths: dict):
    os.makedirs(out_dir, exist_ok=True)
    dl = DataLoader()
    for mkey in metrics:
        repo = METRICS[mkey]
        todo = [p for p in pairs if not _has_summary(os.path.join(out_dir, f"{mkey}.{p}.jsonl"))]
        if not todo:
            print(f"[{mkey}] all pairs done — skip"); continue
        try:
            model = _load(repo, paths)
        except Exception as e:  # licence not accepted / not cached / OOM
            print(f"[{mkey}] LOAD FAILED ({type(e).__name__}: {e}) — skipping", file=sys.stderr); continue
        for p in pairs:
            outp = os.path.join(out_dir, f"{mkey}.{p}.jsonl")
            if _has_summary(outp):
                print(f"  {mkey}.{p} done"); continue
            ch = dl.load("challenge", pair=p)
            ch = ch[ch["da_mean"].notna()].reset_index(drop=True)
            if not len(ch):
                continue
            data = [{"src": str(r["source"]), "mt": str(r["mt"])} for _, r in ch.iterrows()]
            try:
                pred = model.predict(data, batch_size=bs, gpus=gpus, progress_bar=False)
                scores = list(pred.scores)
            except Exception as e:
                print(f"  {mkey}.{p} PREDICT FAILED ({type(e).__name__}: {e})", file=sys.stderr); continue
            with open(outp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"_meta": {"task": "qe", "metric": mkey, "model": repo, "pair": p}}) + "\n")
                preds, golds = [], []
                for (_, r), s in zip(ch.iterrows(), scores):
                    fh.write(json.dumps({"id": str(r["uid"]), "pair": str(r["language_pair"]), "task": "qe",
                                         "route": r.get("route", ""), "is_curated": bool(r.get("is_curated", False)),
                                         "gold_da_mean": float(r["da_mean"]), "pred_score": float(s)},
                                        ensure_ascii=False) + "\n")
                    preds.append(float(s)); golds.append(float(r["da_mean"]))
                c = correlation(preds, golds)
                fh.write(json.dumps({"_summary": {"metric": mkey, "pearson": c.pearson,
                                                  "spearman": c.spearman, "kendall": c.kendall, "n": c.n}}) + "\n")
            print(f"  {mkey}.{p}: pearson={c.pearson:.4f} spearman={c.spearman:.4f} n={c.n}")


def run_ape(ape_dir: str, out_dir: str, gpus: int, bs: int, paths: dict):
    """CometKiwi over generated post-edits: src vs hyp_pe (reference-free quality of the edit)."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ape_dir, "*.jsonl")))
    todo = [f for f in files if not os.path.exists(os.path.join(out_dir, os.path.basename(f).replace(".jsonl", ".comet.json")))]
    if not todo:
        print("APE comet: all done"); return
    try:
        model = _load(METRICS["cometkiwi-da"], paths)
    except Exception as e:
        print(f"APE comet LOAD FAILED ({e}) — skip", file=sys.stderr); return
    for f in files:
        outp = os.path.join(out_dir, os.path.basename(f).replace(".jsonl", ".comet.json"))
        if os.path.exists(outp):
            continue
        data = []
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "_meta" in r or "_summary" in r or not r.get("hyp_pe"):
                continue
            data.append({"src": str(r["source"]), "mt": str(r["hyp_pe"])})
        if not data:
            continue
        pred = model.predict(data, batch_size=bs, gpus=gpus, progress_bar=False)
        res = {"file": os.path.basename(f), "n": len(data),
               "cometkiwi_mean": float(np.mean(pred.scores)), "system_score": float(pred.system_score)}
        json.dump(res, open(outp, "w"))
        print(f"  {res['file']}: cometkiwi(post-edit) mean={res['cometkiwi_mean']:.4f} n={res['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="*", default=list(METRICS), choices=list(METRICS))
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--ape", help="score post-edits in this ape_* dir instead of running QE")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--paths", help="JSON of {repo: local_checkpoint} (default: $IQE_CACHE/comet_paths.json)")
    a = ap.parse_args()
    paths = _load_paths(a.paths)
    if a.ape:
        run_ape(a.ape, a.out, a.gpus, a.batch_size, paths)
    else:
        run_qe(a.out, a.metrics, a.pairs, a.gpus, a.batch_size, paths)


if __name__ == "__main__":
    main()
