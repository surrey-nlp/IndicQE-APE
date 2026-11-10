#!/usr/bin/env python3
from __future__ import annotations

"""
build_qe_data.py

Turn the IndicQE `qe-da` parquet splits into the CSV the BEA hybrid training
pipeline consumes, with a per-pair z-standardised regression target (decision
D2, see src/qe_data.py). Emits:

    data/qe_indic_<variant>.csv          (item_id, subset, source, mt,
                                          language_pair, da_mean, da_z_mean, target)
    data/qe_indic_<variant>.scaler.json  (per-pair {mean, std} for inverse xform)

It REPORTS the actual per-pair / per-split counts it wrote and never asserts a
hardcoded expected total, by design (documented counts are for
cross-checking, not for trusting blindly).

Variants:
  enindic     the 5 en->Indic pairs from data/tasks/qe-da (the main dev set)
  enml        gated en-ml only (headline pair; raw DA, no official z)
  joint_enml  enindic + en-ml (the joint headline model)
  xen         et/ne/si-en (MLQE) sanity/repro — needs the per-pair parquet

Examples:
  python scripts/build_qe_data.py --variant enindic
  python scripts/build_qe_data.py --variant enindic --per-pair 40   # smoke slice
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]          # comet_qe/
sys.path.insert(0, str(REPO_ROOT / "src"))
from qe_data import build_qe_dataframe, save_scaler       # noqa: E402

INDICQE_ROOT = REPO_ROOT.parents[1]                       # IndicQE-paper/
#: Default to the HF snapshot, which is the data authority. `release_build/hf_repo` is a stale
#: intermediate (no mqm / word-qe / explainable-qe configs, no en-ml pair) and building from it
#: silently produces an en-Indic-only frame. Refresh the snapshot with:
#:   python -c "from huggingface_hub import snapshot_download; snapshot_download(
#:     'surrey-nlp/IndicQE-APE', repo_type='dataset', local_dir='dataset/hf_snapshot',
#:     allow_patterns=['data/**'])"
DEFAULT_HF_REPO = Path(
    os.environ.get("INDICQEAPE_DATA") or INDICQE_ROOT / "dataset" / "hf_snapshot")

EN_INDIC = ["en-hi", "en-mr", "en-ta", "en-te", "en-gu"]
SPLITS = ["train", "validation", "test"]


def _read_task(hf_repo: Path, task: str) -> dict:
    base = hf_repo / "data" / "tasks" / task
    return {sp: pd.read_parquet(base / f"{sp}.parquet")
            for sp in SPLITS if (base / f"{sp}.parquet").exists()}


def _read_pair(hf_repo: Path, pair: str, gated: bool = False) -> dict:
    """Read a per-pair config.

    `gated` is retained for callers written before 16 Jul 2026, when en-ml lived under a
    gated/ prefix. Gating was removed and every pair now sits under data/pairs/, so the
    gated path is tried first and falls back rather than returning empty (which used to
    abort the joint_enml build).
    """
    for prefix in (("gated", "data") if gated else ("data",)):
        sub = hf_repo / prefix / "pairs" / pair
        frames = {sp: pd.read_parquet(sub / f"{sp}.parquet")
                  for sp in SPLITS if (sub / f"{sp}.parquet").exists()}
        if frames:
            return frames
    return {}


def load_frames(hf_repo: Path, variant: str):
    if variant == "enindic":
        return _read_task(hf_repo, "qe-da"), EN_INDIC
    if variant == "enml":
        frames = _read_pair(hf_repo, "en-ml", gated=True)
        if not frames:
            raise SystemExit(f"enml: no gated en-ml parquet under {hf_repo}/gated/pairs/en-ml")
        return frames, ["en-ml"]
    if variant == "joint_enml":
        # When en-ml was gated, qe-da omitted it and this variant concatenated the gated
        # per-pair config back in. Gating was removed on 16 Jul 2026 and en-ml is now IN
        # qe-da, so the old concat duplicated all 10,000 en-ml rows. Read qe-da and union
        # only what is genuinely absent, then assert uid uniqueness so a future upstream
        # change cannot reintroduce the duplication silently.
        base = _read_task(hf_repo, "qe-da")
        enml = _read_pair(hf_repo, "en-ml")
        merged = {}
        for sp in SPLITS:
            b, e = base.get(sp), enml.get(sp)
            if b is None and e is None:
                continue
            if b is None:
                merged[sp] = e
            elif e is None:
                merged[sp] = b
            else:
                new = e[~e["uid"].isin(set(b["uid"]))]
                if len(new):
                    print(f"joint_enml/{sp}: adding {len(new)} en-ml rows absent from qe-da")
                merged[sp] = pd.concat([b, new], ignore_index=True) if len(new) else b
            dup = int(merged[sp]["uid"].duplicated().sum())
            if dup:
                raise SystemExit(f"joint_enml/{sp}: {dup} duplicate uids after union")
        return merged, EN_INDIC + ["en-ml"]
    if variant == "xen":
        buckets = {sp: [] for sp in SPLITS}
        for pair in ["et-en", "ne-en", "si-en"]:
            pf = _read_pair(hf_repo, pair, gated=False)
            for sp in SPLITS:
                if sp in pf:
                    buckets[sp].append(pf[sp])
        merged = {sp: pd.concat(v, ignore_index=True) for sp, v in buckets.items() if v}
        if not merged:
            raise SystemExit("xen: no per-pair parquet under data/pairs/{et,ne,si}-en")
        return merged, ["et-en", "ne-en", "si-en"]
    raise SystemExit(f"unknown variant: {variant}")


def report(combined: pd.DataFrame) -> None:
    print("\n=== QE data built ===")
    print("rows per subset:", combined.groupby("subset").size().to_dict())
    piv = combined.groupby(["language_pair", "subset"]).size().unstack(fill_value=0)
    print("\nper-pair x split counts:")
    print(piv.to_string())
    tr = combined[combined.subset == "train"]
    stats = tr.groupby("language_pair")["target"].agg(["count", "mean", "std"]).round(4)
    print("\ntrain target per pair (expect mean~0, std~1):")
    print(stats.to_string())
    print(f"\nda_z_mean availability (all rows): {combined['da_z_mean'].notna().mean():.1%} non-null")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="enindic",
                    choices=["enindic", "enml", "joint_enml", "xen"])
    ap.add_argument("--hf-repo", default=str(DEFAULT_HF_REPO),
                    help="path to the hf_repo dir (or a cluster snapshot of it)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--pairs", nargs="*", default=None, help="override language pairs")
    ap.add_argument("--per-pair", type=int, default=None,
                    help="take the first N rows per pair per split (smoke slices)")
    ap.add_argument("--target-scale", default="perpair_z",
                    choices=["perpair_z", "raw", "minmax"],
                    help="regression target: per-pair z (default), raw da_mean 0-100, "
                         "or minmax da_mean/100. E5 ablation — writes a scale-suffixed file.")
    args = ap.parse_args()

    hf_repo = Path(args.hf_repo)
    frames, default_pairs = load_frames(hf_repo, args.variant)
    pairs = args.pairs or default_pairs

    if args.per_pair:
        frames = {
            sp: df.groupby("language_pair", group_keys=False).head(args.per_pair)
            for sp, df in frames.items()
        }

    combined, scaler = build_qe_dataframe(frames, pairs=pairs)

    # E5 target-scale ablation. build_qe_dataframe always fits `target` = per-pair z (the reference,
    # and what the scaler.json inverts). For the raw/minmax variants we overwrite `target` from
    # da_mean; the scaler.json is still written but is identity for these (no per-pair recentering).
    if args.target_scale == "raw":
        combined["target"] = combined["da_mean"]
    elif args.target_scale == "minmax":
        combined["target"] = combined["da_mean"] / 100.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_sfx = "" if args.target_scale == "perpair_z" else f"_{args.target_scale}"
    stem = (f"qe_indic_{args.variant}" + (f"_smoke{args.per_pair}" if args.per_pair else "")
            + scale_sfx)
    csv_path = out_dir / f"{stem}.csv"
    scaler_path = out_dir / f"{stem}.scaler.json"
    combined.to_csv(csv_path, index=False)
    save_scaler(scaler, scaler_path)

    report(combined)
    n_null_target = int(combined["target"].isna().sum())
    n_null_damean = int(combined["da_mean"].isna().sum())
    # A null target should only ever come from a null da_mean.
    assert n_null_target == n_null_damean, (
        f"target has {n_null_target} nulls but da_mean has {n_null_damean} — scaler gap"
    )
    print(f"\nwrote {csv_path}  ({len(combined)} rows)")
    print(f"scaler -> {scaler_path}")
    print(f"null da_mean/target: {n_null_damean}/{n_null_target}")


if __name__ == "__main__":
    main()
