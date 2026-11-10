#!/usr/bin/env python3
"""Build the unified surrey-nlp/IndicQE-APE release from all source datasets.

Goal: one place for everything — common and uncommon — with precise instance
matching, per-instance provenance ("cadence"), and a configurable HF layout.

Sources (PI decisions, 7 Jun 2026):
  C  surrey-nlp/Low-resource-QE-DA-dataset   QE-DA backbone, 8 pairs, official splits
  A  sourabhd13/ape-qe-combined-data         post-edits (ape_qe_combined ⊂ C; ape_only extra)
  B  surrey-nlp/{Health,Tourism,General,Legal}-QE   domain-tagged DA, en→5 Indic
  R  surrey-nlp/ALOPE-RL-Dataset            word-QE (WTAG) + explainable-QE (TQR,
        synthetic descriptions/categories) + En-Ml (NEW, Anuvaad). en-ta/mr/hi here are
        the SAME WMT instances as C — WTAG/explanations MUST attach to the matching C row.
        Gated until paper 2602.08600 is accepted; ingested from a local clone meanwhile.

Precise matching: a deterministic content hash over NFC-normalised, whitespace-collapsed
(source ||| mt) is the join key (`uid`). The same (src, mt) from any source maps to one
uid, so annotations from different sources merge onto one instance. Human-readable
`cadence_id` ({SRC}-{PAIR}-{seq}) is assigned in stable uid order for citation.

Outputs: data/unified/*.parquet (per pair) + coverage_matrix.csv + a build report.
RL fields are written to a SEPARATE gated partition (data/gated_rl/) so the public
release can omit them until acceptance.

Usage:
  python3 build_unified.py --out-dir ./data --rl-local /path/to/ALOPE-RL-Dataset
  (omit --rl-local to build the public union only; RL columns stay null.)
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

PAIRS = ["enhi", "enmr", "enta", "engu", "ente", "eten", "neen", "sien", "enml"]
PAIR_DISPLAY = {"enhi": "en-hi", "enmr": "en-mr", "enta": "en-ta", "engu": "en-gu",
                "ente": "en-te", "eten": "et-en", "neen": "ne-en", "sien": "si-en",
                "enml": "en-ml"}
SRC_LANG = {"enhi": "en", "enmr": "en", "enta": "en", "engu": "en", "ente": "en",
            "enml": "en", "eten": "et", "neen": "ne", "sien": "si"}
TGT_LANG = {"enhi": "hi", "enmr": "mr", "enta": "ta", "engu": "gu", "ente": "te",
            "enml": "ml", "eten": "en", "neen": "en", "sien": "en"}
# provenance precedence for canonical da_mean when sources disagree
SRC_PRIORITY = {"wmt-qe-da": 0, "wmt20-mlqe": 0, "domain-qe": 1, "ape-combined": 2, "alope-rl": 3}

# Known source data-entry corrections: uid -> corrected annotator score list. da_mean is
# then recomputed from these. The Legal-QE en-te row had scores [7580,75,7580] (a lost-comma
# typo; the repo is uniformly 3-annotator), corrected to [75,75,80] per PI (11 Jun 2026).
_SCORE_PATCHES = {"b6100ed5acbe69f2": [75.0, 75.0, 80.0]}

_ws = re.compile(r"\s+")
_QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'",
              "–": "-", "—": "-"}


# --------------------------------------------------------------------------- #
#  Precise instance matching
# --------------------------------------------------------------------------- #
def norm_match(s):
    """Canonical form for matching ONLY (stored fields keep their original text).

    Verified on RL↔C (7 Jun 2026): the RL MT carries an appended <EOS> token and
    tokeniser spacing around punctuation, so a light norm matched 0%. Stripping
    <EOS>, NFKC, lowercasing, normalising quotes/dashes and removing ALL whitespace
    lifts RL→C matching to ~90-95% while only collapsing C's own identical-DA dupes."""
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("<EOS>", "").replace("<eos>", "")
    for a, b in _QUOTE_MAP.items():
        s = s.replace(a, b)
    return _ws.sub("", s)


def uid_of(src, mt):
    key = norm_match(src) + "|" + norm_match(mt)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _score_list(v):
    """Robustly coerce a source 'scores' cell to list[float] | None.

    Handles real lists/arrays, JSON strings ('[85, 90, 74]'), and the
    space/comma-separated strings the domain-QE (B) repos use ('70 68 69').
    Earlier `list(v)` on a *string* split it into characters — the bug this
    fixes (it corrupted da_scores + n_annotators for ~12.6k domain-qe rows)."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:           # scalar NaN
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        seq = list(v)
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            j = json.loads(s)
            seq = j if isinstance(j, (list, tuple)) else [j]
        except Exception:
            seq = re.split(r"[,\s]+", s.strip("[]() "))
    else:
        seq = [v]
    out = []
    for x in seq:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out or None


# --------------------------------------------------------------------------- #
#  Source loaders -> list of unified-row dicts
# --------------------------------------------------------------------------- #
def _row(source_dataset, pair, src, mt, original_id, split, **extra):
    r = {"uid": uid_of(src, mt), "source_dataset": source_dataset,
         "language_pair": PAIR_DISPLAY.get(pair, pair),
         "src_lang": SRC_LANG.get(pair), "tgt_lang": TGT_LANG.get(pair),
         "source": src, "mt": mt, "original_id": str(original_id), "split": split,
         "domain": None, "da_scores": None, "da_mean": np.nan,
         "da_z_scores": None, "da_z_mean": np.nan,
         "n_annotators": np.nan, "post_edit": None, "reference": None,
         "wtag": None, "mt_tokens": None, "tqr": None,
         "synthetic_description": None, "error_categories": None, "explanation_model": None,
         "mqm_spans": None, "mqm_score": np.nan}
    r.update(extra)
    return r


def load_C():
    """Surrey low-resource QE-DA = the En->Indic QE-DA backbone. X->En (et/ne/si-en)
    is NOT taken from here: C's X->En `translation` field is corrupted (annotator scores
    appended after a tab, leaving `scores` empty / `mean` null for 274/1868/843 rows).
    Those pairs come from the authoritative wmt20_mlqe_task1 via load_W() instead."""
    from huggingface_hub import hf_hub_download
    repo = "surrey-nlp/Low-resource-QE-DA-dataset"
    rows = []
    for pair in [p for p in PAIRS if p not in ("enml", "eten", "neen", "sien")]:
        for split in ["train", "dev", "test"]:
            try:
                df = pd.read_parquet(hf_hub_download(repo, f"parquet/{pair}/{split}.parquet", repo_type="dataset"))
            except Exception:
                continue
            for _, x in df.iterrows():
                sc = _score_list(x.get("scores"))
                rows.append(_row("wmt-qe-da", pair, x["original"], x["translation"],
                                 x.get("index", ""), split, da_scores=sc,
                                 da_mean=x.get("mean", np.nan),
                                 da_z_scores=_score_list(x.get("z_scores")),
                                 da_z_mean=x.get("z_mean", np.nan),
                                 n_annotators=(len(sc) if sc else np.nan)))
    return rows


def load_W():
    """WMT20 MLQE Task-1 (official) = the authoritative X->En QE-DA source
    (et-en / ne-en / si-en). Clean source/mt + full scores/mean/z_scores/z_mean
    on every split, so it replaces C's corrupted X->En. `translation` is a dict
    {'en': mt, <src>: source}; verified mean==mean(scores), z_mean==mean(z_scores)."""
    from datasets import load_dataset
    rows = []
    cfgmap = {"et-en": ("et", "eten"), "ne-en": ("ne", "neen"), "si-en": ("si", "sien")}
    for cfg, (srclang, pair) in cfgmap.items():
        try:
            ds = load_dataset("wmt/wmt20_mlqe_task1", cfg)
        except Exception:
            continue
        for split in ds:
            sp = {"validation": "dev"}.get(split, split)
            for x in ds[split]:
                tr = x.get("translation") or {}
                src, mt = tr.get(srclang), tr.get("en")
                if not src or not mt:
                    continue
                sc = _score_list(x.get("scores"))

                def _f(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return np.nan
                rows.append(_row("wmt20-mlqe", pair, src, mt, x.get("segid", ""), sp,
                                 da_scores=sc, da_mean=_f(x.get("mean")),
                                 da_z_scores=_score_list(x.get("z_scores")),
                                 da_z_mean=_f(x.get("z_mean")),
                                 n_annotators=(len(sc) if sc else np.nan)))
    return rows


def load_A():
    from datasets import load_dataset
    rows = []
    for cfg, sd in [("ape_qe_combined", "ape-combined"), ("ape_only", "ape-combined")]:
        df = load_dataset("sourabhd13/ape-qe-combined-data", cfg, split="train").to_pandas()
        for _, x in df.iterrows():
            pair = x["language_pair"]
            sc = _score_list(x.get("scores"))
            rows.append(_row("ape-combined", pair, x["original"], x["translation"],
                             x.get("id", ""), "unknown", domain=x.get("domain"),
                             post_edit=x.get("pe"), da_scores=sc,
                             da_mean=x.get("mean", np.nan),
                             n_annotators=(len(sc) if sc else np.nan),
                             _config=cfg))
    return rows


def load_B():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    DOM = {"health": "surrey-nlp/health-QE", "tourism": "surrey-nlp/Tourism-QE",
           "general": "surrey-nlp/General-QE", "legal": "surrey-nlp/Legal-QE"}
    pairmap = {"en-hindi": "enhi", "en-marathi": "enmr", "en-tamil": "enta",
               "en-gujarati": "engu", "en-telugu": "ente"}
    rows = []
    for dom, repo in DOM.items():
        for s in api.dataset_info(repo).siblings:
            f = s.rfilename
            if not f.endswith(".parquet"):
                continue
            d = f.split("/")[0]
            if d not in pairmap:
                continue
            split = "train" if "/train" in f else "validation" if "/validation" in f else "test"
            df = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"))
            src_c = next((c for c in ("source_text", "source", "original") if c in df.columns), None)
            mt_c = next((c for c in ("target_text", "translation", "mt") if c in df.columns), None)
            idc = next((c for c in ("ids", "id", "index") if c in df.columns), None)
            if not src_c or not mt_c:
                continue
            for _, x in df.iterrows():
                sc = _score_list(x.get("scores"))
                rows.append(_row("domain-qe", pairmap[d], x[src_c], x[mt_c],
                                 x.get(idc, "") if idc else "", split, domain=dom,
                                 da_scores=sc, da_mean=x.get("mean", np.nan),
                                 n_annotators=(len(sc) if sc else np.nan)))
    return rows


def load_R(local_dir):
    """ALOPE-RL (local clone). Canonical rows come from Other_Language_Pairs/*_combined.xlsx
    (en-hi/mr/ta/ml). en-ml is the 16-col superset (per-annotator DA + TQR + WTAG); en-hi/mr/ta
    are the 8-col WMT instances that MATCH onto C. The Deepseek/Gemini synthetic folder holds
    the same en-ml instances with alternative descriptions — not loaded as new rows."""
    if not local_dir or not os.path.isdir(local_dir):
        print("   [RL] no local clone given — RL columns stay null", file=sys.stderr)
        return []
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(local_dir, "Other_Language_Pairs", "*_combined.xlsx"))):
        base = os.path.basename(f).lower()           # e.g. en-hi_train_combined.xlsx
        pair = base.split("_")[0].replace("-", "")   # enhi/enmr/enta/enml
        split = "train" if "train" in base else "test"
        df = pd.read_excel(f)
        cols = {c.lower(): c for c in df.columns}

        def gv(x, key):
            c = cols.get(key)
            v = x[c] if c is not None else None
            return None if (v is None or (isinstance(v, float) and v != v)) else v

        for _, x in df.iterrows():
            das = [x[cols[k]] for k in ("da-a1", "da-a2", "da-a3")
                   if k in cols and pd.notna(x.get(cols[k]))]
            tqr = [x[cols[k]] for k in ("remark-a1", "remark-a2", "remark-a3")
                   if k in cols and _present(x.get(cols[k]))]
            raw_mt = str(x[cols["target"]])                 # tokenised, may carry <EOS>
            clean_mt = raw_mt.replace("<EOS>", "").strip()  # stored canonical mt
            # mt_tokens preserves the tokenisation WTAG aligns to (independent of the
            # canonical mt, which after merge comes from C's detokenised text)
            mt_tokens = raw_mt.split() if _present(gv(x, "word-level tags")) else None
            rows.append(_row("alope-rl", pair, x[cols["source"]], clean_mt,
                             gv(x, "id") or gv(x, "index") or "", split,
                             domain=gv(x, "domain"),
                             da_scores=([float(d) for d in das] or None),
                             da_mean=gv(x, "da_mean"),
                             n_annotators=(len(das) if das else np.nan),
                             post_edit=gv(x, "post-edited translation"),
                             wtag=gv(x, "word-level tags"), mt_tokens=mt_tokens,
                             tqr=(tqr or None),
                             synthetic_description=gv(x, "description of the translation"),
                             error_categories=gv(x, "identified error categories"),
                             explanation_model="deepseek"))
        print(f"   [RL] {base}: {len(df)} rows ({pair}/{split})", file=sys.stderr)
    print(f"   [RL] ingested {len(rows)} rows total", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------- #
#  Merge by uid  (precise instance matching across sources)
# --------------------------------------------------------------------------- #
def merge(rows):
    """Vectorised precise-match merge: group rows sharing a uid, carry the
    highest-priority non-null value for every field. No per-group Python loop."""
    df = pd.DataFrame(rows)
    df["prio"] = df["source_dataset"].map(SRC_PRIORITY).fillna(9)
    split_rank = {"test": 0, "dev": 1, "validation": 1, "train": 2, "unknown": 3}
    df["srank"] = df["split"].map(split_rank).fillna(3)

    # base = highest-priority row per uid (gives source, mt, ids, lang fields)
    base = (df.sort_values(["uid", "prio"]).drop_duplicates("uid", keep="first")
            .set_index("uid"))

    def first_valid(col, sort_col="prio"):
        sub = df[["uid", col, sort_col]].copy()
        sub = sub[sub[col].map(_present)].sort_values(["uid", sort_col])
        return sub.groupby("uid")[col].first()

    for col in ["domain", "post_edit", "reference", "wtag", "mt_tokens", "tqr",
                "synthetic_description", "error_categories", "explanation_model",
                "mqm_spans", "da_scores", "da_mean", "da_z_scores", "da_z_mean", "n_annotators"]:
        base[col] = first_valid(col).reindex(base.index)

    # DA-conflict flag: same uid, materially different DA across sources
    da = df[df["da_mean"].notna()]
    dstats = da.groupby("uid")["da_mean"].agg(["min", "max", "nunique"])
    conflict_mask = (dstats["nunique"] > 1) & (dstats["max"] - dstats["min"] > 1.0)
    conflicts = int(conflict_mask.sum())
    base["da_conflict"] = conflict_mask.reindex(base.index).fillna(False)

    # provenance set + most-held-out split, per uid
    base["sources"] = (df.drop_duplicates(["uid", "source_dataset"])
                       .groupby("uid")["source_dataset"]
                       .agg(lambda s: ",".join(sorted(s))).reindex(base.index))
    base["split"] = (df.sort_values("srank").drop_duplicates("uid", keep="first")
                     .set_index("uid")["split"].reindex(base.index))
    out = base.reset_index().drop(columns=["prio", "srank"], errors="ignore")
    # task-availability flags
    out["has_da"] = out["da_mean"].notna()
    out["has_pe"] = out["post_edit"].map(_present)
    out["has_wtag"] = out["wtag"].map(_present)
    out["has_tqr"] = out["tqr"].map(_present)
    out["has_synth_expl"] = out["synthetic_description"].map(_present)
    out["has_domain"] = out["domain"].map(_present)
    out["has_mqm"] = out["mqm_spans"].map(_present)
    out["gated_rl"] = out["has_wtag"] | out["has_tqr"] | out["has_synth_expl"]
    # cadence id: {SRC}-{PAIR}-{seq} in stable uid order (re-assigned in main() after any
    # signal-less rows are dropped, so it stays contiguous).
    out = out.sort_values(["language_pair", "uid"]).reset_index(drop=True)
    seq = out.groupby("language_pair").cumcount()
    # cadence prefix = the highest-PRIORITY contributing source (C before RL), not alphabetical
    src_tag = out["source_dataset"].str.upper().str.replace("-", "")
    out["cadence_id"] = (src_tag + "-" + out["language_pair"].str.upper().str.replace("-", "")
                         + "-" + seq.astype(str).str.zfill(6))
    return out, conflicts


def _present(v):
    if v is None:
        return False
    if isinstance(v, float) and v != v:
        return False
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) > 0
    return str(v).strip() not in ("", "nan", "None")


def _pick_split(g):
    for s in ["test", "dev", "validation", "train"]:   # prefer the most-held-out label
        if (g["split"] == s).any():
            return s
    return "unknown"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--rl-local", help="path to a local clone of the ALOPE-RL-Dataset")
    ap.add_argument("--sample", type=int, default=0, help="cap rows per source for a fast dry run")
    args = ap.parse_args()

    print(">> loading sources ...", file=sys.stderr)
    rows = load_C() + load_A() + load_B() + load_W() + load_R(args.rl_local)
    if args.sample:
        rows = rows[:args.sample]
    print(f">> {len(rows)} raw rows; merging by uid ...", file=sys.stderr)
    out, conflicts = merge(rows)

    # ---- domain normalisation + inference (canonical 6-way taxonomy + provenance) ----
    #   Collapses the two-lineage label redundancy (Health/health, General/News/general, ...)
    #   and fills a domain for every unlabelled row (all X->En + some En->X) via a TF-IDF
    #   English-side classifier. Adds domain_provenance / domain_confidence / domain_raw and
    #   redefines has_domain = gold-only. See dataset/normalize_domain.py + domain_report.md.
    try:
        from normalize_domain import finalize_domains
    except ImportError:
        from .normalize_domain import finalize_domains
    out, _dom_rep = finalize_domains(out)
    print(f">> domains: {_dom_rep.provenance_counts} "
          f"(classifier holdout acc={_dom_rep.holdout_accuracy}, macro-F1={_dom_rep.holdout_macro_f1})",
          file=sys.stderr)

    # ---- known source corrections (uid -> corrected scores); da_mean recomputed below ----
    for _u, _sc in _SCORE_PATCHES.items():
        for _i in out.index[out["uid"] == _u]:
            out.at[_i, "da_scores"] = list(_sc)
            out.at[_i, "n_annotators"] = float(len(_sc))
            print(f"score patch: uid {_u} -> {_sc}", file=sys.stderr)

    # ---- DA means are computed FROM the annotator score arrays (definitional):
    #        da_mean   := mean(da_scores)    wherever numeric da_scores exist
    #        da_z_mean := mean(da_z_scores)  wherever numeric z_scores exist
    #      This (a) FILLS nulls where the source `mean` column was blank but scores are
    #      present (e.g. 1,288 domain-QE en-mr rows), and (b) CORRECTS rows whose source
    #      mean contradicted its own scores (domain-QE + 1 C row). Rows with no scores keep
    #      their source value (e.g. APE-only rows stay DA-null; X->En now has full scores). ----
    def _mean_scores(v):
        if not isinstance(v, (list, tuple, np.ndarray)) or len(v) == 0:
            return np.nan
        try:
            return float(np.mean([float(x) for x in v]))
        except (TypeError, ValueError):
            return np.nan
    _dm0 = pd.to_numeric(out["da_mean"], errors="coerce")
    _ms = out["da_scores"].map(_mean_scores)
    n_fill = int((_ms.notna() & _dm0.isna()).sum())
    n_corr = int((_ms.notna() & _dm0.notna() & ((_ms - _dm0).abs() > 0.01)).sum())
    out.loc[_ms.notna(), "da_mean"] = _ms[_ms.notna()].round(6)
    _zms = out["da_z_scores"].map(_mean_scores)
    out.loc[_zms.notna(), "da_z_mean"] = _zms[_zms.notna()].round(6)
    print(f"DA from scores: da_mean=mean(scores) on {int(_ms.notna().sum())} rows "
          f"(filled {n_fill} nulls, corrected {n_corr}); da_z_mean=mean(z_scores) on "
          f"{int(_zms.notna().sum())} rows", file=sys.stderr)

    # ---- DA hygiene: DA must lie in [0,100]; null out-of-range (e.g. a "7580" typo
    #      in General-QE en-te gave a mean of 5078). Keep the instance if it has other
    #      signal (PE/WTAG); only its DA is dropped. ----
    da_num = pd.to_numeric(out["da_mean"], errors="coerce")
    bad_da = (da_num < 0) | (da_num > 100)
    n_bad = int(bad_da.sum())
    if n_bad:
        out.loc[bad_da, ["da_mean", "da_scores", "da_z_mean", "n_annotators"]] = \
            [np.nan, None, np.nan, np.nan]
        print(f"DA hygiene: nulled {n_bad} out-of-range DA value(s)", file=sys.stderr)
    # recompute has_da AFTER all DA edits (fills, corrections, range-nulling)
    out["has_da"] = pd.to_numeric(out["da_mean"], errors="coerce").notna()

    # ---- drop signal-less rows: a released instance must carry at least one task label
    #      (DA, post-edit, or a gated RL field). Catches both no-signal-ever rows and any
    #      row whose only DA was nulled as out-of-range. Then re-assign cadence_id so it
    #      stays contiguous on the final row set. ----
    keep = out["has_da"] | out["has_pe"].astype(bool) | out["gated_rl"].astype(bool)
    n_drop = int((~keep).sum())
    if n_drop:
        out = out[keep].reset_index(drop=True)
        print(f"dropped {n_drop} signal-less row(s) (no DA / PE / RL field)", file=sys.stderr)
    out = out.sort_values(["language_pair", "uid"]).reset_index(drop=True)
    _seq = out.groupby("language_pair").cumcount()
    _srct = out["source_dataset"].str.upper().str.replace("-", "", regex=False)
    out["cadence_id"] = (_srct + "-" + out["language_pair"].str.upper().str.replace("-", "", regex=False)
                         + "-" + _seq.astype(str).str.zfill(6))

    # nested/variable-type columns -> JSON strings for stable parquet typing
    import json
    def _ser(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return json.dumps([x.item() if hasattr(x, "item") else x for x in v], ensure_ascii=False)
        return v
    for col in ["da_scores", "da_z_scores", "wtag", "mt_tokens", "tqr", "error_categories", "mqm_spans"]:
        out[col] = out[col].map(_ser)

    # write ALL rows (incl. en-ml + gated) per pair -> "all" config. The gated_rl flag
    # column lets the public release filter RL-only fields until 2602.08600 is accepted.
    os.makedirs(os.path.join(args.out_dir, "unified"), exist_ok=True)
    for pair, g in out.groupby("language_pair"):
        g.to_parquet(os.path.join(args.out_dir, "unified", f"{pair}.parquet"))
    gated = out[out["gated_rl"]]

    # coverage matrix
    cov = out.groupby("language_pair").agg(
        n=("uid", "size"), has_da=("has_da", "sum"), has_pe=("has_pe", "sum"),
        has_wtag=("has_wtag", "sum"), has_tqr=("has_tqr", "sum"),
        has_synth=("has_synth_expl", "sum"), has_domain=("has_domain", "sum"),
        multi_source=("sources", lambda s: (s.str.contains(",")).sum()))
    cov.to_csv(os.path.join(args.out_dir, "coverage_matrix.csv"))
    print("\n===== UNIFIED COVERAGE MATRIX =====")
    print(cov.to_string())

    # RL ↔ C match rate: of instances carrying alope-rl annotations, how many merged
    # onto an existing WMT (C) instance vs landed as new rows (expected: en-ml = all new).
    rl = out[out["sources"].str.contains("alope-rl")]
    if len(rl):
        print("\n===== ALOPE-RL ↔ C MATCH RATE =====")
        for pair, g in rl.groupby("language_pair"):
            matched = g["sources"].str.contains("wmt-qe-da").sum()
            print(f"  {pair}: alope-rl rows={len(g)}  matched-onto-C={matched} "
                  f"({100*matched/len(g):.1f}%)  new-rows={len(g)-matched}")

    print(f"\ntotal unique instances: {len(out)}   DA-conflicts flagged: {conflicts}")
    print(f"multi-source merged instances: {(out['sources'].str.contains(',')).sum()}")
    print(f"instances carrying gated RL fields (WTAG/TQR/synthetic): {len(gated)}")
    print(f"  en-ml instances: {(out['language_pair']=='en-ml').sum()}")
    print(f"\nwritten to {args.out_dir}/unified/*.parquet (all rows, gated_rl flagged), coverage_matrix.csv")


if __name__ == "__main__":
    main()
