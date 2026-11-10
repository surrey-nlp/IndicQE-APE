#!/usr/bin/env python3
"""Full validation + provenance + missing-value audit of the consolidated IndicQE-APE release.
Runs over the assembled public repo (data/full = the superset) and cross-checks derived configs."""
import json, glob, os, sys
import numpy as np, pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else "release_build/hf_repo_public"
VALID_ROUTES = {"A1_disagreement", "A2_edit_effort", "A3_word_error", "A4_contradiction",
                "easy_control", "distributional", "co_selected", "pair_floor", "uncurated"}
TAX = {"health", "legal", "general", "tourism", "education", "news", "finance", "tech", "sports", "other"}
VALID_SRC = {"wmt-qe-da", "wmt20-mlqe", "domain-qe", "ape-combined", "alope-rl", "mlqe-pe"}
FAIL, WARN = [], []
def fail(m): FAIL.append(m); print("  ❌", m)
def warn(m): WARN.append(m); print("  ⚠️ ", m)

def load(cfg):
    fs = sorted(glob.glob(f"{BASE}/data/{cfg}/*.parquet"))
    return {os.path.basename(f).split('.')[0]: pd.read_parquet(f) for f in fs}

def jload(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return None
    return list(v) if hasattr(v, "__len__") else None

print("="*70, "\n1. CONFIG LOAD + ROW COUNTS\n", "="*70)
full = load("full"); F = pd.concat(full.values(), ignore_index=True)
print("full:", {k: len(v) for k, v in full.items()}, "total", len(F))
for cfg in ["all", "tasks/qe-da", "tasks/ape", "tasks/word-qe", "tasks/explainable-qe"]:
    d = load(cfg); n = sum(len(v) for v in d.values())
    print(f"{cfg}: {n}")
ch = pd.read_parquet(f"{BASE}/data/challenge/test.parquet"); print("challenge test:", len(ch))
pairs = load  # noqa

print("\n2. SCHEMA + PER-COLUMN NULL MAP (data/full)\n", "-"*60)
print(f"cols: {len(F.columns)}")
req = ["uid", "cadence_id", "sources", "language_pair", "src_lang", "tgt_lang", "source", "mt", "domain"]
for c in req:
    n = F[c].isna().sum() if c in F else len(F)
    if c not in F: fail(f"required col MISSING: {c}")
    elif n: fail(f"required col '{c}' has {n} nulls")
if not FAIL: print("  required columns all present + non-null ✓")
# per-pair null map for the informative columns
INFO = ["domain", "post_edit", "reference", "da_scores", "da_mean", "da_z_mean",
        "wtag", "tqr", "synthetic_description", "error_categories", "mqm_score"]
print("\n  per-pair present-count (non-null):")
tbl = F.groupby("language_pair").apply(lambda g: pd.Series({c: int(g[c].notna().sum()) for c in INFO}))
print(tbl.to_string())

print("\n3. PROVENANCE\n", "-"*60)
badsrc = set()
for s in F["sources"].dropna().unique():
    for part in str(s).split(","):
        if part and part not in VALID_SRC: badsrc.add(part)
if badsrc: fail(f"unknown source tags: {badsrc}")
else: print("  all source tags valid ✓ ; distribution:", F.sources.value_counts().to_dict())
if F["cadence_id"].nunique() != len(F): fail(f"cadence_id NOT unique ({F.cadence_id.nunique()} vs {len(F)})")
else: print("  cadence_id unique ✓")
print("  domain_provenance:", F.domain_provenance.value_counts().to_dict())
badprov = set(F.domain_provenance.unique()) - {"annotated", "inferred", "fallback"}
if badprov: fail(f"bad domain_provenance: {badprov}")

print("\n4. REFERENTIAL INTEGRITY\n", "-"*60)
for cfg in ["full", "all", "tasks/qe-da", "tasks/ape"]:
    d = load(cfg)
    tr, dv, te = (set(d.get(s, pd.DataFrame({'uid':[]})).uid) for s in ("train", "validation", "test"))
    leak = (tr & te) | (tr & dv) | (te & dv)
    if leak: fail(f"{cfg}: {len(leak)} uid split-leak")
    allu = pd.concat(d.values()).uid
    if allu.duplicated().any(): warn(f"{cfg}: {int(allu.duplicated().sum())} duplicate uids within config")
print("  split-disjointness checked ✓ (fails above if any)")
# challenge ⊆ full test
ft = full["test"];
if not set(ch.uid).issubset(set(ft.uid)): fail("challenge test NOT subset of full test")
else: print("  challenge ⊆ full test ✓")
# is_curated consistency
cur_pairs = set(F[F.is_curated].language_pair.unique())
print("  curated pairs:", sorted(cur_pairs))

print("\n5. VALUE SANITY\n", "-"*60)
da = pd.to_numeric(F.da_mean, errors="coerce")
oob = ((da < 0) | (da > 100)).sum()
if oob: fail(f"{oob} da_mean out of [0,100]")
else: print("  da_mean in [0,100] ✓ (range %.1f–%.1f)" % (da.min(), da.max()))
# da_mean == mean(da_scores) and n_annotators == len(scores)
mism_mean = mism_n = checked = 0
for r in F[F.da_scores.notna()].itertuples():
    sc = jload(r.da_scores)
    if not sc: continue
    checked += 1
    if not pd.isna(r.da_mean) and abs(float(r.da_mean) - float(np.mean(sc))) > 1.0: mism_mean += 1
    if not pd.isna(r.n_annotators) and int(r.n_annotators) != len(sc): mism_n += 1
print(f"  da_scores checked {checked}: da_mean≠mean {mism_mean}, n_annotators≠len {mism_n}")
if mism_mean > checked*0.02: warn(f"{mism_mean} rows da_mean far from mean(scores)")
if mism_n: warn(f"{mism_n} rows n_annotators != len(da_scores)")
# route + domain validity
br = set(F.route.dropna().unique()) - VALID_ROUTES
if br: fail(f"invalid routes: {br}")
bd = set(F.domain.dropna().unique()) - TAX
if bd: fail(f"invalid domains: {bd}")
if not br and not bd: print("  routes + domains within controlled vocab ✓")
# has_* consistency
for flag, col in [("has_da", "da_mean"), ("has_pe", "post_edit"), ("has_domain", "domain")]:
    if flag in F and col in F:
        # has_domain is provenance==annotated, not "domain present" — check accordingly
        if flag == "has_domain":
            bad = (F.has_domain != (F.domain_provenance == "annotated")).sum()
        else:
            present = F[col].notna() if col == "da_mean" else F[col].map(lambda v: bool(v) and str(v).strip() not in ("", "None"))
            bad = (F[flag].fillna(False) != present).sum()
        if bad: warn(f"{flag} inconsistent with {col}: {bad} rows")

print("\n6. TEXT QUALITY\n", "-"*60)
emp_s = (F.source.astype(str).str.strip() == "").sum(); emp_m = (F.mt.astype(str).str.strip() == "").sum()
if emp_s or emp_m: fail(f"empty source={emp_s} mt={emp_m}")
else: print("  no empty source/mt ✓")
srcmt = (F.source.astype(str).str.strip() == F.mt.astype(str).str.strip()).sum()
if srcmt: warn(f"{srcmt} rows have source==mt (untranslated?)")
dup = F.duplicated(["language_pair", "source", "mt"]).sum()
if dup: warn(f"{dup} duplicate (pair,source,mt) rows")
else: print("  no duplicate (pair,source,mt) ✓")
# PE == MT (no-edit) informational
pe = F[F.post_edit.notna()]
noedit = (pe.post_edit.astype(str).str.strip() == pe.mt.astype(str).str.strip()).sum()
print(f"  post-edits: {len(pe)} ; identical-to-MT (no-edit) {noedit} ({100*noedit/max(len(pe),1):.0f}%)")

print("\n7. MISSING-VALUE MAP (what each pair lacks)\n", "-"*60)
for p, g in F.groupby("language_pair"):
    miss = []
    if g.post_edit.notna().sum() == 0: miss.append("post_edit(APE)")
    if g.wtag.notna().sum() == 0: miss.append("wtag(word-QE)")
    if g.tqr.notna().sum() == 0: miss.append("tqr(remarks)")
    if g.da_z_mean.notna().sum() == 0: miss.append("da_z(z-scores)")
    if g.reference.notna().sum() == 0: miss.append("reference")
    if g.mqm_score.notna().sum() == 0: miss.append("mqm")
    infd = int((g.domain_provenance == "inferred").sum())
    print(f"  {p:7s} n={len(g):6d}  missing: {', '.join(miss) or 'nothing'}"
          + (f"  (+{infd} inferred domains)" if infd else ""))

print("\n" + "="*70)
print(f"RESULT: {len(FAIL)} FAIL, {len(WARN)} WARN")
if FAIL: print("FAILURES:", *FAIL, sep="\n  - ")
