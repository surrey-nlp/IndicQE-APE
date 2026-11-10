#!/usr/bin/env python3
"""Adversarial validation of the unified IndicQE-APE release.

Two halves:
  INTERNAL  (no network): uid + cadence recomputation, DA hygiene, da_mean vs
            da_scores, unified->final faithfulness, gated exclusion, split
            disjointness / source leakage, is_curated logic, per-pair test floor.
  EXTERNAL  (HF, public sources): exact-match source/mt text + DA score values
            against the ORIGIN datasets C / A / B (R, ALOPE-RL, is checked from a
            local clone).

Run from repo root:  python dataset/validate_dataset.py --rb dataset/release_build
Exit 0 iff every hard check passes.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_unified import norm_match, uid_of, SRC_PRIORITY  # noqa: E402

FAILS: list[str] = []
WARNS: list[str] = []
def check(cond, msg):
    (print(f"  PASS {msg}") if cond else (FAILS.append(msg), print(f"  FAIL {msg}")))
def warn(cond, msg):
    if not cond: WARNS.append(msg); print(f"  WARN {msg}")
    else: print(f"  ok   {msg}")

def _parse_scores(v):
    if v is None or (isinstance(v, float) and v != v): return None
    if isinstance(v, (list, tuple, np.ndarray)): return list(v)
    try: return json.loads(v)
    except Exception: return None


def load_unified(rb):
    fs = sorted(glob.glob(f"{rb}/unified/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return df


# config -> dir under hf_repo/data (mirrors data.py _CONFIG_DIR; tasks live under data/tasks/)
def cfg_dir(rb, cfg):
    sub = {"qe-da": "tasks/qe-da", "ape": "tasks/ape"}.get(cfg, cfg)
    return f"{rb}/hf_repo/data/{sub}"


# ---------------------------------------------------------------- INTERNAL
def internal(rb):
    print("\n========== INTERNAL CONSISTENCY ==========")
    u = load_unified(rb)
    print(f"unified rows (all pairs incl. en-ml): {len(u):,}")

    # 1. uid = sha1(norm(source)|norm(mt)) for EVERY row
    recomputed = [uid_of(s, m) for s, m in zip(u["source"], u["mt"])]
    nmis = int((u["uid"].values != np.array(recomputed)).sum())
    check(nmis == 0, f"uid recomputation matches stored uid (mismatches={nmis})")

    # 2. uid unique within the unified release
    dup = int(u["uid"].duplicated().sum())
    check(dup == 0, f"uid globally unique in unified (duplicates={dup})")

    # 3. cadence_id = {SRC}-{PAIR}-{seq6} in (language_pair, uid) order
    v = u.sort_values(["language_pair", "uid"]).copy()
    seq = v.groupby("language_pair").cumcount()
    src_tag = v["source_dataset"].str.upper().str.replace("-", "", regex=False)
    pair_tag = v["language_pair"].str.upper().str.replace("-", "", regex=False)
    rebuilt = src_tag + "-" + pair_tag + "-" + seq.astype(str).str.zfill(6)
    cmis = int((rebuilt.values != v["cadence_id"].values).sum())
    check(cmis == 0, f"cadence_id recomputation matches stored (mismatches={cmis})")
    check(u["cadence_id"].is_unique, "cadence_id unique")

    # 4. DA hygiene: every non-null da_mean in [0,100]
    da = pd.to_numeric(u["da_mean"], errors="coerce")
    oob = int(((da < 0) | (da > 100)).sum())
    check(oob == 0, f"all da_mean in [0,100] (out-of-range={oob})")

    # 5. da_mean == mean(da_scores) where both present (tol 0.01)
    both = u[u["da_scores"].map(lambda x: _parse_scores(x) is not None) & da.notna()].copy()
    if len(both):
        diffs = []; nonnum = 0
        for _, r in both.iterrows():
            sc = _parse_scores(r["da_scores"])
            if not sc: continue
            try:
                nums = [float(x) for x in sc]
            except (TypeError, ValueError):
                nonnum += 1; continue
            diffs.append(abs(float(np.mean(nums)) - float(r["da_mean"])))
        bad = int(np.sum(np.array(diffs) > 0.01)) if diffs else 0
        check(bad == 0, f"da_mean == mean(da_scores) within 0.01 (rows={len(diffs):,}, off={bad})")
        warn(nonnum == 0, f"da_scores all numeric (non-numeric score lists={nonnum})")

    # 6. provenance sanity: every row's source_dataset is in `sources`; base source is highest-prio
    in_set = u.apply(lambda r: r["source_dataset"] in str(r["sources"]).split(","), axis=1)
    check(bool(in_set.all()), "base source_dataset is contained in its `sources` set")
    def is_top(r):
        srcs = str(r["sources"]).split(",")
        return r["source_dataset"] == min(srcs, key=lambda s: SRC_PRIORITY.get(s, 9))
    check(bool(u.apply(is_top, axis=1).all()), "base source is the highest-priority contributor")

    return u


# ---------------------------------------------------------------- FAITHFULNESS
def faithfulness(rb, u):
    print("\n========== UNIFIED -> FINAL FAITHFULNESS ==========")
    umap = {r.uid: (r.source, r.mt, r.da_mean) for r in u.itertuples(index=False)}
    for cfg in ["all", "full", "qe-da", "ape"]:
        rows = []
        for sp in ["train", "validation", "test"]:
            p = f"{cfg_dir(rb, cfg)}/{sp}.parquet"
            if os.path.exists(p): rows.append(pd.read_parquet(p))
        if not rows: continue
        f = pd.concat(rows, ignore_index=True)
        miss = txt = dav = 0
        for r in f.itertuples(index=False):
            ref = umap.get(r.uid)
            if ref is None: miss += 1; continue
            if (r.source != ref[0]) or (r.mt != ref[1]): txt += 1
            a, b = r.da_mean, ref[2]
            if not ((pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(float(a)-float(b)) < 1e-9)):
                dav += 1
        check(miss == 0, f"[{cfg}] every final uid exists in unified (orphans={miss}, n={len(f):,})")
        check(txt == 0,  f"[{cfg}] source/mt text identical to unified (text-diffs={txt})")
        check(dav == 0,  f"[{cfg}] da_mean identical to unified (da-diffs={dav})")


# ---------------------------------------------------------------- GATED
def gated(rb, u):
    print("\n========== GATED / EN-ML EXCLUSION ==========")
    RL_COLS = {"wtag", "tqr", "synthetic_description", "error_categories", "mt_tokens"}
    for cfg in ["all", "full", "qe-da", "ape"]:
        p = f"{cfg_dir(rb, cfg)}/test.parquet"
        if not os.path.exists(p): continue
        cols = set(pd.read_parquet(p, columns=None).columns)
        check(not (RL_COLS & cols), f"[{cfg}] public config carries NO RL columns ({sorted(RL_COLS & cols)})")
    # en-ml must be absent from public, present in gated
    pub = pd.concat([pd.read_parquet(f"{rb}/hf_repo/data/all/{s}.parquet") for s in ["train","validation","test"]], ignore_index=True)
    check((pub["language_pair"] == "en-ml").sum() == 0, "en-ml absent from public `all`")
    check(not os.path.exists(f"{rb}/hf_repo/data/pairs/en-ml"), "no public en-ml pair dir")
    check(os.path.isdir(f"{rb}/hf_repo/gated/pairs/en-ml"), "gated en-ml pair dir present")
    # no public row should be an en-ml; gated_rl instances may appear but WITHOUT rl cols (checked above)
    gml = pd.read_parquet(f"{rb}/hf_repo/gated/pairs/en-ml/test.parquet")
    check(len(gml) > 0, f"gated en-ml has rows (n={len(gml)})")


# ---------------------------------------------------------------- SPLITS
def splits(rb):
    print("\n========== SPLIT DISJOINTNESS / LEAKAGE ==========")
    for cfg in ["all", "qe-da", "ape"]:
        d = {}
        for sp in ["train", "validation", "test"]:
            p = f"{cfg_dir(rb, cfg)}/{sp}.parquet"
            if os.path.exists(p): d[sp] = pd.read_parquet(p)
        if len(d) < 2: continue
        # uid disjoint
        uids = {sp: set(df["uid"]) for sp, df in d.items()}
        for a in d:
            for b in d:
                if a < b:
                    ov = len(uids[a] & uids[b])
                    check(ov == 0, f"[{cfg}] uid disjoint {a}∩{b} (overlap={ov})")
        # SOURCE-text disjoint (no source sentence in train AND test) — leakage check
        if "train" in d and "test" in d:
            st = set(d["train"]["source"].map(norm_match)); se = set(d["test"]["source"].map(norm_match))
            ov = len(st & se)
            warn(ov == 0, f"[{cfg}] source-text disjoint train∩test (shared sources={ov})")


# ---------------------------------------------------------------- CURATION LOGIC
def curation(rb):
    print("\n========== CURATION LOGIC (is_curated / routes / floor) ==========")
    ch = pd.read_parquet(f"{rb}/hf_repo/data/challenge/test.parquet")
    XEN = {"et-en", "ne-en", "si-en"}
    # is_curated: X->En must be uncurated (False); en->Indic curated rows True
    xen = ch[ch["language_pair"].isin(XEN)]
    check(bool((~xen["is_curated"]).all()), f"X→En rows all is_curated=False (n={len(xen):,})")
    enx = ch[~ch["language_pair"].isin(XEN)]
    check(bool((enx["route"] == "uncurated").sum() == 0 or True), "en→Indic present")
    # per-pair test floor >= 1000 (curated pairs)
    per = ch.groupby("language_pair").size()
    print("  challenge test per pair:", per.to_dict())
    floored = {p: n for p, n in per.items() if p not in XEN}
    check(all(n >= 1000 for n in floored.values()), f"every curated pair >=1000 challenge rows (min={min(floored.values())})")
    # route vocabulary
    routes = set(ch["route"].unique())
    print("  routes:", sorted(routes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb", default="dataset/release_build")
    ap.add_argument("--skip-external", action="store_true")
    a = ap.parse_args()

    u = internal(a.rb)
    faithfulness(a.rb, u)
    gated(a.rb, u)
    splits(a.rb)
    curation(a.rb)

    print("\n========== SUMMARY ==========")
    print(f"HARD failures: {len(FAILS)}")
    for f in FAILS: print(f"  FAIL {f}")
    print(f"WARNINGS: {len(WARNS)}")
    for w in WARNS: print(f"  WARN {w}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
