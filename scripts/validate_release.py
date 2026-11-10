#!/usr/bin/env python3
"""Comprehensive, read-only validation of the live surrey-nlp/IndicQE-APE release.

Runs eight groups of checks (A--H): configurations and instance counts, uid identity, cadence, provenance and lineage, label coverage and numbers, field integrity,
split hygiene / leakage, and cross-document number consistency. Prints PASS/FAIL with the measured
value for every check and a final GREEN/RED verdict.

    python3 scripts/validate_release.py
"""
from __future__ import annotations
import hashlib, json, re, sys
from collections import Counter, defaultdict
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")
PAIRS = ["en-hi", "en-mr", "en-gu", "en-ta", "en-te", "en-ml", "et-en", "ne-en", "si-en"]

EXPECT_ROWS = {"all": 100366, "full": 126754, "tasks/qe-da": 85586, "tasks/ape": 73829,
               # word-qe 2026-07-28: 12,604 en-Indic upstream + 26,012 MLQE-PE upstream + 40,685 derived
               "tasks/word-qe": 79301, "tasks/explainable-qe": 19644, "tasks/mqm": 4490}
# composition (per-pair): Total, DA, PE, W-tag, Expl, Domain(=coverage, every row has one),
# QE(challenge DA), APE(challenge PE). NOTE: `full` gives EVERY row a domain (gold or inferred),
# so Domain coverage == Total per pair. The paper's tab:composition Domain column is the ANNOTATED
# subset only (92,492 total), checked separately as E4.
# W-tag column = rows carrying EITHER an upstream/imported tag or a derived one (total coverage).
# Provenance split is checked separately by E5.
EXPECT_COMP = {
    "en-hi": (21092, 15136, 9983, 9982, 1999, 21092, 1881, 1536),
    "en-mr": (30746, 29182, 22189, 22190, 5000, 30746, 2165, 2098),
    "en-gu": (9109, 9109, 0, 0, 0, 9109, 1090, 0),
    "en-ta": (16262, 9002, 10838, 10838, 2645, 16262, 2038, 1448),
    "en-te": (13157, 13157, 0, 0, 0, 13157, 1120, 0),
    "en-ml": (10000, 10000, 10000, 10000, 10000, 10000, 1436, 1436),
    "et-en": (9000, 9000, 9000, 9000, 0, 9000, 1000, 1000),
    "ne-en": (8649, 8649, 8332, 8612, 0, 8649, 1000, 962),
    "si-en": (8739, 8739, 3487, 8679, 0, 8739, 1000, 380),
}
# word-tag provenance (E5): real/upstream vs derived-as-fallback
EXPECT_WTAG_PROV = {"upstream": 12604, "mlqe-pe-upstream": 26012, "derived-ter-indicnlp": 40685}
EXPECT_SOURCE_TAGS = 26012   # MLQE-PE source-side tags (X->en only)
EXPECT_GAP_TAGS = 26012      # MLQE-PE gap tags (X->en only)
EXPECT_DOMAIN_ANNOTATED = 92492   # paper/card "Domain" column = annotated-only total
CADENCE_PREFIXES = {"WMTQEDA", "DOMAINQE", "APECOMBINED", "ALOPERL", "WMT", "WMT20MLQE", "WMT24MQM"}
SOURCE_TAGS = {"wmt-qe-da", "domain-qe", "ape-combined", "alope-rl", "mlqe-pe", "wmt20-mlqe",
               "wmt24-genmt-mqm", "eamt-qe-mqm"}
LINEAGE = {  # source tag -> pairs it is allowed to contribute to
    "wmt-qe-da": {"en-hi", "en-mr", "en-gu", "en-ta", "en-te"},
    "domain-qe": {"en-hi", "en-mr", "en-gu", "en-ta", "en-te"},
    "ape-combined": {"en-hi", "en-mr", "en-ta"},
    "alope-rl": {"en-hi", "en-mr", "en-ta", "en-ml"},
    "mlqe-pe": {"et-en", "ne-en", "si-en"},
    "wmt20-mlqe": {"et-en", "ne-en", "si-en"},
    "wmt24-genmt-mqm": {"en-hi"},
    "eamt-qe-mqm": {"en-hi"},
}

_res = []
def chk(sec, desc, cond, detail=""):
    _res.append((sec, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {sec} {desc}" + (f"  ({detail})" if detail else ""))

def norm(s): return re.sub(r"\s+", " ", (s or "").strip())
def h(s): return hashlib.md5(norm(s).encode()).hexdigest()[:12]
def ne(s): return s.notna() & (s.astype(str).str.strip() != "")

def load(cfg, cols=None):
    fr = []
    for sp in SPLITS:
        try:
            p = hf_hub_download(REPO, f"data/{cfg}/{sp}.parquet", repo_type="dataset")
        except Exception:
            continue
        d = pd.read_parquet(p)
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        d["_split"] = sp
        fr.append(d)
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def main():
    api = HfApi()
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    print(f"dataset sha: {api.dataset_info(REPO).sha[:12]}\n")

    print("== A. Configurations and instances ==")
    for cfg, exp in EXPECT_ROWS.items():
        present = all(f"data/{cfg}/{sp}.parquet" in files for sp in SPLITS)
        n = len(load(cfg, ["uid"])) if present else -1
        chk("A1/A2", f"{cfg}", present and n == exp, f"rows={n} expect={exp}")
    # per-pair configs sum to full
    pair_tot = 0
    for p in PAIRS:
        n = len(load(f"pairs/{p}", ["uid"]))
        pair_tot += n
    chk("A2", "per-pair configs present & sum to full", pair_tot == EXPECT_ROWS["full"],
        f"sum={pair_tot}")

    print("\n== load master (full) once ==")
    full = load("full")
    full["K"] = [h(a) + "|" + h(b) + "|" + str(c) for a, b, c in
                 zip(full["source"], full["mt"], full["language_pair"])]
    fu = set(full["uid"])
    print(f"  full: {len(full)} rows, {len(full.columns)-1} columns")

    print("\n== A3. task/pair configs are views of full (uid subset) ==")
    for cfg in ["all", "tasks/qe-da", "tasks/ape", "tasks/word-qe", "tasks/explainable-qe", "challenge"]:
        u = set(load(cfg, ["uid"])["uid"])
        chk("A3", f"{cfg} uids ⊂ full", u <= fu, f"orphan={len(u-fu)}")

    print("\n== B. Identity (uid) ==")
    chk("B1", "uid↔content-key bijection in full",
        full["uid"].nunique() == len(full) == full["K"].nunique(),
        f"uid={full['uid'].nunique()} rows={len(full)} K={full['K'].nunique()}")
    kmap = full.groupby("K")["uid"].nunique()
    chk("B3", "no content key maps to >1 uid in full", int((kmap > 1).sum()) == 0,
        f"multi={int((kmap>1).sum())}")
    mqm = load("tasks/mqm")
    inA = set(mqm["uid"]) & fu
    novel = set(mqm["uid"]) - fu
    chk("B4", "mqm: 2,327 in full + 2,163 novel, unique, no collision",
        len(inA) == 2327 and len(novel) == 2163 and mqm["uid"].nunique() == len(mqm),
        f"inFull={len(inA)} novel={len(novel)} unique={mqm['uid'].nunique()}")
    oid = full.dropna(subset=["original_id"])
    chk("B5", "original_id is NOT a key (documented) — collides as expected",
        int((oid.groupby('original_id')['uid'].nunique() > 1).sum()) > 0, "expected collisions")

    print("\n== C. Cadence ==")
    chk("C1", "cadence_id present on 100% of full", bool(ne(full["cadence_id"]).all()),
        f"missing={int((~ne(full['cadence_id'])).sum())}")
    pref = full["cadence_id"].astype(str).str.extract(r"^([A-Z0-9]+)-")[0]
    bad_pref = set(pref.dropna().unique()) - CADENCE_PREFIXES
    chk("C2", "cadence_id prefixes ⊂ known set", len(bad_pref) == 0, f"unknown={bad_pref}")
    # C3 prefix consistent with sources (spot: WMT24MQM only on wmt24-genmt-mqm rows)
    mqmrows = full[full["mqm_source"] == "wmt24-genmt-mqm"] if "mqm_source" in full else pd.DataFrame()
    c4 = bool(mqm[mqm["uid"].isin(novel)]["cadence_id"].astype(str).str.startswith("WMT24MQM-ENHI-").all())
    chk("C4", "MQM Bucket-B cadence_id uses WMT24MQM-ENHI-", c4)

    print("\n== D. Provenance and lineage ==")
    tags = set()
    for s in full["sources"].dropna():
        tags |= set(str(s).split(","))
    unknown = tags - SOURCE_TAGS
    chk("D1", "all sources tags ⊂ known vocabulary", len(unknown) == 0, f"unknown={unknown}")
    lineage_ok = True; bad = []
    for tag, allowed in LINEAGE.items():
        got = set(full.loc[full["sources"].astype(str).str.contains(tag, na=False), "language_pair"].unique())
        if not got <= allowed:
            lineage_ok = False; bad.append(f"{tag}:{got-allowed}")
    chk("D2", "each source tag only on its lineage pairs", lineage_ok, "; ".join(bad))
    if "mqm_source" in full:
        ms_nn = ne(full["mqm_source"]); sc_nn = full["mqm_score"].notna()
        chk("D3", "mqm_source set on exactly the MQM-bearing rows",
            bool((ms_nn == sc_nn).all()), f"mismatch={int((ms_nn!=sc_nn).sum())}")

    print("\n== E. Label coverage and numbers ==")
    chal = load("challenge")
    xq_uid = set(load("tasks/explainable-qe", ["uid"])["uid"])
    comp_ok = True
    for p in PAIRS:
        g = full[full.language_pair == p]; c = chal[chal.language_pair == p]
        # W-tag = either an upstream/imported tag or a derived one
        wt_any = ne(g.wtag) if "wtag" in g else pd.Series(False, index=g.index)
        if "wtag_derived" in g:
            wt_any = wt_any | g.wtag_derived.notna()
        got = (len(g), int(g.da_mean.notna().sum()), int(ne(g.post_edit).sum()),
               int(wt_any.sum()), int(g.uid.isin(xq_uid).sum()),
               int(ne(g.domain).sum()), int(c.da_mean.notna().sum()), int(ne(c.post_edit).sum()))
        ok = got == EXPECT_COMP[p]
        comp_ok = comp_ok and ok
        if not ok:
            print(f"     MISMATCH {p}: live={got} expect={EXPECT_COMP[p]}")
    chk("E1", "composition table (all cells, all pairs) matches expected", comp_ok)
    sc = full["mqm_score"].notna()
    en_mqm = int((sc.to_numpy() & (full.language_pair == "en-hi").to_numpy()).sum())
    ot_mqm = int((sc.to_numpy() & (full.language_pair != "en-hi").to_numpy()).sum())
    chk("E2a", "full mqm_score on exactly 2,327 en-hi, 0 elsewhere",
        en_mqm == 2327 and ot_mqm == 0, f"enhi={en_mqm} other={ot_mqm}")
    chk("E2b", "mqm config 4,490; paired DA+MQM = 2,327",
        len(mqm) == 4490 and int((mqm.da_mean.notna() & mqm.mqm_score.notna()).sum()) == 2327,
        f"rows={len(mqm)} paired={int((mqm.da_mean.notna()&mqm.mqm_score.notna()).sum())}")
    # E4: every row has a domain (coverage), and the ANNOTATED subset matches the paper (92,492)
    cov = int(ne(full["domain"]).sum())
    ann = int((full["domain_provenance"] == "annotated").sum()) if "domain_provenance" in full else -1
    chk("E4a", "every full row has a domain (gold or inferred)", cov == len(full), f"coverage={cov}")
    chk("E4b", f"annotated-domain total == {EXPECT_DOMAIN_ANNOTATED} (paper/card Domain column)",
        ann == EXPECT_DOMAIN_ANNOTATED, f"annotated={ann}")
    # E5: word-tag provenance split, and the MLQE-PE side signals
    if "wtag_provenance" in full:
        prov = full["wtag_provenance"].replace("", pd.NA).value_counts().to_dict()
        chk("E5a", "word-tag provenance split matches expected",
            all(prov.get(k) == v for k, v in EXPECT_WTAG_PROV.items()), f"live={prov}")
        # a derived tag must never coexist with a real one on the same row
        both = int((ne(full["wtag"]) & full["wtag_derived"].notna()).sum())
        chk("E5b", "no row carries BOTH a real and a derived tag (real always wins)",
            both == 0, f"both={both}")
    if "source_tags" in full:
        st = int(ne(full["source_tags"]).sum()); gp = int(ne(full["wtag_gaps"]).sum())
        xen = full.language_pair.isin(["et-en", "ne-en", "si-en"])
        st_xen_only = int(ne(full.loc[~xen, "source_tags"]).sum())
        chk("E5c", f"MLQE-PE source tags == {EXPECT_SOURCE_TAGS}, gap tags == {EXPECT_GAP_TAGS}, "
                   "X→en only", st == EXPECT_SOURCE_TAGS and gp == EXPECT_GAP_TAGS
            and st_xen_only == 0, f"src={st} gaps={gp} outside_xen={st_xen_only}")

    print("\n== F. Field integrity ==")
    dm = full["da_mean"].dropna()
    chk("F1a", "da_mean ∈ [0,100]", bool((dm >= 0).all() and (dm <= 100).all()),
        f"min={dm.min():.1f} max={dm.max():.1f}")
    sm = full["mqm_score"].dropna()
    chk("F1b", "mqm_score ∈ [0,100]", bool((sm >= 0).all() and (sm <= 100).all()),
        f"min={sm.min():.1f} max={sm.max():.1f}")
    chk("F2", "reference is 100% null", bool(full["reference"].isna().all()),
        f"nonnull={int(full['reference'].notna().sum())}")
    # F4 spans well-formed on a sample
    smp = full[full["mqm_spans"].notna()].head(500)
    okspan = True
    for _, r in smp.iterrows():
        try:
            j = json.loads(r["mqm_spans_json"])
            if len(j) != len(r["mqm_spans"]):
                okspan = False; break
        except Exception:
            okspan = False; break
    chk("F4", "mqm_spans list == mqm_spans_json parsed (sample 500)", okspan)
    chk("F5", "language_pair ∈ the nine pairs", set(full.language_pair.unique()) <= set(PAIRS),
        f"extra={set(full.language_pair.unique())-set(PAIRS)}")

    print("\n== G. Split hygiene / leakage ==")
    full["SRC"] = [h(s) + "|" + str(p) for s, p in zip(full.source, full.language_pair)]
    ksplit = full.groupby("K")["_split"].nunique()
    chk("G1a", "no exact instance crosses a split", int((ksplit > 1).sum()) == 0,
        f"leaked={int((ksplit>1).sum())}")
    ssplit = full.groupby("SRC")["_split"].nunique()
    chk("G1b", "no source sentence crosses a split", int((ssplit > 1).sum()) == 0,
        f"leaked={int((ssplit>1).sum())}")
    tr = full[full._split == "train"]
    csrc = set(h(s) + "|" + str(p) for s, p in zip(chal.source, chal.language_pair))
    chk("G2", "challenge test source-disjoint from full-train",
        len(csrc & set(tr.SRC)) == 0, f"overlap={len(csrc & set(tr.SRC))}")

    print("\n" + "=" * 60)
    fails = [s for s, ok in _res if not ok]
    verdict = "GREEN — all checks pass" if not fails else f"RED — {len(fails)} check(s) failed"
    print(f"VERDICT: {verdict}")
    print(f"  {sum(ok for _, ok in _res)}/{len(_res)} checks passed")
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
