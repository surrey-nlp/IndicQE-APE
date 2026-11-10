#!/usr/bin/env python3
r"""axis_contrast.py — QE accuracy on each difficulty axis, against a control drawn from the same pair.

THE QUANTITY
For a system, a language pair and an axis, the contrast is

    rho(the pair's flagged segments) - rho(a control drawn from the same pair)

so a pair is only ever compared against itself. Two controls are reported. The plain control is
every unflagged segment of the pair. The matched control is a subsample of those unflagged segments
drawn to reproduce the flagged set's direct-assessment histogram, in bins of five DA points; it is
averaged over DRAWS independent draws. The matched control is the estimate to read, because A1 and
A4 are defined partly on the direct assessment and so select a narrower slice of the score range
than the pair as a whole, which attenuates a rank correlation whether or not the segments are
harder. The axis figure is the mean over the pairs that carry the axis, and the interval is a
cluster bootstrap that resamples pairs.

THE AXES
Deterministic functions of released columns. Quantiles and z scores are taken within pair.

    da_std        population standard deviation of the released per-annotator scores
    human_ter     token TER of the MT against the human post-edit
    bad_density   share of BAD tokens in the released word tags
    hard_error    error_categories names mistranslation, untranslated or addition

    A1  da_std in the pair's top quartile AND da_mean in [35, 75]
    A2  human_ter in the pair's top quartile
    A3  bad_density in the pair's top quartile OR hard_error
    A4  (z(da_mean) > 0.5 AND z(bad_density) > 0.5)                          scored well, error-dense
        OR (da_mean <= the pair's 30th percentile AND z(human_ter) < -0.5)   scored badly, barely edited

COVERAGE IS AN INVENTORY
A1 needs the per-annotator scores, which all nine pairs carry. A2 needs post-edits, A3 needs word
tags or error categories, A4 needs a direct assessment against one of those. en-gu and en-te ship
from WMT23/24 as direct assessments alone, so they carry A1 and are null, not negative, elsewhere.

    python3 analysis/axis_contrast.py           # report, write CSVs, splice the tables
    python3 analysis/axis_contrast.py --check   # fail if the tex has drifted from the data
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

sys.path.insert(0, str(ROOT / "src"))
from indicqe_ape.prompts import parse_qe_score_strict  # noqa: E402

TEX = _PAPER_TEX
OUTD = ROOT / "results" / "analysis"
SNAP = ROOT / "dataset" / "hf_snapshot" / "data"

PAIRS = ["en-hi", "en-mr", "en-ta", "en-gu", "en-te", "en-ml", "et-en", "ne-en", "si-en"]
AXES = ["A1", "A2", "A3", "A4"]
HARD = ("mistranslation", "untranslated", "addition")
MIN_N = 30       # a cell smaller than this is not estimated
BINW = 5.0       # DA bin width for the matched control
DRAWS = 60       # matched-control draws averaged over
BOOT = 10000
SEED = 20260812

PROMPTED = {"gpt-5.5": "qe_closed_rerun", "sarvam-m": "qe_rerun512",
            "aya-expanse-32b": "qe_rerun512", "tiny-aya-fire": "qe_rerun512",
            "Llama-3.2-3B-Instruct": "qe_rerun512", "BharatGPT-3B-Indic": "qe_rerun512"}
TRAINED = ["cometkiwi-da", "cometkiwi-xl", "xcomet-xl"]
MACRO = {"gpt-5.5": r"\mGPT", "sarvam-m": r"\mSarvam", "aya-expanse-32b": r"\mAyaB",
         "tiny-aya-fire": r"\mTiny", "Llama-3.2-3B-Instruct": r"\mLlamaB",
         "BharatGPT-3B-Indic": r"\mBharatB", "cometkiwi-da": r"\mCKDA",
         "cometkiwi-xl": r"\mCKXL", "xcomet-xl": r"\mXC"}
ORDER = ["gpt-5.5", "sarvam-m", "aya-expanse-32b", "tiny-aya-fire", "Llama-3.2-3B-Instruct",
         "BharatGPT-3B-Indic", "cometkiwi-da", "cometkiwi-xl", "xcomet-xl"]


# --------------------------------------------------------------------------- signals and axes
def jl(x):
    """Released list columns ship in two encodings: `wtag` space-separated, `da_scores` as JSON."""
    if x is None or (isinstance(x, float) and x != x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    s = str(x).strip()
    if not s:
        return []
    try:
        j = json.loads(s)
        return list(j) if isinstance(j, (list, tuple)) else []
    except Exception:
        return s.split()


def token_ter(mt, pe) -> float:
    a, b = str(mt).split(), str(pe).split()
    if not a and not b:
        return np.nan
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur, ai = [i] + [0] * n, a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[n] / max(1, n)


def signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _std(s):
        v = [float(x) for x in jl(s) if str(x).strip() not in ("", "nan", "None")]
        return float(np.std(v)) if len(v) > 1 else np.nan
    df["da_std"] = df.da_scores.map(_std)
    has_pe = df.post_edit.notna() & df.post_edit.astype(str).str.strip().ne("")
    df["human_ter"] = [token_ter(m, p) if h else np.nan
                       for m, p, h in zip(df.mt, df.post_edit, has_pe)]

    def bd(w):
        t = [str(x).upper() for x in jl(w)]
        return (sum(x.startswith("BAD") for x in t) / max(1, len(t))) if t else np.nan
    df["bad_density"] = df.wtag.map(bd)
    df["hard_error"] = df.error_categories.map(
        lambda c: any(h in str(c).lower() for h in HARD) if c is not None else False)
    return df


def axis_flags(df: pd.DataFrame, q: float = 0.75) -> pd.DataFrame:
    df = df.copy()

    def topq(c):
        return df.groupby("language_pair")[c].transform(
            lambda s: s >= s.quantile(q) if s.notna().any() else False).fillna(False)

    def z(c):
        return df.groupby("language_pair")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9) if s.notna().any() else s)

    df["A1"] = topq("da_std") & df.da_mean.between(35, 75)
    df["A2"] = topq("human_ter")
    df["A3"] = topq("bad_density") | df.hard_error.astype(bool)
    arm_a = ((z("da_mean") > 0.5) & (z("bad_density") > 0.5)).fillna(False)
    arm_b = ((df.da_mean <= df.groupby("language_pair").da_mean.transform(
        lambda s: s.quantile(0.3))) & (z("human_ter") < -0.5)).fillna(False)
    df["A4"] = arm_a | arm_b
    df["A4_arm"] = np.where(arm_a & arm_b, "both",
                   np.where(arm_a, "scored well, error-dense",
                   np.where(arm_b, "scored badly, barely edited", "")))
    # A pair with no signal for an axis is null on it, never False.
    for a, ok in [("A2", lambda s: s.human_ter.notna().any()),
                  ("A3", lambda s: s.bad_density.notna().any() or bool(s.hard_error.any())),
                  ("A4", lambda s: s.bad_density.notna().any() or s.human_ter.notna().any())]:
        for p, sub in df.groupby("language_pair"):
            if not ok(sub):
                df.loc[df.language_pair == p, a] = pd.NA
    return df


def load_challenge() -> pd.DataFrame:
    full = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SNAP / "full/*.parquet")))],
                     ignore_index=True)
    ch = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SNAP / "challenge/*.parquet")))],
                   ignore_index=True)
    keep = set(ch[ch.has_da.astype("boolean").fillna(False)].uid)
    # flags are computed over the whole corpus so a segment's flag does not depend on its split
    g = axis_flags(signals(full))
    return g[g.uid.isin(keep)].reset_index(drop=True)


def load_preds() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m, d in PROMPTED.items():
        o = {}
        for f in sorted(glob.glob(str(ROOT / "results" / d / f"{m}.s0.*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if "_meta" in r or "_summary" in r or r.get("id") is None:
                    continue
                raw = (r.get("raw") or {}).get("gemba-da")
                v = parse_qe_score_strict(raw) if raw is not None else (r.get("pred") or {}).get("gemba-da")
                if v is not None:
                    o[r["id"]] = float(v)
        out[m] = o
    for m in TRAINED:
        o = {}
        for f in sorted(glob.glob(str(ROOT / "results/qe_metrics_ext" / f"{m}.*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if "_meta" in r or "_summary" in r or r.get("id") is None:
                    continue
                if r.get("pred_score") is not None:
                    o[r["id"]] = float(r["pred_score"])
        out[m] = o
    return out


# --------------------------------------------------------------------------- the contrast
class Panel:
    """Rank correlations and matched controls over one challenge frame."""

    def __init__(self, g: pd.DataFrame, seed: int = SEED):
        self.g = g
        self.da = g.da_mean.to_numpy(float)
        self.rng = np.random.default_rng(seed)

    def rho(self, idx: np.ndarray, y: np.ndarray) -> float:
        v = y[idx]
        k = ~np.isnan(v)
        i = idx[k]
        if len(i) < MIN_N:
            return np.nan
        x = self.da[i]
        if len(np.unique(x)) < 3 or len(np.unique(v[k])) < 3:
            return np.nan
        return float(np.corrcoef(rankdata(x), rankdata(v[k]))[0, 1])

    def controls(self, on: np.ndarray, off: np.ndarray) -> list[np.ndarray]:
        """DRAWS subsamples of `off` reproducing the DA histogram of `on`."""
        bo, bf = np.floor(self.da[on] / BINW), np.floor(self.da[off] / BINW)
        cnt = {b: int((bo == b).sum()) for b in np.unique(bo)}
        pools = {b: off[bf == b] for b in cnt}
        out = []
        for _ in range(DRAWS):
            idx = [self.rng.choice(p, min(cnt[b], len(p)), replace=False)
                   for b, p in pools.items() if len(p)]
            out.append(np.concatenate(idx))
        return out

    def cells(self, preds: dict[str, np.ndarray], systems: list[str]) -> pd.DataFrame:
        rows = []
        for a in AXES:
            for pair, s in self.g.groupby("language_pair"):
                if s[a].isna().all():
                    continue
                sel = s[a].fillna(False).astype(bool).to_numpy()
                on, off = s.index.to_numpy()[sel], s.index.to_numpy()[~sel]
                if len(on) < MIN_N or len(off) < MIN_N:
                    continue
                ctl = self.controls(on, off)
                for m in systems:
                    y = preds[m]
                    r_on, r_off = self.rho(on, y), self.rho(off, y)
                    cs = [c for c in (self.rho(c, y) for c in ctl) if not np.isnan(c)]
                    if np.isnan(r_on) or not cs:
                        continue
                    rows.append({"axis": a, "pair": pair, "system": m, "n_on": int(len(on)),
                                 "n_ctl": int(np.mean([len(c) for c in ctl])), "rho_on": r_on,
                                 "rho_off": r_off, "rho_ctl": float(np.mean(cs)),
                                 "plain": r_on - r_off, "matched": r_on - float(np.mean(cs))})
        return pd.DataFrame(rows)

    def summarise(self, C: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(SEED)
        out = []
        for a in AXES:
            sub = C[C.axis == a]
            if sub.empty:
                continue
            prs = sorted(sub.pair.unique())
            rec = {"axis": a, "pairs": len(prs), "cells": len(sub),
                   "n_on": int(sub.groupby("pair").n_on.first().sum()),
                   "rho_on": sub.rho_on.mean(), "rho_ctl": sub.rho_ctl.mean()}
            for col in ("plain", "matched"):
                by = {p: sub[sub.pair == p][col].to_numpy() for p in prs}
                pt = float(np.mean([np.mean(by[p]) for p in prs]))
                d = np.array([np.mean(np.concatenate(
                    [by[prs[i]] for i in rng.choice(len(prs), len(prs), True)])) for _ in range(BOOT)])
                rec[col] = pt
                rec[f"{col}_lo"], rec[f"{col}_hi"] = np.percentile(d, [2.5, 97.5])
                rec[f"{col}_negpairs"] = int(sum(np.mean(by[p]) < 0 for p in prs))
                rec[f"{col}_negcells"] = int((sub[col] < 0).sum())
            out.append(rec)
        return pd.DataFrame(out)


# --------------------------------------------------------------------------- main
def main() -> int:
    check = "--check" in sys.argv
    g = load_challenge()
    P = load_preds()
    pred = {m: g.uid.map(P[m]).to_numpy(float) for m in ORDER}
    pan = Panel(g)
    OUTD.mkdir(parents=True, exist_ok=True)
    print(f"challenge test, DA-bearing: {len(g)} rows over {g.language_pair.nunique()} pairs\n")

    # ---- coverage and range restriction --------------------------------------------
    cov = []
    for p in PAIRS:
        s = g[g.language_pair == p]
        row = {"pair": p, "n": len(s)}
        for a in AXES:
            row[a] = pd.NA if s[a].isna().all() else int(s[a].fillna(False).sum())
        cov.append(row)
    cov = pd.DataFrame(cov)
    cov.to_csv(OUTD / "axis_coverage.csv", index=False)
    print("axis coverage on the challenge test")
    print(cov.to_string(index=False, na_rep="--"))
    print("\nDA spread of the flagged segments relative to the rest of the pair")
    sdr = {}
    for a in AXES:
        s = g[g[a].notna()]
        on, off = s[s[a].astype(bool)], s[~s[a].astype(bool)]
        sdr[a] = float(on.da_mean.std() / off.da_mean.std())
        print(f"  {a}: sd {on.da_mean.std():5.2f} against {off.da_mean.std():5.2f}, ratio {sdr[a]:.2f}")

    # ---- is the challenge test actually enriched for difficulty? ---------------------
    allrows = axis_flags(signals(pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(SNAP / "full/*.parquet")))],
        ignore_index=True)))
    flagged = allrows[AXES].fillna(False).astype(bool).any(axis=1)
    inch = allrows.uid.isin(set(g.uid))
    da = allrows.has_da.astype("boolean").fillna(False)
    enr_in = float(flagged[inch].mean())
    enr_out = float(flagged[da & ~inch].mean())
    per_pair = [(p_, float(flagged[inch & (allrows.language_pair == p_)].mean()),
                 float(flagged[da & ~inch & (allrows.language_pair == p_)].mean()))
                for p_ in PAIRS]
    n_enriched = sum(1 for _, a, b in per_pair if a > b)
    print(f"\nenrichment: {enr_in:.3f} of the challenge test carries an axis against {enr_out:.3f} "
          f"of the DA-bearing rows outside it; enriched on {n_enriched} of {len(per_pair)} pairs")
    pd.DataFrame(per_pair, columns=["pair", "challenge", "rest"]).to_csv(
        OUTD / "axis_enrichment.csv", index=False)

    # ---- the contrast ---------------------------------------------------------------
    C = pan.cells(pred, ORDER)
    C.to_csv(OUTD / "axis_cells.csv", index=False)
    S = pan.summarise(C)
    S.to_csv(OUTD / "axis_summary.csv", index=False)
    print("\ncontrast against the rest of the pair (plain) and against a DA-matched control")
    print(f"{'axis':5s} {'pairs':>5s} {'flagged':>8s} {'plain':>8s} {'95% CI':>18s} "
          f"{'matched':>8s} {'95% CI':>18s} {'neg pairs':>10s}")
    for _, r in S.iterrows():
        print(f"{r.axis:5s} {int(r.pairs):5d} {int(r.n_on):8,d} {r.plain:>+8.3f} "
              f"{f'[{r.plain_lo:+.3f},{r.plain_hi:+.3f}]':>18s} {r.matched:>+8.3f} "
              f"{f'[{r.matched_lo:+.3f},{r.matched_hi:+.3f}]':>18s} "
              f"{f'{r.matched_negpairs}/{int(r.pairs)}':>10s}")

    # ---- per system -----------------------------------------------------------------
    piv = C.pivot_table(index="system", columns="axis", values="matched", aggfunc="mean").reindex(ORDER)
    pln = C.pivot_table(index="system", columns="axis", values="plain", aggfunc="mean").reindex(ORDER)
    piv.join(pln, lsuffix="_matched", rsuffix="_plain").to_csv(OUTD / "axis_per_system.csv")
    worst = piv.idxmin(axis=1)
    print("\nper system, matched contrast by axis")
    print(f"{'system':24s} " + "".join(f"{a:>9s}" for a in AXES) + "   worst")
    for m in ORDER:
        print(f"{m:24s} " + "".join(f"{piv.loc[m, a]:>+9.3f}" for a in AXES) + f"   {worst[m]}")
    n_a4 = int((piv.A4 < 0).sum())
    print(f"\n  A4 is negative for {n_a4} of {len(ORDER)} systems and is the worst axis for "
          f"{int((worst == 'A4').sum())}")

    # ---- A4 arms ---------------------------------------------------------------------
    print("\nA4 by arm")
    arms = []
    for arm in ("scored badly, barely edited", "scored well, error-dense"):
        pm = {}
        for pair, s in g.groupby("language_pair"):
            if s.A4.isna().all():
                continue
            f = s.A4.fillna(False).astype(bool).to_numpy()
            on = s.index.to_numpy()[f & (s.A4_arm == arm).to_numpy()]
            off = s.index.to_numpy()[~f]
            if len(on) < MIN_N or len(off) < MIN_N:
                continue
            ctl = pan.controls(on, off)
            d = []
            for m in ORDER:
                r_on = pan.rho(on, pred[m])
                cs = [c for c in (pan.rho(c, pred[m]) for c in ctl) if not np.isnan(c)]
                if cs and not np.isnan(r_on):
                    d.append(r_on - float(np.mean(cs)))
            if d:
                pm[pair] = (len(on), float(np.mean(d)))
        for p, (n, v) in pm.items():
            arms.append({"arm": arm, "pair": p, "n": n, "matched": v})
        if pm:
            arms.append({"arm": arm, "pair": "all", "n": sum(n for n, _ in pm.values()),
                         "matched": float(np.mean([v for _, v in pm.values()]))})
        if pm:
            print(f"  {arm:30s} {np.mean([v for _, v in pm.values()]):+.3f} over {len(pm)} pairs, "
                  f"{sum(n for n, _ in pm.values()):,} segments, negative in "
                  f"{sum(1 for _, v in pm.values() if v < 0)}/{len(pm)}   "
                  + " ".join(f"{p}:{v:+.2f}(n={n})" for p, (n, v) in pm.items()))
    pd.DataFrame(arms).to_csv(OUTD / "axis_a4_arms.csv", index=False)

    # ---- overlap ---------------------------------------------------------------------
    n_ax = g[AXES].fillna(False).astype(bool).sum(axis=1)
    ov = [{"quantity": f"segments on exactly {k} axes", "n": int(v)}
          for k, v in n_ax.value_counts().sort_index().items()]
    ov += [{"quantity": "segments on at least one axis", "n": int((n_ax >= 1).sum())},
           {"quantity": "segments on no axis", "n": int((n_ax == 0).sum())},
           {"quantity": "DA-bearing challenge segments", "n": len(g)}]
    print(f"\naxes per segment {n_ax.value_counts().sort_index().to_dict()}, "
          f"at least one: {int((n_ax >= 1).sum())} of {len(g)}")
    for i, a in enumerate(AXES):
        for b in AXES[i + 1:]:
            m = g[g[a].notna() & g[b].notna()]
            u = int((m[a].astype(bool) | m[b].astype(bool)).sum())
            j = int((m[a].astype(bool) & m[b].astype(bool)).sum()) / u
            ov.append({"quantity": f"Jaccard {a}/{b}", "n": round(j, 4)})
            print(f"  Jaccard {a}/{b}: {j:.2f}")
    pd.DataFrame(ov).to_csv(OUTD / "axis_overlap.csv", index=False)

    # domain composition of the same population, so the appendix table is traceable
    dom = g.domain.value_counts().rename_axis("domain").reset_index(name="n")
    dom["share"] = (100 * dom.n / len(g)).round(1)
    dom.loc[len(dom)] = ["total", len(g), 100.0]
    dom.to_csv(OUTD / "challenge_domains.csv", index=False)

    # ---- splice ----------------------------------------------------------------------
    body = "\n".join(
        MACRO[m] + " & " + " & ".join(
            (f"\\best{{${piv.loc[m, a]:+.3f}$}}" if a == worst[m] else f"${piv.loc[m, a]:+.3f}$")
            for a in AXES) + r" \\" for m in ORDER) + "\n"
    covbody = "\n".join(
        f"{r.pair} & {r.n:,} & ".replace(",", "{,}") + " & ".join(
            ("---" if pd.isna(r[a]) else f"{int(r[a]):,}".replace(",", "{,}")) for a in AXES) + r" \\"
        for _, r in cov.iterrows()) + "\n"
    fullbody = "\n".join(
        r.axis + f" & {int(r.pairs)} & {int(r.n_on):,} & ${r.rho_on:.3f}$ & ${r.rho_ctl:.3f}$ & "
        f"${r.plain:+.3f}$ & ${r.matched:+.3f}$ \\tiny{{[{r.matched_lo:+.3f}, {r.matched_hi:+.3f}]}}"
        f" & {r.matched_negpairs}/{int(r.pairs)}" + r" \\"
        for _, r in S.iterrows()).replace(",", "{,}") + "\n"

    pp = C.pivot_table(index="pair", columns="axis", values="matched", aggfunc="mean")
    ppn = C.pivot_table(index="pair", columns="axis", values="n_on", aggfunc="first")
    pp = pp.reindex([p for p in PAIRS if p in pp.index])
    pp.join(ppn, lsuffix="_matched", rsuffix="_n").to_csv(OUTD / "axis_per_pair.csv")
    print("\nper pair, matched contrast by axis")
    print(pp.to_string(float_format=lambda v: f"{v:+.3f}", na_rep="   ---"))
    for a in AXES:
        v = pp[a].dropna()
        loo = {p: float(v.drop(p).mean()) for p in v.index}
        print(f"  {a} leave-one-pair-out: {min(loo.values()):+.3f} to {max(loo.values()):+.3f}"
              f"   (largest single pair {min(loo, key=loo.get)} -> {max(loo.values()):+.3f})")
    ppbody = "\n".join(
        p + " & " + " & ".join(
            (f"${int(ppn.loc[p, a]):,}$ & ${pp.loc[p, a]:+.3f}$".replace(",", "{,}")
             if a in pp.columns and not pd.isna(pp.loc[p, a]) else "--- & ---")
            for a in AXES) + r" \\" for p in pp.index) + "\n"

    # the numeric cut each pair receives, computed over the whole corpus as the flags are
    full = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SNAP / "full/*.parquet")))],
                     ignore_index=True)
    fs = signals(full)
    thr = []
    for p in PAIRS:
        s = fs[fs.language_pair == p]
        def q(c, k):
            v = s[c].quantile(k)
            return "---" if pd.isna(v) else f"${v:.2f}$"

        def zc(c, k):
            m, sd = s[c].mean(), s[c].std(ddof=0)
            return "---" if pd.isna(m) else f"${m + k * sd:.2f}$"
        thr.append(f"{p} & {q('da_std', .75)} & {q('human_ter', .75)} & {q('bad_density', .75)} & "
                   f"{zc('da_mean', .5)} & {zc('bad_density', .5)} & {q('da_mean', .30)} & "
                   f"{zc('human_ter', -.5)}" + r" \\")
    thrbody = "\n".join(thr) + "\n"

    txt = TEX.read_text()
    want = txt
    for tag, b in [("tab_axiscontrast", body), ("tab_axiscoverage", covbody),
                   ("tab_axissummary", fullbody), ("tab_axisthresholds", thrbody),
                   ("tab_axisperpair", ppbody)]:
        pat = re.compile(rf"(% BEGIN GENERATED {tag}[^\n]*\n).*?(% END GENERATED {tag})", re.S)
        if not pat.search(want):
            print(f"note: {tag} sentinels are not in the tex")
            continue
        want = pat.sub(lambda m: m.group(1) + b + m.group(2), want)

    fails = []
    if want != txt:
        fails.append("a generated axis table differs from the data")
    flat = " ".join(want.split())
    W = {6: "six", 7: "seven", 8: "eight", 9: "nine"}
    a4 = S[S.axis == "A4"].iloc[0]
    a1 = S[S.axis == "A1"].iloc[0]

    # Every figure the prose quotes, recomputed here. The tex is checked for the string, so a
    # number that drifts in the data fails the check rather than sitting wrong in the paper.
    v4 = pp["A4"].dropna()
    loo = {p: float(v4.drop(p).mean()) for p in v4.index}
    big = max(loo, key=loo.get)     # the pair whose removal pulls the mean furthest towards zero
    others = [v for p, v in loo.items() if p != big]
    en_hi30 = fs[fs.language_pair == "en-hi"].da_mean.quantile(.30)
    lo30 = fs.groupby("language_pair").da_mean.quantile(.30).min()
    lo30_pair = fs.groupby("language_pair").da_mean.quantile(.30).idxmin()
    ml = fs[(fs.language_pair == "en-ml") & fs.human_ter.notna()]
    ml_cut = ml.human_ter.mean() - 0.5 * ml.human_ter.std(ddof=0)
    ta = fs[(fs.language_pair == "en-ta") & fs.human_ter.notna()]
    ta_cut = ta.human_ter.mean() - 0.5 * ta.human_ter.std(ddof=0)

    want_strings = [
        (f"${a4.matched:.3f}$", "A4 matched contrast"),
        (f"${a1.matched:.3f}$".replace("-0.00", "0.00"), "A1 matched contrast"),
        (f"negative for all {W[len(ORDER)]}", "A4 system count"),
        (f"${v4[big]:.3f}$ on {big}", "the largest single A4 pair"),
        (f"from ${v4.mean():.3f}$ to ${loo[big]:.3f}$", "A4 leave-one-out"),
        (f"between ${max(others):.3f}$ and ${min(others):.3f}$", "A4 leave-one-out bounds"),
        (f"is ${en_hi30:.1f}$ for en-hi and ${lo30:.1f}$ for {lo30_pair}", "DA p30 spread"),
        (f"that cut falls at ${ml_cut:.2f}$", "the en-ml low-edit cut"),
        (f"${100 * (ml.mt.astype(str) == ml.post_edit.astype(str)).mean():.1f}\\%$ of its post-edits",
         "en-ml zero-edit share, exact equality as Appendix L reports it"),
        (f"a mean of ${ml.human_ter.mean():.3f}$", "en-ml TER mean"),
        (f"deviation of ${ml.human_ter.std(ddof=0):.3f}$", "en-ml TER sd"),
        (f"it falls at ${ta_cut:.2f}$", "the en-ta low-edit cut"),
        (f"${100 * enr_in:.1f}\\%$ of DA-bearing challenge segments", "challenge enrichment"),
        (f"against ${100 * enr_out:.1f}\\%$ of the DA-bearing segments", "enrichment vs the rest"),
        (f"every one of the {W[len(per_pair)]} pairs is", "enriched on every pair")
        if n_enriched == len(per_pair) else ("", "unreachable"),
    ]
    for s, what in want_strings:
        if " ".join(s.split()) not in flat:
            fails.append(f"{what}: the tex does not contain {s!r}")
    if check:
        for f in fails:
            print(f"FAIL: {f}")
        return 1 if fails else 0
    if want != txt:
        TEX.write_text(want)
        print("\nspliced the axis tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
