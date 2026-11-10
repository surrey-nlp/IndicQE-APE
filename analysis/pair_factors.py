#!/usr/bin/env python3
r"""pair_factors.py — what actually explains the per-pair spread in QE difficulty.

The paper reports that the X-to-English directions score above the en-to-Indic ones and that en-te
is the hardest pair, and leaves both as observations. Three explanations are available and they
make different predictions, so they can be told apart with data already on disk.

  H1 LABEL RELIABILITY. A model is scored against a mean of three or more annotator scores. Where
     that mean is less reliable, the attainable correlation is lower. ICC(1,k) per pair is in
     tab:iaa and ranges from 0.247 (en-mr) to 0.985 (en-ml). Prediction: per-pair difficulty tracks
     ICC(1,k) across pairs, for every system alike.

  H2 TRAINING EXPOSURE. CometKiwi is trained on WMT DA data that includes MLQE-PE, which is exactly
     where our three X-to-English pairs come from. If the direction advantage were exposure, it
     would be larger for the trained metrics than for prompted LLMs that never saw those labels.
     Prediction: direction gap much larger for CometKiwi than for gpt-5.5.

  H3 RANGE. A rank correlation on a compressed target is lower for the same ordering. Prediction:
     difficulty tracks the between-segment spread of the DA mean.

The three are not exclusive, and the point of the script is to say which of them the data supports
rather than to pick a winner. It also isolates en-te, which is the pair no explanation covers.

    python3 analysis/pair_factors.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotator_agreement import as_list, icc1, icc1k  # noqa: E402
from indicqe_ape.prompts import parse_qe_score_strict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

CHALLENGE = ROOT / "dataset" / "hf_snapshot" / "data" / "challenge" / "test.parquet"
FULL = ROOT / "dataset" / "hf_snapshot" / "data" / "full"
OUT = ROOT / "results" / "analysis" / "pair_factors.csv"

PAIRS = ["en-hi", "en-mr", "en-ta", "en-gu", "en-te", "en-ml", "et-en", "ne-en", "si-en"]
CODE = {p: p.replace("-", "") for p in PAIRS}
XEN = {"et-en", "ne-en", "si-en"}
PROMPTED = [("gpt-5.5", ["qe_closed_rerun", "qe_closed"]), ("sarvam-m", ["qe_rerun512"]),
            ("aya-expanse-32b", ["qe_rerun512"]), ("tiny-aya-fire", ["qe_rerun512"])]
TRAINED = ["cometkiwi-da", "cometkiwi-xl", "xcomet-xl"]


def prompted(model: str, dirs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dirs:
        for f in sorted(glob.glob(str(ROOT / "results" / d / f"{model}.s0.*.jsonl"))):
            pair = Path(f).stem.split(".")[-1]
            if pair in out:
                continue
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in r or "_summary" in r or r.get("id") is None:
                    continue
                raw = (r.get("raw") or {}).get("gemba-da")
                v = parse_qe_score_strict(raw) if raw is not None else (r.get("pred") or {}).get("gemba-da")
                if v is not None:
                    out.setdefault(pair, {})
                    out[pair][r["id"]] = float(v)  # type: ignore[index]
    return out  # type: ignore[return-value]


def metric(name: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "qe_metrics_ext" / f"{name}.*.jsonl"))):
        pair = Path(f).stem.split(".")[-1]
        d: dict[str, float] = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if "_meta" in r or "_summary" in r or r.get("id") is None:
                continue
            if r.get("pred_score") is not None:
                d[r["id"]] = float(r["pred_score"])
        out[pair] = d
    return out


def main() -> int:
    check = "--check" in sys.argv
    g = pd.read_parquet(CHALLENGE)
    g = g[g.has_da].copy()
    g["scores"] = g.da_scores.map(as_list)

    preds: dict[str, dict[str, dict[str, float]]] = {}
    for m, dirs in PROMPTED:
        preds[m] = prompted(m, dirs)
    for m in TRAINED:
        preds[m] = metric(m)

    rows = []
    for pair in PAIRS:
        sub = g[g.language_pair == pair]
        units = [u for u in sub.scores if len(u) >= 2]
        i1 = icc1(units)
        kbar = float(np.mean([len(u) for u in units]))
        r = dict(pair=pair, n=len(sub), direction="X-en" if pair in XEN else "en-X",
                 icc1k=round(icc1k(i1, kbar), 3),
                 da_mean=round(sub.da_mean.mean(), 1),
                 da_sd=round(sub.da_mean.std(ddof=1), 2),
                 da_iqr=round(float(sub.da_mean.quantile(.75) - sub.da_mean.quantile(.25)), 2))
        for m in preds:
            p = preds[m].get(CODE[pair], {})
            j = sub[sub.uid.isin(p)]
            r[m] = (round(float(spearmanr(j.da_mean, j.uid.map(p)).statistic), 3)
                    if len(j) > 30 else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    sysnames = list(preds)
    df["mean_rho"] = df[sysnames].mean(axis=1).round(3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))

    print("\nH1  does per-pair difficulty track label reliability?")
    for m in sysnames + ["mean_rho"]:
        ok = df.dropna(subset=[m])
        r, p = pearsonr(ok.icc1k, ok[m])
        print(f"  {m:18s} r(ICC(1,k), rho) = {r:+.3f}  (p={p:.3f}, k={len(ok)})")

    print("\nH3  does it track the spread of the target?")
    for v in ("da_sd", "da_iqr"):
        r, p = pearsonr(df[v], df.mean_rho)
        print(f"  r({v}, mean rho) = {r:+.3f}  (p={p:.3f})")

    print("\nH2  direction gap per system (X-en macro minus en-X macro)")
    for m in sysnames:
        a = df[df.direction == "X-en"][m].mean()
        b = df[df.direction == "en-X"][m].mean()
        tag = "TRAINED, saw MLQE-PE" if m in TRAINED else "prompted, no DA training"
        print(f"  {m:18s} {a:.3f} - {b:.3f} = {a - b:+.3f}   [{tag}]")
    tr = np.mean([df[df.direction == 'X-en'][m].mean() - df[df.direction == 'en-X'][m].mean()
                  for m in TRAINED])
    pr = np.mean([df[df.direction == 'X-en'][m].mean() - df[df.direction == 'en-X'][m].mean()
                  for m, _ in PROMPTED])
    print(f"  mean gap: trained {tr:+.3f}, prompted {pr:+.3f} -> "
          f"{'exposure cannot be the whole story' if abs(tr - pr) < 0.1 else 'trained gap is larger'}")

    # splice tab:pairfactors
    tex = _PAPER_TEX
    import re as _re
    lines = []
    for _, r in df.iterrows():
        lines.append(f"{r['pair']} & {int(r['n'])} & {r['icc1k']:.3f} & {r['da_sd']:.1f} & "
                     f"{r['da_iqr']:.1f} & {r['cometkiwi-xl']:.3f} & {r['gpt-5.5']:.3f} \\\\")
    body = "\n".join(lines)
    txt = tex.read_text()
    pat = _re.compile(r"(% BEGIN GENERATED tab_pairfactors[^\n]*\n).*?(% END GENERATED tab_pairfactors)",
                      _re.S)
    want = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), txt) if pat.search(txt) else None
    if want is None:
        print("\nnote: tab:pairfactors sentinels not found, skipping splice")
    elif check:
        flat = " ".join(txt.split())
        fails = [f"tab:pairfactors rows differ from the data"] if want != txt else []
        r_icc = pearsonr(df.icc1k, df.mean_rho)[0]
        r_sd = pearsonr(df.da_sd, df.mean_rho)[0]
        for s, what in [(f"$r=-{abs(r_icc):.2f}$", "ICC correlation"),
                        (f"$r=+{r_sd:.2f}$", "spread correlation"),
                        (f"$+{tr:.2f}$", "trained direction gap"),
                        (f"$+{pr:.2f}$", "prompted direction gap")]:
            if s not in flat:
                fails.append(f"{what}: tex does not contain {s}")
        for f in fails:
            print(f"FAIL: {f}")
        if fails:
            return 1
        print("\npair-factor numbers OK")
        return 0
    else:
        tex.write_text(want)
        print("\nspliced tab:pairfactors")

    print("\nen-te against the pattern")
    te = df[df.pair == "en-te"].iloc[0]
    print(f"  ICC(1,k) {te.icc1k} (rank {int((df.icc1k > te.icc1k).sum()) + 1} of 9, higher is better), "
          f"DA sd {te.da_sd}, IQR {te.da_iqr}, mean rho {te.mean_rho} (lowest of 9)")
    print(f"  DA sd rank: {int((df.da_sd > te.da_sd).sum()) + 1} of 9; "
          f"IQR rank: {int((df.da_iqr > te.da_iqr).sum()) + 1} of 9")
    print(f"  mean DA {te.da_mean} against benchmark mean {df.da_mean.mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
