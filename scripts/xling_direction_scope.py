#!/usr/bin/env python3
"""Cross-lingual coefficients, pooled over all pairs vs scoped to en->Indic.

Why this exists
---------------
The cross-lingual score in tab:qe is the Pearson correlation, across language pairs,
between a model's mean score for a pair and the mean human score for that pair. It is
estimated from one point per pair.

Pooling en->Indic with X->en puts two clusters on that scatter: X->en carries mean DA
37.8-64.9 and en->Indic 67.9-83.6. A model that does nothing more than notice X->en is
worse therefore earns a large positive coefficient for free. This script measures how
much of each published coefficient is that artefact.

Result (see results/analysis/xling_direction_scope.csv): most of it, for most models.

Outputs
-------
results/analysis/xling_direction_scope.csv

Usage
-----
    PYTHONPATH=.:scripts python3 scripts/xling_direction_scope.py
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
from statistics import mean

import pandas as pd
from scipy.stats import pearsonr

EN_INDIC = ["enhi", "enmr", "enta", "engu", "ente", "enml"]
X_EN = ["eten", "neen", "sien"]
ALL = EN_INDIC + X_EN

# GEMBA-DA prompted systems, 0-shot, on the extended challenge test.
LLM_SYSTEMS = {
    "BharatGPT-3B": ("results/qe_open", "BharatGPT-3B-Indic.s0"),
    "Llama-3.2-3B": ("results/qe_open", "Llama-3.2-3B-Instruct.s0"),
    "tiny-aya": ("results/qe_open", "tiny-aya-fire.s0"),
    "sarvam-m": ("results/qe_open", "sarvam-m.s0"),
    "gpt-5.5": ("results/qe_closed", "gpt-5.5.s0"),
    "aya-expanse-32B": ("results/qe_open", "aya-expanse-32b.s0"),
}

# Trained metrics on the FULL population (73,829 rows). Reported separately because the
# population differs from the LLM rows; do not put these two blocks in one table without
# saying so.
METRIC_SYSTEMS = {
    "CometKiwi-XL": ("results/qe_metrics_full", "cometkiwi-xl-reffree", False),
    "XCOMET-XL (ref-free)": ("results/qe_metrics_full", "xcomet-xl-reffree", False),
    "MetricX-24 XL (QE)": ("results/qe_metrics_full", "metricx24-xl-qe", True),
    "MetricX-24 XL (ref)": ("results/qe_metrics_full", "metricx24-xl-refbased", True),
}

BOOTSTRAP = 20_000
random.seed(0)


def _rows(path: str):
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_meta" in obj and len(obj) == 1:
            continue
        yield obj


def _gold(o):
    try:
        v = float(o.get("gold_da_mean"))
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _llm_pred(o):
    p = o.get("pred")
    if isinstance(p, str):
        try:
            p = ast.literal_eval(p)
        except Exception:
            return None
    if isinstance(p, dict):
        p = p.get("gemba-da")
    try:
        v = float(p)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _metric_pred(o):
    try:
        v = float(o.get("pred_score"))
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def boot_ci(xs, ys, b=BOOTSTRAP):
    n = len(xs)
    out = []
    for _ in range(b):
        idx = [random.randrange(n) for _ in range(n)]
        a = [xs[i] for i in idx]
        c = [ys[i] for i in idx]
        if len(set(a)) < 3 or len(set(c)) < 3:
            continue
        out.append(pearsonr(a, c)[0])
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def pair_means(directory: str, stem: str, predfn) -> dict[str, tuple[float, float]]:
    means = {}
    for pair in ALL:
        path = os.path.join(directory, f"{stem}.{pair}.jsonl")
        if not os.path.exists(path):
            continue
        pts = []
        for o in _rows(path):
            g, p = _gold(o), predfn(o)
            if g is not None and p is not None:
                pts.append((g, p))
        if len(pts) >= 10:
            means[pair] = (mean([g for g, _ in pts]), mean([p for _, p in pts]))
    return means


def coefficient(means, scope):
    sel = [means[p] for p in scope if p in means]
    if len(sel) < 4:
        return None
    a = [x for x, _ in sel]
    b = [y for _, y in sel]
    r = pearsonr(a, b)[0]
    lo, hi = boot_ci(a, b)
    return {"k": len(sel), "r": r, "lo": lo, "hi": hi}


def main() -> None:
    records = []
    blocks = [
        ("LLM (GEMBA-DA, extended challenge)",
         {k: (d, s, _llm_pred) for k, (d, s) in LLM_SYSTEMS.items()}),
        ("trained metric (full population)",
         {k: (d, s, _metric_pred) for k, (d, s, _) in METRIC_SYSTEMS.items()}),
    ]
    for block, systems in blocks:
        for name, (directory, stem, predfn) in systems.items():
            means = pair_means(directory, stem, predfn)
            pooled = coefficient(means, ALL)
            scoped = coefficient(means, EN_INDIC)
            if pooled is None or scoped is None:
                continue
            records.append({
                "block": block,
                "system": name,
                "k_pooled": pooled["k"],
                "r_pooled": round(pooled["r"], 4),
                "k_scoped": scoped["k"],
                "r_scoped": round(scoped["r"], 4),
                "scoped_ci_lo": round(scoped["lo"], 4),
                "scoped_ci_hi": round(scoped["hi"], 4),
                "scoped_ci_excludes_zero": bool(scoped["lo"] * scoped["hi"] > 0),
                "shift": round(scoped["r"] - pooled["r"], 4),
            })

    df = pd.DataFrame(records)
    out = "results/analysis/xling_direction_scope.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten: {out}")
    n_id = int(df["scoped_ci_excludes_zero"].sum())
    print(f"\nsystems whose en->Indic coefficient is identified "
          f"(bootstrap CI excludes zero): {n_id} of {len(df)}")


if __name__ == "__main__":
    main()
