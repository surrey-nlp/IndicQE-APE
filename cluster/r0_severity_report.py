#!/usr/bin/env python3
"""R0 step 3 (CPU) — the report. Char-matched pairwise accuracy (Eq. pacm) for every metric on the
identical injected severity set, plus the interpretability breakdowns that let a MATRA number be
explained rather than merely quoted.

Generalises scripts/overnight/severity_pa.py, which handled four hardcoded columns. Emits:

  1. Headline PA table            -- every metric, one row, with n and a binomial p vs chance
  2. PA by language               -- isolates the known Tamil defect
  3. PA by MQM gap                -- does the metric only win on easy (wide-gap) pairs?
  4. Per-category mean scores     -- which error types each metric under/over-penalises
  5. Disagreement examples        -- concrete pairs where MATRA and the trained metrics differ

DIRECTION MATTERS. Metrics fall into two camps and mixing them silently inverts a result:
  * error/severity scores (higher = worse): char_ter, all matra*, MetricX prediction
  * quality scores       (higher = better): XCOMET-XL, CometKiwi-XL
Quality scores are negated below so every column reads "higher = more severe" before comparison.
"""
from __future__ import annotations
import json, os, sys, collections, math

SET_F = os.environ.get("R0_SET", "results/severity/r0_severity_input.jsonl")
GPU_F = sys.argv[1] if len(sys.argv) > 1 else "results/severity/r0_gpu_scores.jsonl"
OUT_MD = os.environ.get("R0_REPORT", "results/severity/r0_severity_report.md")
SRC = f"{SET_F} + {GPU_F}"


def load_joined():
    """Join the locally-built set (MATRA, MQM, base_id) with the GPU scores brought back from
    the GPU cluster. Joined on `id`; the GPU side never saw the MATRA columns or the severity prior."""
    rows = [json.loads(l) for l in open(SET_F, encoding="utf-8")]
    if not os.path.exists(GPU_F):
        print(f"[warn] {GPU_F} not found — reporting CPU metrics only")
        return rows
    # Merge every GPU score file present (R0 = COMET family, R4 = MetricX), all keyed on id.
    gpu = {}
    extras = [os.environ.get("R0_GPU_EXTRA", "results/severity/r4_metricx_scores.jsonl"),
              os.environ.get("R0_BASE", "results/severity/r0_baseline_scores.jsonl")]
    for path in [GPU_F] + [e for e in extras if os.path.exists(e)]:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            gpu.setdefault(r["id"], {}).update(r)
        print(f"[join] read {path}")
    hit = 0
    for r in rows:
        g = gpu.get(r.get("id"))
        if g:
            hit += 1
            for k, v in g.items():
                if k != "id":
                    r[k] = v
    if hit != len(rows):
        print(f"[warn] joined {hit}/{len(rows)} records — ids do not fully match, "
              f"check the GPU input was built from THIS set")
    else:
        print(f"[join] {hit}/{len(rows)} records matched on id")
    return rows

# display name -> (field, higher_is_more_severe)
METRICS = collections.OrderedDict([
    ("char-TER",            ("char_ter",            True)),
    # surface / edit-distance baselines (cluster/r0_baselines.py). error-direction first.
    ("word-TER",            ("word_ter",            True)),
    ("CharacTER",           ("character",           True)),
    ("akshara-count",       ("aksh_count",          True)),
    ("chrF",                ("chrf",                False)),
    ("chrF++",              ("chrfpp",              False)),
    ("spBLEU",              ("spbleu",              False)),
    ("MATRA (published)",   ("matra",               True)),
    # Every MATRA++ row is named by VERSION, never by "default". The set carries a bare `matrapp`
    # column recording whatever the default was when it was built (2026-07-19 21:14, i.e. v0.5);
    # the default moved to v0.6 at 23:16 the same evening. A row labelled "default" reading that
    # column silently reported v0.5 as the shipping metric -- 882 of 6,372 values differ, by up to
    # 30 points. `matrapp` is therefore not read at all; ask for a version explicitly.
    ("MATRA++ v0.3",        ("matrapp-v0.3",        True)),
    ("MATRA++ v0.4",        ("matrapp-v0.4",        True)),
    ("MATRA++ v0.5",        ("matrapp-v0.5",        True)),
    ("MATRA++ v0.6 (ships)", ("matrapp-v0.6",       True)),
    ("XCOMET-XL (ref-free)", ("xcomet_xl_reffree",  False)),
    ("XCOMET-XL (ref)",     ("xcomet_xl_refbased",  False)),
    ("CometKiwi-XL",        ("cometkiwi_xl",        False)),
    # MetricX emits an ERROR score (0-25, lower = better) -- same direction as MATRA, opposite
    # to XCOMET/CometKiwi. XL and XXL both present so that no failure can be attributed to
    # model capacity.
    ("MetricX-24 XL (QE)",   ("metricx24_xl_qe",    True)),
    ("MetricX-24 XL (ref)",  ("metricx24_xl_ref",   True)),
    ("MetricX-24 XXL (QE)",  ("metricx24_xxl_qe",   True)),
    ("MetricX-24 XXL (ref)", ("metricx24_xxl_ref",  True)),
    ("MetricX-24 (legacy)",  ("prediction",         True)),
])


def val(r, field, higher_severe):
    v = r.get(field)
    if v is None:
        return None
    return float(v) if higher_severe else -float(v)


def pa(rows, field, higher_severe, keep=lambda a, b: True):
    """Char-matched pairwise accuracy. Same-base pairs, different MQM, |dchar-TER| <= 3."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["base_id"]].append(r)
    corr = tot = 0
    for rs in by.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["mqm"] == b["mqm"] or not keep(a, b):
                    continue
                if abs(a["char_ter"] - b["char_ter"]) > 3:
                    continue
                va, vb = val(a, field, higher_severe), val(b, field, higher_severe)
                if va is None or vb is None:
                    continue
                hi, lo = (va, vb) if a["mqm"] > b["mqm"] else (vb, va)
                if hi == lo:
                    continue
                tot += 1
                corr += int(hi > lo)
    return (corr / tot if tot else float("nan")), tot


def binom_p(k, n, p=0.5):
    """Two-sided binomial tail vs chance.

    Exact for small n; normal approximation with continuity correction above 1000, because the
    exact form overflows float via math.comb at the pair counts this set produces (tens of
    thousands). The approximation is more than adequate there — at n>1000 the two agree to far
    more precision than the third decimal we report."""
    if n == 0:
        return float("nan")
    if n <= 1000:
        obs = abs(k - n * p)
        tot = 0.0
        for i in range(n + 1):
            if abs(i - n * p) >= obs - 1e-9:
                tot += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        return min(1.0, tot)
    mu, sd = n * p, math.sqrt(n * p * (1 - p))
    z = (abs(k - mu) - 0.5) / sd                     # continuity correction
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def main():
    rows = load_joined()
    present = [(d, f, h) for d, (f, h) in METRICS.items() if any(f in r for r in rows)]
    missing = [d for d, (f, _) in METRICS.items() if not any(f in r for r in rows)]
    langs = sorted({r["lang"] for r in rows})
    L = []

    L.append(f"# R0 — severity ordering on the injected char-matched set\n")
    L.append(f"`{SRC}` — **{len(rows)} records**, {len(set(r['base_id'] for r in rows))} bases, "
             f"languages {langs}.\n")
    if missing:
        L.append(f"> Not scored in this file: {', '.join(missing)}.\n")
    L.append("\n## 1. Headline — char-matched pairwise accuracy (0.5 = chance)\n")
    L.append("Same-base pairs, differing MQM, |Δchar-TER| ≤ 3, so length and surface edit distance "
             "cannot explain the ordering — only severity can.\n")
    L.append("\n| metric | PA | n pairs | p vs chance |\n|---|---:|---:|---:|")
    for disp, f, h in present:
        p, n = pa(rows, f, h)
        pv = binom_p(round(p * n), n) if n else float("nan")
        star = " ***" if pv < 0.001 else (" **" if pv < 0.01 else (" *" if pv < 0.05 else ""))
        L.append(f"| {disp} | **{p:.3f}**{star} | {n} | {pv:.2e} |")

    L.append("\n## 2. By language — isolates the Tamil defect\n")
    L.append("\n| metric | " + " | ".join(langs) + " |\n|---" + "|---:" * len(langs) + "|")
    for disp, f, h in present:
        cells = []
        for lg in langs:
            p, n = pa([r for r in rows if r["lang"] == lg], f, h)
            cells.append(f"{p:.3f} ({n})")
        L.append(f"| {disp} | " + " | ".join(cells) + " |")

    L.append("\n## 3. By MQM gap — is the win only on easy pairs?\n")
    L.append("A metric that only separates wide-gap pairs is not doing severity ordering, it is "
             "doing coarse quality detection.\n")
    bands = [("narrow (1-2)", 1, 2), ("mid (3-5)", 3, 5), ("wide (6+)", 6, 99)]
    L.append("\n| metric | " + " | ".join(b[0] for b in bands) + " |\n|---" + "|---:" * len(bands) + "|")
    for disp, f, h in present:
        cells = []
        for _, lo, hi in bands:
            p, n = pa(rows, f, h, keep=lambda a, b, lo=lo, hi=hi: lo <= abs(a["mqm"] - b["mqm"]) <= hi)
            cells.append(f"{p:.3f} ({n})")
        L.append(f"| {disp} | " + " | ".join(cells) + " |")

    L.append("\n## 4. Per-category mean score — what each metric under-penalises\n")
    cats = sorted({r["cat"] for r in rows}, key=lambda c: -max(r["mqm"] for r in rows if r["cat"] == c))
    L.append("\n| category | MQM | n | " + " | ".join(d for d, _, _ in present) + " |")
    L.append("|---|---:|---:|" + "---:|" * len(present))
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        mqm = sub[0]["mqm"]
        cells = []
        for _, f, h in present:
            vs = [val(r, f, h) for r in sub if val(r, f, h) is not None]
            cells.append(f"{sum(vs)/len(vs):.2f}" if vs else "—")
        L.append(f"| `{c}` | {mqm} | {len(sub)} | " + " | ".join(cells) + " |")

    L.append("\n## 5. Interpretability — edit structure by category\n")
    L.append("Carried from the builder so a severity call can be explained without re-running the "
             "metric. Reported per configuration: the unsuffixed fields are published MATRA, the "
             "`_pp` fields are the shipping MATRA++ v0.6. Criticality is the count of edits typed "
             "critical, which is reported **beside** the rate and not folded into it.\n")
    L.append("\n| category | MQM | edits (MATRA) | edits (v0.6) | crit/seg (MATRA) | "
             "crit/seg (v0.6) | fires (v0.6) | typical types (v0.6) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for c in cats:
        sub = [r for r in rows if r["cat"] == c and r.get("n_edits") is not None]
        if not sub:
            continue
        n = len(sub)
        ne = sum(r["n_edits"] for r in sub) / n
        ne_pp = sum(r.get("n_edits_pp") or 0 for r in sub) / n
        nc = sum(r.get("critical") or 0 for r in sub) / n
        nc_pp = sum(r.get("critical_pp") or 0 for r in sub) / n
        fires = 100.0 * sum(1 for r in sub if r.get("critical_pp")) / n
        top = collections.Counter(r.get("edit_types_pp") or "—" for r in sub).most_common(1)[0][0]
        L.append(f"| `{c}` | {sub[0]['mqm']} | {ne:.1f} | {ne_pp:.1f} | {nc:.2f} | {nc_pp:.2f} "
                 f"| {fires:.1f}% | `{top}` |")
    L.append("\nCriticality is a MATRA++ feature: MATRA runs with `number_critical=False`, so its "
             "column is 0 by construction and is shown only to make that explicit.\n")

    # concrete disagreements: MATRA right, best trained metric wrong
    trained = next((f for d, f, h in present if f.startswith("xcomet")), None)
    if trained:
        L.append(f"\n## 6. Disagreements — MATRA++ orders correctly, `{trained}` does not\n")
        by = collections.defaultdict(list)
        for r in rows:
            by[r["base_id"]].append(r)
        shown = 0
        for rs in by.values():
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    a, b = rs[i], rs[j]
                    if a["mqm"] == b["mqm"] or abs(a["char_ter"] - b["char_ter"]) > 3:
                        continue
                    hi, lo = (a, b) if a["mqm"] > b["mqm"] else (b, a)
                    m_hi, m_lo = val(hi, "matrapp-v0.6", True), val(lo, "matrapp-v0.6", True)
                    t_hi, t_lo = val(hi, trained, False), val(lo, trained, False)
                    if None in (m_hi, m_lo, t_hi, t_lo):
                        continue
                    if m_hi > m_lo and t_hi < t_lo and shown < 8:
                        L.append(f"\n**{hi['lang']} / base `{hi['base_id']}`** — "
                                 f"`{hi['cat']}` (MQM {hi['mqm']}) vs `{lo['cat']}` (MQM {lo['mqm']})\n")
                        L.append(f"- more severe: `{hi['hypothesis'][:90]}`")
                        L.append(f"- less severe: `{lo['hypothesis'][:90]}`")
                        L.append(f"- MATRA++ {m_hi:.2f} vs {m_lo:.2f} ✓ · "
                                 f"{trained} {-t_hi:.4f} vs {-t_lo:.4f} ✗")
                        shown += 1

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[write] {OUT_MD}")


if __name__ == "__main__":
    main()
