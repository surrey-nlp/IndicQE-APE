#!/usr/bin/env python3
"""rescore_strict.py — re-derive QE numbers from saved raw replies using the strict parser.

Why this exists
---------------
`parse_qe_score` finds a number almost anywhere in a reply. On four-shot cells that produced
wrong scores while reporting a 100% parse rate:

    'Score (0-100): 88'  -> 0.0    the range start, not the score
    '<590-char essay>'   -> 1.0    the "1." list marker
    '90\\n89\\n65\\n0'      -> 90.0   four numbers; the first is not the test segment

`parse_qe_score_strict` refuses all three. Refusing is the honest outcome: it surfaces as a
compliance failure instead of a silently wrong number.

Every run persists its `raw` reply per instance, so this is a pure re-parse. No GPU, no cluster,
no regeneration.

What it does
------------
1. GATE. Reproduces the published results/analysis/qe_difficulty_routes.csv from the raw files
   using the permissive parser. If that does not match to 0.001 the recipe below is wrong and the
   script refuses to write anything.
2. Four-shot compliance and correlation, permissive vs strict, per model and per pair.
3. Routes recomputed. Two variants, because two things changed and they must not be conflated:
      paired  — same item set as the published table, isolating the parser and thinking effects
      full    — every item available now, which for the open models is a larger challenge split

Published-routes recipe, recovered empirically and asserted by the gate:
    zero-shot, GEMBA-DA, results/qe_open + results/qe_closed, en-ml excluded.

Outputs
    results/analysis/strict_rescore_percell.csv
    results/analysis/qe_difficulty_routes_strict.csv
    results/analysis/strict_rescore.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from indicqe_ape.prompts import parse_qe_score, parse_qe_score_strict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLD_DIRS = ["qe_open", "qe_closed"]
NEW_DIR = "qe_rerun512"
TPL = "gemba-da"
PAIRS = ["enhi", "enmr", "enta", "ente", "engu", "enml", "eten", "neen", "sien"]
ROUTES = ["easy_control", "A1_disagreement", "A2_edit_effort", "A3_word_error",
          "A4_contradiction", "distributional", "co_selected", "pair_floor"]
TOPUPS = {"distributional", "pair_floor"}
DIFFICULTY = [r for r in ROUTES if r not in TOPUPS]
#: systems in the published route table, so the gate can compare against it
LLMS = ["gpt-5.5", "sarvam-m", "tiny-aya-fire", "Llama-3.2-3B-Instruct", "BharatGPT-3B-Indic"]
#: systems in the compliance table (tab:parse4). aya-expanse-32b was never routed but is
#: reported for compliance, so the two lists are not the same.
COMPLIANCE = LLMS + ["aya-expanse-32b"]
#: sarvam-m's qe_open cells were generated with chain-of-thought enabled (2.05% of replies are
#: reasoning prose). qe_rerun512 is the clean regeneration and supersedes them.
SUPERSEDED = {"sarvam-m"}


def read_cells(model: str, shots: str, dirs: list[str], drop_enml: bool) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in glob.glob(str(ROOT / "results" / d / f"{model}.s{shots}.*.jsonl")):
            if drop_enml and ".enml." in f:
                continue
            pair = Path(f).stem.split(".")[-1]
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in r or "_summary" in r:
                    continue
                raw = (r.get("raw") or {}).get(TPL)
                rows.append({
                    "id": r.get("id"), "pair": pair, "route": r.get("route"),
                    "gold": r.get("gold_da_mean"),
                    "lenient": (r.get("pred") or {}).get(TPL),
                    "strict": parse_qe_score_strict(raw),
                    "has_raw": raw is not None,
                    "reparsed": parse_qe_score(raw) if raw is not None else None,
                })
    return pd.DataFrame(rows)


def rho(df: pd.DataFrame, col: str) -> float:
    d = df.dropna(subset=["gold", col])
    if len(d) < 11 or d[col].std() == 0 or d["gold"].std() == 0:
        return np.nan
    return float(stats.spearmanr(d[col], d["gold"]).statistic)


def route_row(df: pd.DataFrame, col: str) -> dict:
    out = {rt: rho(df[df.route == rt], col) for rt in ROUTES}
    out["n_total"] = int(df.dropna(subset=["gold", col]).shape[0])
    return out


def gate(published: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the published routes from the raw files, or refuse to continue.

    Each system is reproduced from the source its published row was built from. That is qe_open +
    qe_closed for everything except sarvam-m, whose row was rebuilt from the clean rerun after its
    originals were found to have chain-of-thought enabled. The gate caught exactly this when the
    row changed and the source here did not, which is what it is for.
    """
    print("GATE: reproducing the published route table from raw files")
    cache, worst = {}, 0.0
    for m in LLMS:
        src = [NEW_DIR] if m in SUPERSEDED else OLD_DIRS
        d = read_cells(m, "0", src, drop_enml=True)
        if d.empty:
            print(f"  {m:<24} NO DATA"); return None
        cache[m] = d
        got = route_row(d, "lenient")
        deltas = {rt: abs(got[rt] - published.loc[m, rt]) for rt in ROUTES
                  if not (np.isnan(got[rt]) or pd.isna(published.loc[m, rt]))}
        bad = max(deltas, key=deltas.get)
        worst = max(worst, deltas[bad])
        ok = "ok" if deltas[bad] < 0.001 else "MISMATCH"
        print(f"  {m:<24} n={got['n_total']:>6} (want {int(published.loc[m,'n_total']):>6})  "
              f"max delta {deltas[bad]:.4f} on {bad}  {ok}")
    if worst >= 0.001:
        print(f"\nREFUSING TO WRITE: cannot reproduce the published table (worst {worst:.4f}).")
        return None
    print(f"  gate passed, worst delta {worst:.4f}\n")
    return cache


def main() -> int:
    published = pd.read_csv(ROOT / "results/analysis/qe_difficulty_routes.csv").set_index("system")
    old0 = gate(published)
    if old0 is None:
        return 1

    L: list[str] = ["# Strict re-parse of the saved QE replies\n\n",
                    "The permissive parser extracts a number from almost any reply, including "
                    "reasoning prose and multi-number four-shot echoes. The strict parser accepts "
                    "a bare number or a single labelled number and refuses everything else. "
                    "Re-parsed from saved `raw`; nothing was regenerated.\n\n"]

    # ---------------------------------------------------------------- 1. per-cell compliance
    print("Four-shot compliance and correlation, permissive vs strict")
    rows = []
    for m in COMPLIANCE:
        for shots in ("0", "4"):
            # Compliance is reported from the rerun (512-token, no-think), which is what
            # tab:parse4 shows. Only gpt-5.5 has no rerun; it falls back to its original cells.
            src = [NEW_DIR] if glob.glob(
                str(ROOT / "results" / NEW_DIR / f"{m}.s{shots}.*.jsonl")) else OLD_DIRS
            d = read_cells(m, shots, src, drop_enml=False)
            if d.empty or not d.has_raw.any():
                continue
            for pair, g in d.groupby("pair"):
                rows.append({
                    "model": m, "shots": shots, "pair": pair, "n": len(g),
                    "parse_lenient": g.lenient.notna().mean(),
                    "parse_strict": g.strict.notna().mean(),
                    "rho_lenient": rho(g, "lenient"), "rho_strict": rho(g, "strict"),
                })
    cell = pd.DataFrame(rows)
    cell.to_csv(ROOT / "results/analysis/strict_rescore_percell.csv", index=False)

    L.append("## Compliance, macro over pairs\n\n")
    L.append("| model | shots | parse (reported) | parse (strict) | rho (reported) | rho (strict) |\n")
    L.append("|---|--:|--:|--:|--:|--:|\n")
    for (m, s), g in cell.groupby(["model", "shots"]):
        L.append(f"| {m} | s{s} | {g.parse_lenient.mean():.3f} | {g.parse_strict.mean():.3f} | "
                 f"{g.rho_lenient.mean():.3f} | {g.rho_strict.mean():.3f} |\n")
        print(f"  {m:<24} s{s}  parse {g.parse_lenient.mean():.3f} -> {g.parse_strict.mean():.3f}"
              f"   rho {g.rho_lenient.mean():.3f} -> {g.rho_strict.mean():.3f}")

    L.append("\n## Four-shot compliance by pair, strict\n\n")
    piv = cell[cell.shots == "4"].pivot_table(index="model", columns="pair", values="parse_strict")
    piv = piv.reindex(columns=[p for p in PAIRS if p in piv.columns])
    L.append("| model | " + " | ".join(piv.columns) + " |\n")
    L.append("|---" * (len(piv.columns) + 1) + "|\n")
    for m, r in piv.iterrows():
        L.append(f"| {m} | " + " | ".join("---" if pd.isna(v) else f"{v:.3f}" for v in r) + " |\n")

    # ---------------------------------------------------------------- 2. routes
    print("\nRoutes: published vs paired vs full")
    out = []
    for m in LLMS:
        pub = {rt: published.loc[m, rt] for rt in ROUTES}
        if m in SUPERSEDED:
            new = read_cells(m, "0", [NEW_DIR], drop_enml=True)
            paired = new[new.id.isin(set(old0[m].id))]          # same items as published
            full = new
        else:
            paired = full = old0[m]
        r_paired, r_full = route_row(paired, "strict"), route_row(full, "strict")
        for label, r in (("published", {**pub, "n_total": published.loc[m, "n_total"]}),
                         ("paired", r_paired), ("full", r_full)):
            out.append({"system": m, "variant": label, **r})
        mv = max(abs(r_paired[rt] - pub[rt]) for rt in ROUTES
                 if not (np.isnan(r_paired[rt]) or pd.isna(pub[rt])))
        print(f"  {m:<24} paired-vs-published max delta {mv:.4f}"
              f"{'   <-- moved' if mv >= 0.005 else ''}")

    routes = pd.DataFrame(out)
    routes.to_csv(ROOT / "results/analysis/qe_difficulty_routes_strict.csv", index=False)

    L.append("\n## Routes, strict parser, zero-shot\n\n")
    L.append("`paired` holds the item set of the published table, so it isolates the change. "
             "`full` uses every item now available.\n\n")
    L.append("| system | variant | n | " + " | ".join(
        r.split("_")[0] for r in ROUTES) + " | hardest difficulty route |\n")
    L.append("|---" * (len(ROUTES) + 4) + "|\n")
    for _, r in routes.iterrows():
        vals = " | ".join("---" if pd.isna(r[rt]) else f"{r[rt]:.3f}" for rt in ROUTES)
        d = {rt: r[rt] for rt in DIFFICULTY if not pd.isna(r[rt])}
        hardest = min(d, key=d.get).split("_")[0] if d else "---"
        L.append(f"| {r.system} | {r.variant} | {int(r.n_total)} | {vals} | {hardest} |\n")

    # does the A4/A1 split in the paper survive?
    L.append("\n## Effect on the reported A4/A1 split\n\n")
    for variant in ("published", "paired", "full"):
        sub = routes[routes.variant == variant]
        tally = {"A4": [], "A1": [], "other": []}
        for _, r in sub.iterrows():
            d = {rt: r[rt] for rt in DIFFICULTY if not pd.isna(r[rt])}
            if not d:
                continue
            h = min(d, key=d.get)
            tally["A4" if h == "A4_contradiction" else
                  "A1" if h == "A1_disagreement" else "other"].append(r.system)
        L.append(f"- **{variant}**: A4 hardest for {len(tally['A4'])} "
                 f"({', '.join(tally['A4']) or 'none'}); A1 for {len(tally['A1'])} "
                 f"({', '.join(tally['A1']) or 'none'})"
                 + (f"; other for {len(tally['other'])} ({', '.join(tally['other'])})"
                    if tally["other"] else "") + "\n")

    (ROOT / "results/analysis/strict_rescore.md").write_text("".join(L))
    print("\nwrote results/analysis/{strict_rescore_percell.csv,"
          "qe_difficulty_routes_strict.csv,strict_rescore.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
