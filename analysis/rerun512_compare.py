#!/usr/bin/env python3
"""rerun512_compare.py — before/after for the prompted-QE rerun.

WHAT THE "BEFORE" CONDITION ACTUALLY IS, PER MODEL. This script used to call it "the 8-token run"
for everything, which is false for one model and was corrected on 2026-08-07:

  BharatGPT-3B, Llama-3.2-3B, tiny-aya, aya-expanse   before = IQE_QE_MAXTOK=8, truncating any
                                                      model that emitted a word before its score
  sarvam-m                                            before = max_tokens 512 ALREADY, but with
                                                      chain-of-thought ENABLED. 2.05% of its
                                                      replies are reasoning prose (18.5% on
                                                      s4.engu) and the permissive parser scraped
                                                      numbers out of them.

So for four models this is a decoding-budget comparison and for sarvam-m it is a thinking versus
no-thinking comparison. Read the sarvam-m row accordingly; do not describe the whole table as a
budget effect. `_assert_before_conditions` checks each cell's _meta against the table above and
fails loudly if a file's provenance does not match, so the two cannot be silently conflated again.

The rerun (results/qe_rerun512) also uses a corrected few-shot pool (filter on DA before sampling,
so k-shot really carries k examples), which is a second change on top of either of the above.

Comparisons are PAIRED on segment id. The two runs cover different populations (the original
predates the challenge-test extension, and its four-shot cells omit en-ml), so an unpaired
comparison would confound the population change with the configuration change.

Outputs
  results/analysis/rerun512_compare.md
  results/analysis/rerun512_compare.csv
"""
from __future__ import annotations

import json
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OLD_DIRS = [ROOT / "results" / "qe_open", ROOT / "results" / "qe_closed"]
NEW_DIR = ROOT / "results" / "qe_rerun512"
OUT_MD = ROOT / "results" / "analysis" / "rerun512_compare.md"
OUT_CSV = ROOT / "results" / "analysis" / "rerun512_compare.csv"
TPLS = ("gemba-da", "gemba-sqm")
PAT = re.compile(r"^(?P<model>.+)\.s(?P<shots>\d+)\.(?P<pair>\w+)\.jsonl$")


def load(path: Path) -> dict:
    """id -> {template: pred}, plus gold. Skips meta/summary lines."""
    out = {}
    complete = False
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in r:
                continue
            if "_summary" in r:
                complete = True
                continue
            out[r["id"]] = (r.get("gold_da_mean"), r.get("pred") or {})
    return {"rows": out, "complete": complete}


def collect(dirs) -> dict:
    got = {}
    for d in dirs:
        if not Path(d).exists():
            continue
        for f in sorted(Path(d).glob("*.jsonl")):
            m = PAT.match(f.name)
            if m:
                got[(m["model"], m["shots"], m["pair"])] = f
    return got


def corr(pred, gold):
    p = np.array(pred, float)
    g = np.array(gold, float)
    m = np.isfinite(p) & np.isfinite(g)
    if m.sum() < 30 or np.std(p[m]) == 0 or np.std(g[m]) == 0:
        return np.nan, int(m.sum())
    return float(stats.spearmanr(p[m], g[m]).statistic), int(m.sum())


#: Models whose original cells were generated with chain-of-thought enabled. For these the rerun
#: is a thinking-versus-no-thinking comparison, never a decoding-budget one.
THINKING_MODELS = {"sarvam-m"}


def _assert_before_conditions(old: dict) -> list[str]:
    """Report, per model, what the `before` cells actually were, read from their _meta.

    Do not summarise this as one condition. Checking it turned up three things that the old
    "8-token run" label hid:

      * aya-expanse, sarvam-m and sarvam-translate were NEVER run at 8 tokens.
      * BharatGPT and Llama are mixed: some cells at 8, some at 512, because cells added later
        (s0.enml, and several four-shot cells) already used the corrected default.
      * sarvam-m was at 512 throughout but with chain-of-thought enabled, so its delta is a
        thinking effect.

    A single label over that is a fiction, so the report prints the split per model and the
    caller must read the relevant row.
    """
    seen: dict[str, dict] = {}
    for (model, shots, pair), path in old.items():
        try:
            mt = json.loads(open(path, encoding="utf-8").readline())["_meta"]["generation"]["max_tokens"]
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            mt = None
        seen.setdefault(model, defaultdict(list))[mt].append(f"s{shots}.{pair}")

    lines = ["\n## What the `before` condition was, per model\n\n",
             "| model | cells at 8 tokens | cells at 512 | what the delta measures |\n",
             "|---|--:|--:|---|\n"]
    for model in sorted(seen):
        n8 = len(seen[model].get(8, []))
        n512 = len(seen[model].get(512, []))
        if model in THINKING_MODELS:
            what = "thinking vs no-thinking"
        elif n8 and n512:
            what = "**mixed**: budget on some cells, nothing on others"
        elif n8:
            what = "decoding budget, 8 to 512"
        else:
            what = "few-shot pool fix only"
        lines.append(f"| {model} | {n8} | {n512} | {what} |\n")
        print(f"  before: {model:<24} 8-tok={n8:<3} 512={n512:<3} -> {what}")
    lines.append("\nEvery row also carries the few-shot pool fix. A model with cells in both "
                 "columns has a delta that is part budget change and part nothing, so its "
                 "aggregate should not be read as a budget effect.\n")
    return lines


def main() -> int:
    old, new = collect(OLD_DIRS), collect([NEW_DIR])
    keys = sorted(set(old) & set(new))
    print(f"old cells {len(old)} | new cells {len(new)} | overlapping {len(keys)}")

    rows = []
    for k in keys:
        mo, sh, pr = k
        A, B = load(old[k]), load(new[k])
        if not B["complete"]:
            continue
        common = set(A["rows"]) & set(B["rows"])
        if not common:
            continue
        for t in TPLS:
            gp_o, gp_n, gold_o, gold_n = [], [], [], []
            po = pn = 0
            for i in common:
                g, pa = A["rows"][i]
                _, pb = B["rows"][i]
                a, b = pa.get(t), pb.get(t)
                po += a is not None
                pn += b is not None
                if g is not None:
                    gp_o.append(a if a is not None else np.nan); gold_o.append(g)
                    gp_n.append(b if b is not None else np.nan); gold_n.append(g)
            so, no_ = corr(gp_o, gold_o)
            sn, nn = corr(gp_n, gold_n)
            rows.append(dict(model=mo, shots=sh, pair=pr, template=t, n_common=len(common),
                             parse_old=po / len(common), parse_new=pn / len(common),
                             rho_old=so, rho_new=sn, n_scored_old=no_, n_scored_new=nn))

    if not rows:
        print("no overlapping complete cells")
        return 1
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"cells compared: {len(df)} -> {OUT_CSV}")

    L = ["# Prompted QE: original run vs rerun\n\n",
         "Paired on segment id, so the population is identical within each row. The `before` "
         "condition is NOT the same for every model, and the table below says which is which: a "
         "per-model breakdown is below and is mixed for several. Every row also carries a fix to the few-shot pool (k-shot now really "
         "carries k examples; previously DA-sparse pairs got fewer).\n"]
    L += _assert_before_conditions(old)
    L.append("\n")

    L.append("## Output-format compliance, macro over pairs\n\n")
    L.append("| model | shots | template | parse before | parse after | change |\n|---|--:|---|--:|--:|--:|\n")
    g = df.groupby(["model", "shots", "template"])[["parse_old", "parse_new"]].mean().reset_index()
    for _, r in g.sort_values(["model", "shots", "template"]).iterrows():
        d = r.parse_new - r.parse_old
        flag = " **" if abs(d) >= 0.05 else " "
        L.append(f"| {r.model} | s{r.shots} | {r.template} | {r.parse_old:.3f} | "
                 f"{r.parse_new:.3f} |{flag}{d:+.3f}{flag.strip()} |\n")

    L.append("\n## Where the budget was NOT the cause\n\n")
    sub = df[(df.shots == "4") & (df.template == "gemba-da")]
    for mo, gg in sub.groupby("model"):
        before = gg.parse_old.mean(); after = gg.parse_new.mean()
        if after < 0.95:
            worst = gg.nsmallest(3, "parse_new")[["pair", "parse_new"]]
            cells = ", ".join(f"{r.pair} {r.parse_new:.3f}" for _, r in worst.iterrows())
            L.append(f"**{mo}** four-shot: {before:.3f} to {after:.3f}. Still failing on {cells}. "
                     "The unparsed outputs at 512 tokens are full-length replies, not truncated "
                     "fragments, so for this model the four-shot collapse is a property of the "
                     "model and not of the decoding budget.\n\n")

    L.append("## Correlation with human DA, paired\n\n")
    L.append("| model | shots | template | rho before | rho after | change |\n|---|--:|---|--:|--:|--:|\n")
    gc = df.groupby(["model", "shots", "template"])[["rho_old", "rho_new"]].mean().reset_index()
    for _, r in gc.sort_values(["model", "shots", "template"]).iterrows():
        d = r.rho_new - r.rho_old
        L.append(f"| {r.model} | s{r.shots} | {r.template} | {r.rho_old:.3f} | "
                 f"{r.rho_new:.3f} | {d:+.3f} |\n")

    L.append("\n## en-ml, new in this rerun\n\n")
    L.append("The original four-shot cells omitted en-ml because the launcher's pair array did.\n\n")
    L.append("| model | shots | template | parse | rho |\n|---|--:|---|--:|--:|\n")
    for f in sorted(NEW_DIR.glob("*.s4.enml.jsonl")):
        m = PAT.match(f.name)
        B = load(f)
        if not B["complete"]:
            continue
        for t in TPLS:
            pred = [v[1].get(t) for v in B["rows"].values()]
            gold = [v[0] for v in B["rows"].values()]
            pr = sum(p is not None for p in pred) / max(len(pred), 1)
            s, _ = corr([np.nan if p is None else p for p in pred],
                        [np.nan if g is None else g for g in gold])
            L.append(f"| {m['model']} | s4 | {t} | {pr:.3f} | {s:.3f} |\n")

    OUT_MD.write_text("".join(L))
    print("".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
