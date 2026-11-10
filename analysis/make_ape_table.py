#!/usr/bin/env python3
"""make_ape_table.py — generate tab:ape from the surface metrics and the CometKiwi pass.

Two histories worth keeping, because both were silent-error classes:

ONE POPULATION. Until 2026-08-07 this table was assembled by hand from three places, and carried
two aggregation conventions: en-hi/mr/ta came from results/analysis/improve/APE_MATRA.md (the
per-instance rows having been lost to 300-byte stubs), while en-ml came from
results/enml-runs/enml_ape_scores.json scored as the MEAN OF PER-SENTENCE chrF, where the
canonical scripts/ape_metrics.py uses corpus-level chrF and micro-averaged char-TER. Restoring the
stubs from results_srv and rescoring reproduced the nine en-hi/mr/ta values exactly, which is what
confirmed results_srv was the right source.

ONE TEST SET. Until 2026-08-08 the en-hi/mr/ta cells were still on the OLD challenge test
(1,490 / 2,024 / 1,375) while en-ml was on the current one (1,436). The ape_regen cluster job put
all of them on the current test (1,536 / 2,098 / 1,448). Every value moved by at most 0.4 and no
best-per-pair marker changed, so the regeneration reproduced the lost runs rather than replacing
them with something else.

THE COMETKIWI COLUMN. char-TER and chrF++ score distance to the human post-edit, which is anchored
to the MT, so they measure residual repair effort (S3, "human-targeted"). CometKiwi scores the
output against the source alone. sarvam-translate retranslates instead of post-editing, so it
departs from the post-edit by construction and the surface metrics mark it down however good the
translation is. Reporting only the surface pair therefore hides a systematic ranking disagreement,
which is why the neural column is in the table rather than in a sentence.

Sources: results/analysis/ape_metrics_table.csv (scripts/ape_metrics.py)
         results/analysis/ape_comet_table.csv  (analysis/make_ape_comet_table.py)

Splices into acl_latex.tex between the tab_ape sentinels. --check exits 1 if out of date.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

TEX = _PAPER_TEX
CSV = ROOT / "results" / "analysis" / "ape_metrics_table.csv"
COMET = ROOT / "results" / "analysis" / "ape_comet_table.csv"
DONOTHING = ROOT / "results" / "analysis" / "ape_do_nothing.csv"

#: (csv prefix, LaTeX macro, short name used in ape_comet_table.csv) in table order
SYSTEMS = [("sarvam-m.s0", r"\mSarvam", "sarvam-m"),
           ("sarvam-translate.mt", r"\mNMT", "sarvam-t"),
           ("tiny-aya-fire.s0", r"\mTiny", "tiny-aya")]
PAIRS = [("enhi", "en-hi"), ("enmr", "en-mr"), ("enta", "en-ta"), ("enml", "en-ml")]
#: key for the do-nothing baseline in the per-pair best comparison
DN = "do-nothing"


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(% BEGIN GENERATED {re.escape(name)}[^\n]*\n).*?"
                     rf"(% END GENERATED {re.escape(name)})", re.S)
    if not pat.search(text):
        raise SystemExit(f"sentinels for {name} not found in {TEX.name}")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    df = pd.read_csv(CSV).set_index("system")
    missing = [f"{p}.{pair}" for p, _, _ in SYSTEMS for pair, _ in PAIRS
               if f"{p}.{pair}" not in df.index]
    if missing:
        raise SystemExit(f"missing cells in {CSV.name}: {missing}")

    ck = {}
    if COMET.exists():
        cdf = pd.read_csv(COMET)
        ck = {(r["system"], r["pair"]): (float(r["cometkiwi"]), int(r["n"]))
              for _, r in cdf.iterrows()}

    # the two passes must have scored the same hypotheses, or the columns are not comparable
    for prefix, _, short in SYSTEMS:
        for pair, _ in PAIRS:
            if (short, pair) in ck and int(df.loc[f"{prefix}.{pair}", "n"]) != ck[(short, pair)][1]:
                raise SystemExit(
                    f"POPULATION MISMATCH {short}.{pair}: surface n="
                    f"{df.loc[f'{prefix}.{pair}', 'n']} vs CometKiwi n={ck[(short, pair)][1]}")

    # The do-nothing baseline: char-TER and chrF++ of the unedited MT against the human post-edit,
    # on exactly the rows the systems are scored on. It competes for the bold marks, because a
    # system that is worse than not editing should not be marked best.
    dn = {}
    if DONOTHING.exists():
        dn = {r["pair"]: r for _, r in pd.read_csv(DONOTHING).iterrows()}
        for pair, _ in PAIRS:
            if pair in dn and int(dn[pair]["n"]) != int(df.loc[f"{SYSTEMS[0][0]}.{pair}", "n"]):
                raise SystemExit(f"POPULATION MISMATCH do-nothing.{pair}")

    best = {}
    for pair, _ in PAIRS:
        col = {mac: df.loc[f"{p}.{pair}"] for p, mac, _ in SYSTEMS}
        cks = {mac: ck.get((short, pair), (float("-inf"),))[0] for _, mac, short in SYSTEMS}
        if pair in dn:
            col[DN] = {"charTER": float(dn[pair]["charTER"]), "chrF++": float(dn[pair]["chrfpp"])}
        best[pair] = (min(col, key=lambda m: col[m]["charTER"]),
                      max(col, key=lambda m: col[m]["chrF++"]),
                      max(cks, key=lambda m: cks[m]) if ck else None)

    L = []
    if dn:
        for pi, (pair, label) in enumerate(PAIRS):
            r = dn[pair]
            ter, chrf = f"{float(r['charTER']):.1f}", f"{float(r['chrfpp']):.1f}"
            if best[pair][0] == DN:
                ter = f"\\best{{{ter}}}"
            if best[pair][1] == DN:
                chrf = f"\\best{{{chrf}}}"
            head = "\\multirow{4}{*}{do-nothing}" if pi == 0 else ""
            L.append(f"{head} & {label} & {ter} & {chrf} & --- \\\\\n")
            print(f"  {'do-nothing':<10} {label}  char-TER {float(r['charTER']):>5.1f}  "
                  f"chrF++ {float(r['chrfpp']):>5.1f}  zero-edit {float(r['zero_edit']):.3f}")
    for si, (prefix, mac, short) in enumerate(SYSTEMS):
        if si or dn:
            L.append("\\midrule\n")
        for pi, (pair, label) in enumerate(PAIRS):
            r = df.loc[f"{prefix}.{pair}"]
            ter, chrf = f"{r['charTER']:.1f}", f"{r['chrF++']:.1f}"
            if best[pair][0] == mac:
                ter = f"\\best{{{ter}}}"
            if best[pair][1] == mac:
                chrf = f"\\best{{{chrf}}}"
            if (short, pair) in ck:
                c = f"{ck[(short, pair)][0]:.3f}"
                if best[pair][2] == mac:
                    c = f"\\best{{{c}}}"
            else:
                c = "---"
            head = f"\\multirow{{4}}{{*}}{{{mac}}}" if pi == 0 else ""
            L.append(f"{head} & {label} & {ter} & {chrf} & {c} \\\\\n")
            print(f"  {mac:<10} {label}  char-TER {r['charTER']:>5.1f}  chrF++ {r['chrF++']:>5.1f}"
                  f"  CometKiwi {c.replace(chr(92) + 'best{', '').rstrip('}'):>6}  n={int(r['n'])}")

    # tab:apeword — the same rows under the two metrics the APE shared tasks report. char-TER and
    # chrF++ stay primary because 13a word tokenisation charges a whole-word edit for a
    # one-character Indic inflection, but a reader comparing against a shared-task table needs the
    # word-level pair, and the conclusion should be shown not to depend on the choice.
    W = []
    wbest = {}
    for pair, _ in PAIRS:
        col = {mac: (float(df.loc[f"{p}.{pair}", "wordTER"]),
                     float(df.loc[f"{p}.{pair}", "wordBLEU"])) for p, mac, _ in SYSTEMS}
        if pair in dn:
            col[DN] = (float(dn[pair]["wordTER"]), float(dn[pair]["wordBLEU"]))
        wbest[pair] = (min(col, key=lambda m: col[m][0]), max(col, key=lambda m: col[m][1]))

    def wrow(head: str, label: str, mac: str, ter: float, bleu: float, pair: str) -> str:
        t = f"\\best{{{ter:.1f}}}" if wbest[pair][0] == mac else f"{ter:.1f}"
        b = f"\\best{{{bleu:.1f}}}" if wbest[pair][1] == mac else f"{bleu:.1f}"
        return f"{head} & {label} & {t} & {b} \\\\\n"

    if dn:
        for pi, (pair, label) in enumerate(PAIRS):
            r = dn[pair]
            W.append(wrow("\\multirow{4}{*}{do-nothing}" if pi == 0 else "", label, DN,
                          float(r["wordTER"]), float(r["wordBLEU"]), pair))
    for si, (prefix, mac, _) in enumerate(SYSTEMS):
        if si or dn:
            W.append("\\midrule\n")
        for pi, (pair, label) in enumerate(PAIRS):
            r = df.loc[f"{prefix}.{pair}"]
            W.append(wrow(f"\\multirow{{4}}{{*}}{{{mac}}}" if pi == 0 else "", label, mac,
                          float(r["wordTER"]), float(r["wordBLEU"]), pair))
    dn_wins = [p for p, _ in PAIRS if wbest[p] == (DN, DN)]
    print(f"\n  word-level: do-nothing is best on both metrics for {dn_wins}")

    tex = TEX.read_text()
    want = splice(tex, "tab_ape", "".join(L))
    if "% BEGIN GENERATED tab_ape_word" in want:
        want = splice(want, "tab_ape_word", "".join(W))
    if want == tex:
        print("\nacl_latex.tex is up to date")
        return 0
    if "--check" in sys.argv:
        print("\nOUT OF DATE")
        return 1
    TEX.write_text(want)
    print(f"\nspliced tab:ape and tab:apeword into {TEX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
