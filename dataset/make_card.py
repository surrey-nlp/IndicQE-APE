#!/usr/bin/env python3
r"""make_card.py — write dataset/hf_snapshot/README.md from the release itself.

Every count in the card is read out of the parquet files rather than typed, so the card cannot
drift from the data it describes. The YAML front matter is kept in dataset/templates/card_header.yaml
(tracked, so a fresh snapshot regenerates the same card), the citation comes from
analysis/citation.py, and the results tables are read from the generated LaTeX in the paper.

    python3 dataset/make_card.py
    python3 dataset/make_card.py --check   # fail if the card on disk is not what this would write
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
from citation import ANTHOLOGY_URL, ARXIV, ARXIV_URL, BIBTEX, REPO_URL, TITLE  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

SNAP = ROOT / "dataset" / "hf_snapshot"
DATA = SNAP / "data"
CARD = SNAP / "README.md"
HEADER = ROOT / "dataset" / "templates" / "card_header.yaml"
TEX = _PAPER_TEX

PAIRS = ["en-hi", "en-mr", "en-ta", "en-gu", "en-te", "en-ml", "et-en", "ne-en", "si-en"]
AX = ["axis_a1", "axis_a2", "axis_a3", "axis_a4"]
CFG = [("full", "data/full", "nine pairs, every instance with a QE, APE or explanation label"),
       ("all", "data/all", "en&rarr;Indic only"),
       ("qe-da", "data/tasks/qe-da", "en&rarr;Indic, direct-assessment bearing"),
       ("ape", "data/tasks/ape", "the seven post-edited pairs"),
       ("word-qe", "data/tasks/word-qe", "word-level OK/BAD tags"),
       ("explainable-qe", "data/tasks/explainable-qe", "error explanations"),
       ("mqm", "data/tasks/mqm", "the en-hi MQM layer"),
       ("challenge", "data/challenge", "the curated difficulty-stratified test, no train split")]


def load(rel: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(SNAP / rel / "*.parquet")))
    fs = [f for f in fs if "blind" not in f]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def nn(s: pd.Series) -> pd.Series:
    return s.notna() & s.astype(str).str.strip().ne("") & s.astype(str).ne("None")


def texrows(tag: str) -> list[list[str]]:
    """Pull a generated LaTeX table body out of the paper and strip it back to plain cells."""
    m = re.search(rf"% BEGIN GENERATED {tag}[^\n]*\n(.*?)% END GENERATED {tag}", TEX.read_text(), re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).strip().split("\n"):
        if line.startswith("%") or r"\midrule" in line:
            continue
        line = re.sub(r"\\multirow\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", line)
        cells = [re.sub(r"\\best\{|\\mathbf\{|[{}$]", "", c).strip()
                 for c in line.rstrip("\\ ").split("&")]
        out.append([c.replace("\\m", "").replace("$-$", "-").replace("---", "—") for c in cells])
    return out


#: The paper's system labels (its \m... macros), with the size or mode added in brackets where the
#: label alone does not say it, so the tables can be cross-checked against the paper directly.
MODEL = {"GPT": "GPT-5.5", "Sarvam": "sarvam-m (24B)", "AyaB": "aya-expanse-32B",
         "Tiny": "tiny-aya (tiny-aya-fire, 3.4B)", "LlamaB": "Llama-3.2-3B",
         "BharatB": "BharatGPT-3B", "CKDA": "CometKiwi-DA", "CKXL": "CometKiwi-XL",
         "XC": "XCOMET-XL (QE mode)", "NMT": "sarvam-t (sarvam-translate)"}


def _f(x: str) -> float:
    """A table cell as a float; the paper typesets minus as an en dash or unicode minus."""
    return float(x.replace("\u2212", "-").replace("\u2013", "-").replace("+", "").strip())


def qe_takeaway(rows: list[list[str]]) -> str:
    """The QE table's one-sentence reading, computed from the rows rather than typed."""
    trained = {MODEL[k] for k in ("CKDA", "CKXL", "XC")}
    named = [(MODEL.get(r[0], r[0]), _f(r[2]), _f(r[4])) for r in rows]
    if not named:
        return ""
    best = max(named, key=lambda t: t[1])
    tm = [t for t in named if t[0] in trained]
    if best[0] in trained and best[2] == min(t[2] for t in tm):
        return (f"{best[0]} has the highest within-language correlation of any system and the weakest "
                f"cross-lingual agreement of the three trained metrics ({best[2]:+.2f}), so raw "
                f"reference-free scores are not comparable across language pairs.")
    return "Raw reference-free scores are not comparable across language pairs."


def main() -> int:
    check = "--check" in sys.argv
    full, ch = load("data/full"), load("data/challenge")
    wq, mq = load("data/tasks/word-qe"), load("data/tasks/mqm")
    da = ch[ch.has_da.astype("boolean").fillna(False)]
    n = lambda v: f"{int(v):,}"  # noqa: E731

    # ---- per-pair composition -------------------------------------------------------------
    comp = []
    for p in PAIRS:
        s = full[full.language_pair == p]
        comp.append(f"| {p} | {n(len(s))} | {n(nn(s.da_mean).sum())} | {n(nn(s.post_edit).sum())} "
                    f"| {n(nn(s.wtag).sum() + nn(s.wtag_derived).sum())} "
                    f"| {n((nn(s.synthetic_description) | nn(s.tqr)).sum())} "
                    f"| {n(s.uid.isin(set(mq.uid)).sum())} |")
    comp.append(f"| **total** | **{n(len(full))}** | **{n(nn(full.da_mean).sum())}** "
                f"| **{n(nn(full.post_edit).sum())}** "
                f"| **{n(nn(full.wtag).sum() + nn(full.wtag_derived).sum())}** "
                f"| **{n((nn(full.synthetic_description) | nn(full.tqr)).sum())}** "
                f"| **{n(full.uid.isin(set(mq.uid)).sum())}** |")

    # ---- configs --------------------------------------------------------------------------
    cfg = []
    for name, rel, scope in CFG:
        d = load(rel)
        parts = {Path(f).stem: len(pd.read_parquet(f))
                 for f in sorted(glob.glob(str(SNAP / rel / "*.parquet"))) if "blind" not in f}
        cfg.append(f"| `{name}` | {n(parts.get('train', 0)) if 'train' in parts else '—'} "
                   f"| {n(parts.get('validation', 0)) if 'validation' in parts else '—'} "
                   f"| {n(parts.get('test', 0))} | {n(len(d))} | {scope} |")

    # ---- axis coverage ---------------------------------------------------------------------
    cov = []
    for p in PAIRS:
        s = da[da.language_pair == p]
        cells = " | ".join("—" if s[a].isna().all() else n(s[a].fillna(False).sum()) for a in AX)
        cov.append(f"| {p} | {n(len(s))} | {cells} |")
    n_ax = da[AX].fillna(False).astype(bool).sum(axis=1)
    flagged, arms = int((n_ax >= 1).sum()), da[da.axis_a4.fillna(False)].axis_a4_arm.value_counts()

    # ---- results ---------------------------------------------------------------------------
    qrows = texrows("tab_qe_body")
    qe = [f"| {MODEL.get(r[0], r[0])} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in qrows]
    ape_rows, cur = [], None
    for r in texrows("tab_ape"):
        if len(r) == 5:
            cur = MODEL.get(r[0].replace("multirow{4}{*}{", "").strip(), r[0])
            r = r[1:]
        ape_rows.append(f"| {cur} | {r[0]} | {r[1]} | {r[2]} | {r[3]} |")

    card = f"""---
{HEADER.read_text(encoding="utf-8").rstrip()}
---

# IndicQE-APE

**{TITLE}** — WMT 2026 (Eleventh Conference on Machine Translation).
[Paper (ACL Anthology)]({ANTHOLOGY_URL}) · [arXiv:{ARXIV}]({ARXIV_URL}) · [Code]({REPO_URL})

A quality-estimation and automatic-post-editing benchmark for Indic and other low-resource
language pairs. It consolidates the WMT 2020–2024 QE and APE shared-task lineage with an extended
English→Malayalam resource into **{n(len(full))} instances over nine directional pairs**, with up
to four label types aligned on the same segment: a direct assessment, a human post-edit,
word-level OK/BAD tags and an error explanation. The test set is built rather than sampled, and
every test segment carries a flag on each difficulty axis its labels can express.

```python
from datasets import load_dataset

ds   = load_dataset("surrey-nlp/IndicQE-APE", "full")        # everything, nine pairs
qe   = load_dataset("surrey-nlp/IndicQE-APE", "qe-da")       # source, MT, DA score
ape  = load_dataset("surrey-nlp/IndicQE-APE", "ape")         # source, MT, human post-edit
hard = load_dataset("surrey-nlp/IndicQE-APE", "challenge")   # the curated test set
hi   = load_dataset("surrey-nlp/IndicQE-APE", "en-hi")       # one pair, all tasks

row = qe["test"][0]
print(row["source"], row["mt"], row["da_mean"], row["language_pair"])
```

## What is in a row

| field | what it holds |
|---|---|
| `source`, `mt` | the source segment and the machine translation being judged |
| `da_mean`, `da_scores`, `da_std`, `n_annotators` | direct assessment, 0–100, and the individual annotator scores behind it |
| `post_edit`, `human_ter` | the human post-edit and its token TER against the MT |
| `wtag`, `wtag_derived`, `wtag_provenance` | word-level OK/BAD tags. `wtag` holds the {n(nn(wq.wtag).sum())} released tags, `wtag_derived` the {n(nn(wq.wtag_derived).sum())} derived here from MT and post-edit where no released tag exists, and `wtag_provenance` says which a row carries. They are separate columns so the two are never mixed by accident. |
| `tqr`, `synthetic_description`, `error_categories` | error explanations, human and model-generated, with `explanation_model` recording the source |
| `mqm_spans`, `mqm_score` | the en-hi MQM layer |
| `axis_a1` … `axis_a4`, `axis_a4_arm`, `n_axes` | the difficulty axes, below |
| `domain`, `domain_provenance` | a ten-domain label, annotated or inferred |
| `sources`, `cadence_id`, `original_id` | provenance back to the corpus each instance came from |

Missing labels are null, never zero. `has_da`, `has_pe`, `has_wtag`, `has_tqr`, `has_mqm` give the
same information as booleans.

## Configurations

| config | train | validation | test | total | scope |
|---|---:|---:|---:|---:|---|
{chr(10).join(cfg)}

Per-pair configurations are also available under the pair name: `en-hi`, `en-mr`, `en-ta`,
`en-gu`, `en-te`, `en-ml`, `et-en`, `ne-en`, `si-en`.

`full` holds every instance carrying a QE, APE or explanation label. The {n(len(mq) - mq.uid.isin(set(full.uid)).sum())}
en-hi segments annotated for MQM alone are in `mqm` and nowhere else. `challenge` also ships
`test_blind.parquet`, the same segments with every label withheld.

## Composition

| pair | instances | direct assessment | post-edit | word tags | explanations | MQM |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comp)}

en-gu and en-te arrive from WMT 2023 and WMT 2024 as direct assessments alone: no post-edits, no
word tags, no error categories. The three X→en pairs come from MLQE-PE.

## Difficulty axes

Each is a function of released columns alone, and every threshold is taken **within language
pair**, so a segment is placed against its own pair's distribution and never against the corpus.
The one exception is A1's [35, 75] band on the 0–100 DA scale, which is the same for every pair.

| axis | fires when |
|---|---|
| `axis_a1` annotator disagreement | the per-annotator DA standard deviation is in the pair's top quartile and the mean is in [35, 75] |
| `axis_a2` edit effort | MT→post-edit token TER is in the pair's top quartile |
| `axis_a3` word error | BAD-tag density is in the pair's top quartile, or the error category is mistranslation, untranslated or addition |
| `axis_a4` signal conflict | the holistic score and the token-level evidence disagree: half a standard deviation above the pair's mean yet error-dense by the same margin ({n(arms.get('scored well, error-dense', 0))} segments), or in the pair's bottom 30% yet edited half a standard deviation less than its norm ({n(arms.get('scored badly, barely edited', 0))} segments). `axis_a4_arm` records which. |

An axis a pair has no signal for is **null**, not false. The flags overlap, so they ship as four
booleans and a count rather than one label: of the {n(flagged)} flagged segments in the
DA-bearing challenge test only {n(int((n_ax == 1).sum()))} sit on exactly one axis.

| pair | DA-bearing test | `axis_a1` | `axis_a2` | `axis_a3` | `axis_a4` |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cov)}

```python
hard = load_dataset("surrey-nlp/IndicQE-APE", "challenge")["test"]
conflict = hard.filter(lambda r: r["axis_a4"] is True)          # the hardest axis
clean    = hard.filter(lambda r: r["n_axes"] == 0)              # on no axis at all
```

## Baseline results

Segment-level QE on the DA-bearing challenge test, nine pairs, zero-shot. The first three columns
are within-language, each the mean of the nine per-pair scores. `cross-ling. r` is the Pearson
correlation between per-pair model means and per-pair human means, an estimate from nine points.

| system | Pearson r | Spearman ρ | Kendall τ | cross-ling. r |
|---|---:|---:|---:|---:|
{chr(10).join(qe)}

{qe_takeaway(qrows)}

Automatic post-editing on the four pairs with system outputs. `do-nothing` is the unedited MT
scored against the human post-edit, which is the baseline an APE system has to beat.

| system | pair | char-TER ↓ | chrF++ ↑ | CometKiwi ↑ |
|---|---|---:|---:|---:|
{chr(10).join(ape_rows)}

The unedited MT is the best row on three of the four pairs under both surface metrics, while
CometKiwi scores the system output above the unedited MT in eleven of the twelve system–pair cells. A post-edit is produced by editing the MT, so a
surface metric against post-edits measures residual repair effort rather than quality. **An APE
result on this data cannot be read without the do-nothing row.**

## Licence

Two licences apply and the per-row `sources` field says which. The three X→en pairs are MLQE-PE,
redistributed under its **CC0 1.0**. Everything else, including the English→Malayalam data, the
derived word-level tags, the MQM layer, the completed domain labels and the difficulty axes, is
**CC BY 4.0**. A configuration mixing the two carries the CC BY 4.0 condition, since CC0 adds none.

## Citation

Please cite the benchmark paper and the shared-task releases the data comes from.

```bibtex
{BIBTEX}
```

{UPSTREAM}
"""
    if check:
        cur_card = CARD.read_text(encoding="utf-8") if CARD.exists() else ""
        if cur_card != card:
            print("FAIL: the card on disk differs from what the data would produce")
            return 1
        print("card matches the data")
        return 0
    CARD.write_text(card, encoding="utf-8")
    print(f"wrote {CARD.relative_to(ROOT)}  ({len(card):,} chars, {card.count(chr(10))} lines)")
    return 0


UPSTREAM = ""   # filled in from dataset/templates/upstream_citations.bib at import time
_bib = ROOT / "dataset" / "templates" / "upstream_citations.bib"
if _bib.exists():
    UPSTREAM = "```bibtex\n" + _bib.read_text().strip() + "\n```"


if __name__ == "__main__":
    sys.exit(main())
