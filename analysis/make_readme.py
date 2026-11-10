#!/usr/bin/env python3
r"""make_readme.py — write the repository README from the release and the paper's tables.

The results come out of the generated LaTeX in the paper and the citation out of
analysis/citation.py, so neither can drift from what the paper reports. The setup guide lives in docs/SETUP.md and is not touched.

    python3 analysis/make_readme.py
    python3 analysis/make_readme.py --check
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
from make_card import MODEL, qe_takeaway, texrows  # noqa: E402

README = ROOT / "README.md"
CITATION = ROOT / "CITATION.cff"
sys.path.insert(0, str(ROOT / "analysis"))
from citation import ANTHOLOGY_URL, ARXIV, ARXIV_URL, BIBTEX, HF_URL as HF, TITLE, cff  # noqa: E402


def main() -> int:
    check = "--check" in sys.argv
    qrows = texrows("tab_qe_body")
    qe = [f"| {MODEL.get(r[0], r[0])} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in qrows]
    ape, cur = [], None
    for r in texrows("tab_ape"):
        if len(r) == 5:
            cur, r = MODEL.get(r[0].strip(), r[0]), r[1:]
        ape.append(f"| {cur} | {r[0]} | {r[1]} | {r[2]} | {r[3]} |")

    txt = f"""# IndicQE-APE

**{TITLE}** — WMT 2026 (Eleventh Conference on Machine Translation)

[Paper (ACL Anthology)]({ANTHOLOGY_URL}) · [arXiv:{ARXIV}]({ARXIV_URL}) ·
[Dataset on the Hugging Face Hub]({HF}) · [Poster](poster/IndicQE-APE_WMT26_poster.pdf)

Quality estimation (QE) and automatic post-editing (APE) for Indic and other low-resource language
pairs: **126,754 instances over nine directional pairs**, consolidated from the WMT 2020–2024 QE and
APE shared-task lineage with an extended English→Malayalam resource. Up to four label types are
aligned on the same segment — a direct assessment, a human post-edit, word-level OK/BAD tags and an
error explanation — so QE, APE and explainable QE can be studied without re-aligning corpora.

This repository holds the evaluation code, the curation and verification scripts, the model outputs
and the analysis behind the numbers in the paper. The data itself is on the Hub.

## Install and use the data

```bash
git clone https://github.com/surrey-nlp/IndicQE-APE.git && cd IndicQE-APE
pip install -e .          # Python >= 3.10; full guide for a fresh machine: docs/SETUP.md
```

```python
from datasets import load_dataset

qe   = load_dataset("surrey-nlp/IndicQE-APE", "qe-da")       # source, MT, DA score
ape  = load_dataset("surrey-nlp/IndicQE-APE", "ape")         # source, MT, human post-edit
hard = load_dataset("surrey-nlp/IndicQE-APE", "challenge")   # the curated test set
```

Nine pairs: en→{{hi, mr, ta, te, gu, ml}} and {{et, ne, si}}→en. Configurations are `full`, `all`,
`qe-da`, `ape`, `word-qe`, `explainable-qe`, `mqm`, `challenge`, and one per language pair. The
[dataset card]({HF}) documents every field, configuration and count.

## Difficulty axes

Every segment in the test set carries a flag on each difficulty axis its labels can express. Each
axis is a function of released columns alone. Every threshold is taken **within language pair**,
except A1's [35, 75] band on the 0–100 DA scale, which is the same for every pair.

| flag | fires when |
|---|---|
| `axis_a1` | per-annotator DA standard deviation in the pair's top quartile, mean in [35, 75] |
| `axis_a2` | MT→post-edit token TER in the pair's top quartile |
| `axis_a3` | BAD-tag density in the pair's top quartile, or a mistranslation, untranslated or addition error |
| `axis_a4` | the holistic score and the token-level evidence disagree; `axis_a4_arm` says which way |

An axis a pair has no signal for is null, not false. The flags overlap, so they ship as four
booleans and a count (`n_axes`) rather than one label.

```python
hard = load_dataset("surrey-nlp/IndicQE-APE", "challenge")["test"]
conflict = hard.filter(lambda r: r["axis_a4"] is True)
```

## Results

Segment-level QE on the DA-bearing challenge test, nine pairs, zero-shot (Table 4 of the paper).
The first three columns are within-language, each the mean over the nine pairs; the last is the
Pearson correlation between per-pair model means and per-pair human means.

| system | Pearson r | Spearman ρ | Kendall τ | cross-ling. r |
|---|---:|---:|---:|---:|
{chr(10).join(qe)}

{qe_takeaway(qrows)}

Automatic post-editing on the four pairs with system outputs (Table 9). `do-nothing` is the unedited
MT scored against the human post-edit.

| system | pair | char-TER ↓ | chrF++ ↑ | CometKiwi ↑ |
|---|---|---:|---:|---:|
{chr(10).join(ape)}

The unedited MT is the best row on three of the four pairs under both surface metrics, while
CometKiwi scores the system output above the unedited MT in eleven of the twelve system–pair cells.
An APE result on post-edit references cannot be read without the do-nothing row.

## Run the baselines

```bash
python -m indicqe_ape.cli qe  --provider openai --model gpt-5.5 \\
       --pair enhi --subset qe-da --shots 0 --out results/qe_enhi.jsonl
python -m indicqe_ape.cli ape --provider hf     --model sarvamai/sarvam-m \\
       --pair enmr --subset ape   --shots 0 --out results/ape_enmr.jsonl
python -m indicqe_ape.cli stats
```

Providers are `openai`, `anthropic`, `google`, `cohere`, `hf` and `vllm`; each is available once its
client library is installed (see `pyproject.toml` extras), and API keys are read from the
environment (`.env.example`). Batch manifests are in `configs/`, Slurm job scripts in `cluster/`
(`cluster/DEPLOY.md`), and the full setup guide in `docs/SETUP.md`.

## Reproduce the paper

```bash
python3 dataset/build_axes.py            # recompute the difficulty axes over the release
python3 analysis/axis_contrast.py        # the axis result of the paper, plus its tables
bash    analysis/run_all_checks.sh       # every generator and verifier; fails on drift
python3 analysis/audit_all_numbers.py    # trace every number in the paper to a result file
```

`analysis/run_all_checks.sh` regenerates nothing but fails if any table, or any of the checked
prose figures, has drifted from the data. Checks that compare against the paper's LaTeX source
need that source on disk (see `analysis/paper_source.py`); without it they are skipped, and the
remaining checks run on the CSVs and model outputs in this repository.

The per-instance model outputs live under `results/` and `experiments/comet_qe/results/`. Three
output sets used by the QE tables are still to be added: the GPT-5.5 prompting runs
(`results/qe_closed`, `results/qe_closed_rerun`) and the full-set learned-metric scores
(`results/qe_metrics_full`). Until they are, the checks that read them report missing inputs.

## Layout

```
src/          the indicqe_ape package and its CLI
analysis/     scripts that turn results into the paper's tables and figures
scripts/      release verification: composition, duplication, leakage, tag import
dataset/      curation, the axis definition applied to the release, and the dataset card
experiments/  the COMET-encoder regression head (configs, training code, run outputs)
configs/      experiment manifests
cluster/      Slurm job scripts for the scoring and training runs, configured by env vars
tests/        unit tests, including the strict score parser
evidence/     aggregate CSVs from the layer probe, read by the analysis scripts
results/      model outputs and the aggregate CSVs behind the paper's tables
docs/         SETUP.md, the fresh-machine guide
poster/       the WMT 2026 poster
```

Two directories hold one-off data-preparation code kept for provenance rather than for reuse:
`deepseek-enml/`, which generated the model-written English–Malayalam error descriptions, and
`gpt-enml-qe/`, which ran GPT-5.5 QE and APE on the English–Malayalam test split.

## Verification scripts

| script | what it does |
|---|---|
| `scripts/verify_tab_composition.py` | re-derives every per-pair and per-label count from the live repo |
| `scripts/audit_duplication.py` | asserts the `uid`↔content bijection and zero train/test leakage |
| `scripts/verify_tab_qe.py` | recomputes the QE baseline correlations cell by cell |
| `scripts/validate_release.py` | full release validator: configs, uids, provenance, counts, leakage |
| `dataset/apply_axes_to_release.py` | brings every config onto the released schema; refuses on any row change |

## Licence

Code is released under the Apache License 2.0 (`LICENSE`). The data carries two licences and the
per-row `sources` field says which: the three X→en pairs are MLQE-PE under **CC0 1.0**, everything
else is **CC BY 4.0**.

## Acknowledgements

IndicQE-APE was made possible by its data sponsors and collaborating institutions: the
People-Centred AI Institute, University of Surrey; Computation for Indian Language Technology
(CFILT), IIT Bombay; and the European Association for Machine Translation (EAMT). We thank the
upstream data providers whose corpora this benchmark consolidates — the WMT Quality Estimation
shared tasks (2020, 2022, 2023, 2024), the WMT Automatic Post-Editing shared tasks, MLQE-PE, and the
annotation teams behind the English–Hindi MQM and English–Malayalam QE data — and Rajen Chatterjee
(Apple Inc.) for his work towards creating the APE data. We also thank the data annotation agencies
Zibanka Media and Techliebe for their help with human annotation over the years.

## Citation

```bibtex
{BIBTEX}
```

The benchmark consolidates the WMT QE and APE shared tasks and MLQE-PE; the upstream releases are
listed with their BibTeX on the [dataset card]({HF}) and should be cited alongside it.

Questions and issues: open a GitHub issue, or write to d.kanojia@surrey.ac.uk.
"""
    cffs = cff()
    if check:
        if CITATION.exists() and CITATION.read_text(encoding="utf-8") != cffs:
            print("FAIL: CITATION.cff differs from what analysis/citation.py would produce")
            return 1
        if README.read_text(encoding="utf-8") != txt:
            print("FAIL: README.md differs from what the data would produce")
            return 1
        print("README matches the data")
        return 0
    README.write_text(txt, encoding="utf-8")
    CITATION.write_text(cffs, encoding="utf-8")
    print(f"wrote README.md ({len(txt):,} chars, {txt.count(chr(10))} lines) and CITATION.cff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
