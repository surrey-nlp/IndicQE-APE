# IndicQE-APE

**IndicQE-APE: A Consolidated Benchmark for Quality Estimation and Automatic Post-Editing over Indic Languages** — WMT 2026 (Eleventh Conference on Machine Translation)

[Paper (ACL Anthology)](https://aclanthology.org/2026.wmt-main.19/) · [arXiv:2608.16344](https://arxiv.org/abs/2608.16344) ·
[Dataset on the Hugging Face Hub](https://huggingface.co/datasets/surrey-nlp/IndicQE-APE) · [Poster](poster/IndicQE-APE_WMT26_poster.pdf)

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

Nine pairs: en→{hi, mr, ta, te, gu, ml} and {et, ne, si}→en. Configurations are `full`, `all`,
`qe-da`, `ape`, `word-qe`, `explainable-qe`, `mqm`, `challenge`, and one per language pair. The
[dataset card](https://huggingface.co/datasets/surrey-nlp/IndicQE-APE) documents every field, configuration and count.

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
| BharatGPT-3B | 0.134 | 0.098 | 0.075 | -0.46 |
| Llama-3.2-3B | 0.256 | 0.217 | 0.172 | 0.86 |
| tiny-aya (tiny-aya-fire, 3.4B) | 0.311 | 0.290 | 0.232 | 0.93 |
| aya-expanse-32B | 0.404 | 0.404 | 0.321 | 0.91 |
| sarvam-m (24B) | 0.455 | 0.440 | 0.344 | 0.91 |
| GPT-5.5 | 0.627 | 0.623 | 0.461 | 0.94 |
| CometKiwi-DA | 0.625 | 0.621 | 0.452 | 0.78 |
| CometKiwi-XL | 0.680 | 0.671 | 0.497 | -0.21 |
| XCOMET-XL (QE mode) | 0.498 | 0.498 | 0.355 | 0.61 |

CometKiwi-XL has the highest within-language correlation of any system and the weakest cross-lingual agreement of the three trained metrics (-0.21), so raw reference-free scores are not comparable across language pairs.

Automatic post-editing on the four pairs with system outputs (Table 9). `do-nothing` is the unedited
MT scored against the human post-edit.

| system | pair | char-TER ↓ | chrF++ ↑ | CometKiwi ↑ |
|---|---|---:|---:|---:|
| do-nothing | en-hi | 44.5 | 55.4 | — |
|  | en-mr | 26.1 | 74.8 | — |
|  | en-ta | 43.4 | 64.2 | — |
|  | en-ml | 31.1 | 78.1 | — |
| sarvam-m (24B) | en-hi | 37.6 | 58.9 | 0.819 |
|  | en-mr | 36.5 | 63.9 | 0.743 |
|  | en-ta | 54.5 | 54.1 | 0.826 |
|  | en-ml | 44.7 | 60.0 | 0.862 |
| sarvam-t (sarvam-translate) | en-hi | 34.4 | 62.5 | 0.829 |
|  | en-mr | 42.5 | 59.0 | 0.757 |
|  | en-ta | 63.0 | 49.1 | 0.841 |
|  | en-ml | 49.0 | 53.4 | 0.872 |
| tiny-aya (tiny-aya-fire, 3.4B) | en-hi | 46.7 | 55.5 | 0.782 |
|  | en-mr | 36.7 | 70.6 | 0.697 |
|  | en-ta | 57.0 | 56.5 | 0.771 |
|  | en-ml | 37.8 | 71.1 | 0.802 |

The unedited MT is the best row on three of the four pairs under both surface metrics, while
CometKiwi scores the system output above the unedited MT in eleven of the twelve system–pair cells.
An APE result on post-edit references cannot be read without the do-nothing row.

## Run the baselines

```bash
python -m indicqe_ape.cli qe  --provider openai --model gpt-5.5 \
       --pair enhi --subset qe-da --shots 0 --out results/qe_enhi.jsonl
python -m indicqe_ape.cli ape --provider hf     --model sarvamai/sarvam-m \
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
@inproceedings{kanojia-etal-2026-indicqe,
    title = "{I}ndic{QE}-{APE}: A Consolidated Benchmark for Quality Estimation and Automatic Post-Editing over {I}ndic Languages",
    author = "Kanojia, Diptesh  and
      Sindhujan, Archchana  and
      Deoghare, Sourabh  and
      Sokova, Daria  and
      Qian, Shenbin  and
      Koushik, Girish  and
      Ranasinghe, Tharindu  and
      Orasan, Constantin  and
      Zerva, Chrysoula  and
      Rei, Ricardo  and
      Blain, Frederic  and
      Martins, Andr{\'e}  and
      Turchi, Marco  and
      Negri, Matteo  and
      Kunchukuttan, Anoop  and
      Khapra, Mitesh M.  and
      Bhattacharyya, Pushpak",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Eleventh Conference on Machine Translation",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wmt-main.19/",
    doi = "10.18653/2026.wmt-main.19",
    pages = "318--344",
    ISBN = "979-8-89176-474-3"
}
```

The benchmark consolidates the WMT QE and APE shared tasks and MLQE-PE; the upstream releases are
listed with their BibTeX on the [dataset card](https://huggingface.co/datasets/surrey-nlp/IndicQE-APE) and should be cited alongside it.

Questions and issues: open a GitHub issue, or write to d.kanojia@surrey.ac.uk.
