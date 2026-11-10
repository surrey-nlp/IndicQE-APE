# analysis/

Scripts that turn model outputs into the paper's tables, figures and checked figures. They read
the per-instance outputs under `results/` and `experiments/comet_qe/results/`, the aggregate CSVs
under `results/analysis/` and `evidence/`, and the released data; they never re-run a model.

## How the scripts relate to the paper

- **Table generators** (`make_*_table*.py`, `make_qe_tables.py`, `make_prompting_tables.py`,
  `make_split_table.py`, `annotator_agreement.py`, `axis_contrast.py`, `pair_factors.py`, …) write
  their table body between `% BEGIN GENERATED <tag>` / `% END GENERATED <tag>` markers in the
  paper's LaTeX source, and write the underlying numbers to `results/analysis/*.csv`.
  Run with `--check` to compare instead of write; the check fails if the paper and the data
  disagree.
- **Verifiers** (`verify_*.py`, `audit_all_numbers.py`) recompute prose figures and trace numbers
  in the paper back to a result file.
- **Generated documents**: `make_readme.py` writes `README.md` and `CITATION.cff`;
  `../dataset/make_card.py` writes the Hugging Face dataset card. Both take the citation from
  `citation.py`.
- `run_all_checks.sh` runs every generator in `--check` mode plus every verifier and exits
  non-zero on any drift.

The paper's LaTeX source is not part of this repository. Scripts that need it find it through
`paper_source.py` (`$INDICQE_PAPER_TEX`, `$INDICQE_PAPER_DIR`, `./paper/` or `../IndicQEAPE/`)
and are skipped when it is absent; everything else runs on a plain clone.

## Conventions

- One script, one artefact; name the script after what it produces.
- Read inputs from `results/`, `evidence/` and the release; paths are relative to the repository
  root, so the same command works on any clone.
- Print a short manifest at the end: inputs consumed, outputs written, rows skipped and why.
