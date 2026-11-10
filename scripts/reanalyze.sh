#!/usr/bin/env bash
# One-command re-analysis. Run this whenever NEW model results land (e.g. after the Sarvam batch or
# the 4-shot 3B batch). It re-derives every pred from the stored raw text with the hardened parser,
# then regenerates ALL diagnostics + the full meta-evaluation + APE metrics.
#
# Requires: the indicqe env active (or the Mac .venv) and INDICQEAPE_DATA set to the dataset snapshot.
# Usage:  INDICQEAPE_DATA=... bash scripts/reanalyze.sh [results_dir]   (default results_dir=results)
set -uo pipefail
R="${1:-results}"
: "${INDICQEAPE_DATA:?set INDICQEAPE_DATA to the dataset snapshot dir (has data/challenge/...)}"
PY="${PYTHON:-python}"

echo "================ RE-ANALYSIS over $R ================"
echo "== 1/6 audit raw responses (parse QA per model) =="
$PY scripts/audit_responses.py "$R" 2>/dev/null | grep -A14 PER-MODEL || true
echo "== 2/6 reparse preds from raw (hardened, reasoning-aware parser) =="
$PY scripts/reparse_responses.py "$R"
echo "== 3/6 analyse_challenge (diagnostics + prompt-sensitivity + joint QE-APE) =="
$PY scripts/analyse_challenge.py --root "$R" --out "$R/analysis"
echo "== 4/6 meta-evaluation zero-shot (segment P/S/Kendall + system Pearson/PA/SPA + prompt robustness) =="
$PY scripts/meta_evaluation.py "$R" 0
echo "== 5/6 meta-evaluation 4-shot =="
$PY scripts/meta_evaluation.py "$R" 4
echo "== 6/6 APE metrics (chrF++ / char-TER, Indic-correct) =="
$PY scripts/ape_metrics.py "$R" || echo "  (no APE outputs yet)"
echo "================ DONE — CSVs in $R/analysis/ ================"
echo "  qe_diagnostics.csv  qe_prompt_sensitivity.csv  qe_meta_s0.csv  qe_meta_s4.csv  ape_metrics_table.csv  joint_qe_ape.csv"
