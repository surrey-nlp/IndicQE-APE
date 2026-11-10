#!/usr/bin/env bash
# run_all_checks.sh -- every generator in --check mode plus every verifier, in one pass.
#
# Written because invoking these ad hoc from a shell loop kept mis-splitting the "--check"
# argument under zsh, which silently reported FAIL for a reason that had nothing to do with
# the paper. Run this instead; it fails loudly and only for real drift.
#
#   bash analysis/run_all_checks.sh [python-interpreter]
#
# Exit status is 0 only if every check passes.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${1:-python3}"
if ! "$PY" -c "import pandas, scipy, numpy" 2>/dev/null; then
  for cand in "$(command -v python3 || true)" \
              "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" \
              "$HOME/opt/anaconda3/bin/python" /usr/bin/python3; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import pandas, scipy, numpy" 2>/dev/null; then PY="$cand"; break; fi
  done
fi
echo "interpreter: $PY"
"$PY" -c "import pandas,scipy,numpy,sys;print('  pandas',pandas.__version__,'scipy',scipy.__version__,'py',sys.version.split()[0])" || {
  echo "FATAL: no interpreter with pandas+scipy found"; exit 1; }
echo

# The paper's LaTeX source lives in its own repository and is optional here. Checks that compare a
# table or a prose figure against it are skipped, not failed, when it is absent: a missing optional
# input is not drift. analysis/paper_source.py documents where it is looked for.
PAPER_TEX="$("$PY" -c 'import sys; sys.path.insert(0,"analysis"); from paper_source import find_tex; t=find_tex(); print(t or "")' 2>/dev/null)"
if [ -n "$PAPER_TEX" ]; then
  echo "paper source: $PAPER_TEX"
else
  echo "paper source: not present -- checks against the .tex will be SKIPPED"
  echo "              (git clone https://github.com/dipteshkanojia/IndicQEAPE.git paper)"
fi
echo

fail=0
skipped=0
run() {                       # run <label> <script> [args...]
  local label="$1"; shift
  printf '  %-44s' "$label"
  if "$PY" "$@" >/tmp/iqe_check.log 2>&1; then
    if grep -q '^SKIP:' /tmp/iqe_check.log; then echo SKIP; skipped=$((skipped+1)); else echo PASS; fi
  else
    echo FAIL
    sed 's/^/      /' /tmp/iqe_check.log | tail -6
    fail=1
  fi
}

# run_tex <label> <script> [args...] -- only meaningful with the paper source present
run_tex() {
  local label="$1"; shift
  if [ -z "$PAPER_TEX" ]; then
    printf '  %-44s%s\n' "$label" "SKIP (no paper source)"; skipped=$((skipped+1)); return
  fi
  run "$label" "$@"
}

echo "GENERATORS (--check: fail if the .tex does not match the data)"
run_tex "tab:qe + tab:perpairfull"  analysis/make_qe_tables.py --check
run_tex "tab:prompting + tab:parse4" analysis/make_prompting_tables.py --check
run_tex "tab:ape"                    analysis/make_ape_table.py --check
run_tex "tab:splits"                 analysis/make_split_table.py --check
run_tex "tab:iaa"                    analysis/annotator_agreement.py --check
run_tex "GEMBA-DA vs SQM numbers"    analysis/template_compare.py --check
run_tex "pair factors + tab:pairfactors" analysis/pair_factors.py --check
run_tex "length baseline numbers"    analysis/length_baseline.py --check
run "probe vs prompt"            analysis/probe_vs_prompt.py --check
run "subword fertility"          analysis/subword_fertility.py --check
run "model language coverage"    analysis/model_language_coverage.py --check
run "model order + domain"       analysis/model_order_and_domain.py --check
run_tex "custom.bib"                 analysis/make_bib.py --check
run_tex "HF dataset card"            dataset/make_card.py --check
run_tex "repository README"          analysis/make_readme.py --check

echo
echo "VERIFIERS"
run_tex "axis contrast + its 4 tables" analysis/axis_contrast.py --check
run_tex "axis reliability"           analysis/axis_reliability.py --check
run_tex "axis contrast on the head"  analysis/axis_head_contrast.py --check
run_tex "COMET head + probe numbers"  analysis/verify_head_numbers.py
run_tex "tab:fullda + its population"  analysis/verify_fullda.py
run "tab:composition (81 cells)" scripts/verify_tab_composition.py
run_tex "tab:qe structure"           scripts/verify_tab_qe.py
run "duplication + split hygiene" scripts/audit_duplication.py
run "release schema"             dataset/apply_axes_to_release.py --check

echo
if [ "$fail" -eq 0 ]; then if [ "$skipped" -gt 0 ]; then echo "ALL CHECKS PASS ($skipped skipped: paper source absent)"; else echo "ALL CHECKS PASS"; fi; else echo "*** SOME CHECKS FAILED ***"; fi
exit "$fail"
