#!/usr/bin/env bash
# ape_comet_local.sh -- CometKiwi over the regenerated APE post-edits, on a single-GPU workstation.
#
# WHY: tab:ape moved onto the current challenge test when the cluster ape_regen job landed
# (en-hi 1,536 / en-mr 2,098 / en-ta 1,448). The CometKiwi figures quoted in S7 for tiny-aya
# (0.781 / 0.701 / 0.770) were computed over the OLD post-edits at n = 1,490 / 2,024 / 1,375 --
# hypotheses that no longer exist in the table. The ape_regen sbatch used --no-comet because the
# neural pass is a separate second stage, so it was never run. This runs it.
#
# An env with an old unbabel-comet can be broken (e.g. Python 3.8 with a torchmetrics too new for
# it: MulticlassMatthewsCorrCoef import fails), so the script probes candidate interpreters ($PY if
# set, then common conda envs, then python3) and uses the first one where `import comet, torch`
# succeeds AND CUDA is visible. It refuses to run on CPU: a
# CometKiwi pass over ~5,000 segments on CPU takes hours and silently looks like progress.
#
# If the weights are already in the HF cache (models--Unbabel--wmt22-cometkiwi-da), no download
# is needed.
#
# Usage (from the repo root; GPU defaults to 0):
#   nohup bash cluster/ape_comet_local.sh > logs/ape_comet.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
mkdir -p logs results/ape_metrics_regen

# Overridable so the same runner can do the en-ml cells, which live in ape_open rather than
# ape_regen (en-ml was already on the current 1,436-row population and was deliberately excluded
# from the regeneration). Stage them into their own directory first: ape_open also holds the
# SUPERSEDED en-hi/mr/ta cells on the old test, and scoring those would silently mix populations.
APE_DIR="${APE_DIR:-$ROOT/results/ape_regen}"
OUT_DIR="${OUT_DIR:-$ROOT/results/ape_metrics_regen}"
GPU="${GPU:-0}"

echo "=== $(date -Is)  host=$(hostname)  root=$ROOT"
echo "ape dir : $APE_DIR ($(ls "$APE_DIR"/*.jsonl 2>/dev/null | wc -l) cells)"
echo "out dir : $OUT_DIR"

# ---- pick an interpreter -----------------------------------------------------------------------
CANDIDATES=()
[ -n "${PY:-}" ] && CANDIDATES+=("$PY")
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ]; then
  for e in indicqe indicqe-vllm indicqe-comet comet; do CANDIDATES+=("$CONDA_BASE/envs/$e/bin/python"); done
  CANDIDATES+=("$CONDA_BASE/bin/python")
fi
CANDIDATES+=("$(command -v python3 || true)")
PY=""
for cand in "${CANDIDATES[@]}"; do
  [ -x "$cand" ] || continue
  printf '  probe %-58s ' "$cand"
  if out=$(CUDA_VISIBLE_DEVICES="$GPU" "$cand" - <<'PYCHK' 2>&1
import sys
try:
    import comet, torch
except Exception as e:
    print(f"NO ({type(e).__name__}: {str(e)[:70]})"); sys.exit(1)
if not torch.cuda.is_available():
    print(f"no CUDA (comet {comet.__version__})"); sys.exit(1)
print(f"OK comet {comet.__version__} torch {torch.__version__} {torch.cuda.get_device_name(0)}")
PYCHK
  ); then
    echo "$out"; PY="$cand"; break
  else
    echo "$out"
  fi
done

if [ -z "$PY" ]; then
  echo
  echo "FATAL: no environment has a working 'comet' with CUDA."
  echo "Create one with:"
  echo "  conda create -y -n indicqe-comet python=3.11 && \\"
  echo "  conda run -n indicqe-comet pip install unbabel-comet torch"
  echo "or point PY at an interpreter that has them."
  exit 1
fi
echo "using: $PY"
echo

# ---- run ---------------------------------------------------------------------------------------
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/run_comet_metrics.py \
  --ape "$APE_DIR" --out "$OUT_DIR" --gpus 1 --batch-size 16
rc=$?

echo
echo "=== exit=$rc  $(date -Is)"
if [ "$rc" -eq 0 ]; then
  echo "--- results ---"
  for f in "$OUT_DIR"/*.json; do
    [ -e "$f" ] || continue
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"  {d.get('file','?'):<34} n={d.get('n','?'):>5}  cometkiwi={d.get('cometkiwi_mean',float('nan')):.4f}\")" "$f"
  done
  touch "$OUT_DIR/.DONE"
fi
exit "$rc"
