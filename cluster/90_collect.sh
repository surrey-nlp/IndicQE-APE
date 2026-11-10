#!/usr/bin/env bash
# Aggregate run summaries + challenge diagnostics (login node, no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh
export HF_HUB_OFFLINE=1
activate "$CONDA_IQE"
python scripts/aggregate_results.py "$IQE_OUT" 2>/dev/null || \
  echo "(aggregate_results.py: verify args; reads _summary lines from run jsonl)"
python scripts/analyse_challenge.py --root "$IQE_OUT" --out "$IQE_OUT/analysis"
echo "results + diagnostics under $IQE_OUT (and $IQE_OUT/analysis)"
echo "Copy $IQE_OUT/analysis off scratch (rsync/scp) before it is purged."
