#!/usr/bin/env bash
# Run ONCE on the cluster login node (it has network). Installs the package,
# downloads the dataset to scratch, sanity-checks. Then 01_prefetch_models.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh
export HF_HUB_OFFLINE=0          # login node fetches; compute jobs read the cache offline

echo ">> repo at $IQE_ROOT ; scratch caches at $HF_HOME"

# make `conda activate` work in this non-interactive shell, then create the env if missing
CONDA_BASE="$(conda info --base 2>/dev/null)" && source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_IQE"; then
  echo ">> creating conda env '$CONDA_IQE' (python 3.10)"
  conda create -y -n "$CONDA_IQE" python=3.10
fi
conda activate "$CONDA_IQE"

echo ">> [1/4] torch 2.4.1+cu121 (cluster CUDA 12.06) — install first to avoid a double download"
python - <<'PY' || pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
import torch, sys; sys.exit(0 if torch.__version__.startswith("2.4.1") else 1)
PY

echo ">> [2/4] installing indicqe_ape package + hf extra (transformers/accelerate/bitsandbytes)"
pip install -e ".[hf]"

echo ">> [3/4] downloading dataset -> $INDICQEAPE_DATA (needs huggingface-cli login for the private repo)"
python scripts/download_data.py

python - <<'PY'                                          # sanity: loads offline
import os; os.environ["HF_HUB_OFFLINE"]="1"
from indicqe_ape.data import DataLoader
dl = DataLoader(); print("challenge test:", len(dl.load("challenge")))
print(dl.stats().to_string(index=False))
PY
echo "Setup OK. Next: bash cluster/01_prefetch_models.sh (open-weight model weights)."
