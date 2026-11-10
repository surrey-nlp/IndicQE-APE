#!/usr/bin/env bash
# Rebuild the vLLM env so its torch CUDA matches the cluster's GPU driver (CUDA 12.6).
# The old env had vllm 0.22 -> torch 2.11 / CUDA 13, which the 12.6 driver can't run
# ("NVIDIA driver too old, found 12060"). We pin a vLLM whose torch is cu126 and which
# still supports Sarvam's arches: sarvam-m (mistral3) + sarvam-translate (gemma3).
# Run on the LOGIN node (pip = data mgmt). GPU smoke is separate (srun). Idempotent.
set -uo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
VLLM_VER="${VLLM_VER:-0.9.2}"      # torch 2.7.0 / cu126 (<=12.6 driver OK); supports gemma3 + mistral3.
                                   # NB: vllm 0.10.2 pulls torch 2.8/cu128 = TOO NEW for this driver.
ENV="${CONDA_VLLM:-indicqe-vllm}"

echo ">> rebuilding $ENV with vLLM $VLLM_VER (cu126, driver-12.6 compatible)"
conda env remove -n "$ENV" -y 2>/dev/null || true
conda create -n "$ENV" python=3.10 -y
# activate inside a non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV"

pip install -q --upgrade pip
pip install -e .                      # core package (no comet -> no numpy/protobuf fight)
pip install "vllm==${VLLM_VER}"       # pulls its matching torch (cu126)
# vllm 0.9.2 declares transformers>=4.51.1 with NO upper bound, so pip grabs transformers 5.x, which
# clashes with vllm's model-config registration ("'aimv2' is already used"). Pin to a 4.53.x that
# vllm 0.9.2 was built against.
pip install "transformers==4.53.3"

echo ">> verifying torch <-> driver"
python - <<'PY'
import torch, vllm
cu = torch.version.cuda
print("vllm", vllm.__version__, "| torch", torch.__version__, "| torch.cuda", cu)
ok = cu is not None and tuple(int(x) for x in cu.split(".")) <= (12, 6)
print("CU_OK" if ok else "CU_TOO_NEW", "(need <= 12.6 for this driver)")
PY
echo ">> done. Next: GPU smoke via cluster/smoke_vllm.sbatch (see cluster/DEPLOY.md)."
