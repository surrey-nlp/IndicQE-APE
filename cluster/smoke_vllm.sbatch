#!/usr/bin/env bash
#SBATCH --job-name=iqe-smoke
##SBATCH --partition=<your-partition>   # pass -p at submit time (24 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
# Dummy-inference smoke: confirm the rebuilt cu126 vLLM env actually loads + generates on a GPU.
# Default model = sarvam-translate (4B, fits a 24GB GPU). Writes SMOKE_PASS/FAIL to the log.
# Submit:  sbatch -p "$PART_SMOKE" cluster/smoke_vllm.sbatch   (or SMOKE_MODEL=sarvamai/sarvam-m with -p "$PART_BIG")
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
activate "$CONDA_VLLM"
M="${SMOKE_MODEL:-sarvamai/sarvam-translate}"
echo ">> smoke $M on $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__)"
VLLM_TP="${VLLM_TP:-1}" python -m indicqe_ape.cli qe --provider vllm --model "$M" --pair enhi \
  --subset challenge --shots 0 --limit 3 --out /tmp/smoke_$$.jsonl --cache-dir /tmp/smk_$$ \
  && { n=$(grep -c '"task"' /tmp/smoke_$$.jsonl); echo "SMOKE_PASS records=$n"; sed -n '2p' /tmp/smoke_$$.jsonl; } \
  || echo "SMOKE_FAIL"
rm -rf /tmp/smoke_$$.jsonl /tmp/smk_$$
