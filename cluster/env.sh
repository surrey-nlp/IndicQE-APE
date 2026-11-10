#!/usr/bin/env bash
# Central Slurm-cluster config — sourced by every cluster job. Every value below can be
# overridden from the environment; on a cluster set at least SCRATCH and the PART_* partitions.

# ---- paths (MUST be on a filesystem the compute nodes can see) -------------
# SCRATCH is the root for code, data, caches and outputs. Export it before sourcing this file
# (or before sbatch, which forwards the environment), e.g. `export SCRATCH=/path/to/scratch`.
#
# The API-only scripts (23_qe_rerun_closed.sh, run_gpt55_closed.sh) need no GPU and are often run
# from a laptop. When SCRATCH is unset outside a Slurm job, fall back to a repo-local scratch and
# point the data at the in-repo snapshot, so those scripts run without any configuration.
if [ -z "${SCRATCH:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "env.sh: SCRATCH is not set. Export SCRATCH=<compute-visible scratch dir> before submitting." >&2
  return 1 2>/dev/null || exit 1
fi
if [ -z "${SCRATCH:-}" ]; then
  # BASH_SOURCE is bash-only; env.sh is also sourced from an interactive zsh, where it is
  # unset and the repo path silently came out wrong, leaving INDICQEAPE_DATA unset and the
  # loader falling through to the Hub with HF_HUB_OFFLINE=1. $0 covers zsh.
  _IQE_SELF="${BASH_SOURCE[0]:-$0}"
  _IQE_REPO="$(cd "$(dirname "$_IQE_SELF")/.." && pwd)"
  export SCRATCH="$_IQE_REPO/.local-scratch"
  export IQE_ROOT="${IQE_ROOT:-$_IQE_REPO}"
  [ -d "$_IQE_REPO/dataset/hf_snapshot" ] && \
    export INDICQEAPE_DATA="${INDICQEAPE_DATA:-$_IQE_REPO/dataset/hf_snapshot}"
  # The cluster conda envs (indicqe, indicqe-vllm) do not exist off-cluster. For the API-only
  # scripts the local equivalent is indicqe-api, which carries openai + pandas + pyarrow and can
  # read the in-repo snapshot. Without this, activate() fails with EnvironmentNameNotFound and the
  # run stops before making a single call.
  export CONDA_IQE="${CONDA_IQE:-indicqe-api}"
  echo "env.sh: SCRATCH unset — local mode, SCRATCH=$SCRATCH, CONDA_IQE=$CONDA_IQE" >&2
fi
export SCRATCH                                              # cluster scratch (compute-visible)
export IQE_ROOT="${IQE_ROOT:-$SCRATCH/indicqe-ape}"        # git clone here (compute-visible)
export INDICQEAPE_DATA="${INDICQEAPE_DATA:-$SCRATCH/indicqeape/snapshot}"  # download-once snapshot
export IQE_OUT="${IQE_OUT:-$SCRATCH/indicqeape/out}"       # run outputs (regenerable)
export IQE_CACHE="${IQE_CACHE:-$SCRATCH/indicqeape/llmcache}"  # LLM response cache
export ALOPE_REPO="${ALOPE_REPO:-$SCRATCH/ALOPE-QE}"      # clone of the ALOPE repo (compute-visible)
export MTMETRIX_REPO="${MTMETRIX_REPO:-$SCRATCH/mt-metrix}"  # already on scratch
mkdir -p "$IQE_OUT" "$IQE_CACHE" logs

# ---- HuggingFace cache redirect (CRITICAL: $HOME is small & not compute-visible)
# Set ONLY HF_HOME: transformers then uses $HF_HOME/hub and datasets uses $HF_HOME/datasets.
# Do NOT set TRANSFORMERS_CACHE — it's deprecated AND misdirects the model lookup away from
# $HF_HOME/hub (where `hf download` puts weights), causing offline cache-miss failures.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
# Dataset is read as local parquet (no HF call). Models must be PRE-CACHED on the login
# node (cluster/01_prefetch_models.sh); offline then forces cache use and fails loudly.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"               # set 0 on a login node to fetch
export PYTHONPATH="$IQE_ROOT/src:${PYTHONPATH:-}"

# ---- partitions (site-specific; list yours with `sinfo -s`) ----------------
# Tiers map to GPU memory. Deliberately EMPTY by default: set them for your cluster
# (e.g. `export PART_ITER=<48GB-partition>`) and pass them at submit time with `-p "$PART_ITER"`.
export PART_ITER="${PART_ITER:-}"     # >=48GB: 3B/8B/14B + 4-bit 27B, ALOPE, metrics<=xxl
export PART_BIG="${PART_BIG:-}"       # 80GB: tower-72b, bf16 27B, headline runs
export PART_SMOKE="${PART_SMOKE:-}"   # >=24GB: smoke/debug
export FLAKY_NODE="${FLAKY_NODE:-}"   # optional node list to --exclude (empty = exclude nothing)
export GRES="${GRES:-gpu:1}"                    # partition restricts hardware; 1 GPU default
export CPUS="${CPUS:-8}"; export MEM="${MEM:-64G}"

# ---- conda envs (lowercase! verify with `conda env list`) ------------------
# Create `indicqe` for this package (conda create -n indicqe python=3.10; pip install -e .).
export CONDA_IQE="${CONDA_IQE:-indicqe}"        # this package (hf provider + ALOPE + metrics): torch 2.4.1+cu121
export CONDA_VLLM="${CONDA_VLLM:-indicqe-vllm}" # SEPARATE env for the vLLM provider (cu12 vllm build); 27-31B/multimodal/quantized
export CONDA_ALOPE="${CONDA_ALOPE:-alope-qe}"   # existing ALOPE env
export CONDA_MTMETRIX="${CONDA_MTMETRIX:-base}" # VERIFY: env you run mt-metrix in

# ---- API keys for closed-weight runs (export in your shell, not here) ------
#   export GOOGLE_API_KEY=...  OPENAI_API_KEY=...

# robust conda activation for non-interactive shells (no `conda init` needed)
activate () {
  # conda.sh and `conda activate` are not written for `set -u` (they reference unset variables) and
  # every job script here runs `set -euo pipefail`. Under that, activation aborts the whole script
  # at the conda call with no message -- which is exactly what happened on 2026-08-07: the trace
  # ended at "+ conda activate indicqe-api" and the run exited 1 having made no API calls.
  # Relax -e/-u for the activation only, then restore whatever was in force.
  local _flags="$-"
  set +eu
  local base; base="$(conda info --base 2>/dev/null)"
  [ -n "$base" ] && . "$base/etc/profile.d/conda.sh"
  conda activate "$1"
  local rc=$?
  case "$_flags" in *e*) set -e ;; esac
  case "$_flags" in *u*) set -u ;; esac
  if [ "$rc" -ne 0 ]; then
    echo "env.sh: could not activate conda env '$1'. Available:" >&2
    conda info --envs 2>/dev/null | awk 'NR>2 && NF {print "         " $1}' >&2
    echo "         Set CONDA_IQE / CONDA_VLLM to one of these, or create '$1'." >&2
    return 1
  fi
  return 0
}

