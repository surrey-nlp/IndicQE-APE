#!/usr/bin/env bash
#SBATCH --job-name=iqe-all
##SBATCH --partition=<your-partition>   # pass -p at submit time (80 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#
# ONE-JOB orchestrator: runs ALL open-weight server experiments sequentially on a single A100,
# one model after another, for up to 3 days. Fully RESUMABLE — on any failure / timeout / node
# death just re-submit the SAME job; every step skips work that already has a _summary, and every
# cli call resumes from its JSONL + cache. (GPT-5.5 closed runs happen locally via run_gpt55_closed.sh, not here.)
#
# Phases (env-switched): 1) 3B QE (hf)  2) 24-32B QE (vLLM)  3) trained metrics (COMET)
#                        4) APE generate  5) COMET over post-edits  6) collect + analyse
# Toggle with env: RUN_QE3B / RUN_QEBIG / RUN_METRICS / RUN_APE / RUN_COLLECT (default all =1);
#                  DO_4SHOT=1 adds 4-shot for the finalists. ALOPE is separate (30/31_alope).
#
# Prereqs: indicqe + indicqe-vllm envs built; ALL models + the 3 COMET metrics PRE-CACHED on the
# login node (bash cluster/01_prefetch_models.sh, with gated licences accepted). Compute is offline.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
export HF_HUB_OFFLINE=1
mkdir -p "$IQE_OUT/qe_open" "$IQE_OUT/ape_open" logs
PAIRS=(enhi enmr enta ente engu enml eten neen sien); PE=(enhi enmr enta)
HF3B=(CoRover/BharatGPT-3B-Indic CohereLabs/tiny-aya-fire meta-llama/Llama-3.2-3B-Instruct)
# vLLM env is pinned to cu126 (driver 12.6). That supports Sarvam's arches: sarvam-m (mistral3) +
# sarvam-translate (gemma3). Qwen3.6-27B / gemma-4-31B are cu13-only arches -> can't run on this
# driver; excluded (need a node with a newer driver). sarvam-translate is QE-checked here too
# (expected weak — translation-tuned, not a judge). All vLLM models run enforce_eager.
BIG=(sarvamai/sarvam-m sarvamai/sarvam-translate)
# sarvam-m is a hybrid reasoning model: without this it emits a chain of thought before every
# score, at ~28s/item against ~1s, and returns a different score. Set IQE_THINK=1 to opt back in.
[ "${IQE_THINK:-0}" = 1 ] || export IQE_NO_THINK=1
probe(){ # skip a vLLM model entirely if it can't even load (avoids 8 failed pair attempts)
  python -m indicqe_ape.cli qe --provider vllm --model "$1" --pair enhi --subset challenge --shots 0 \
    --limit 2 --out "$IQE_OUT/qe_open/_probe.${1##*/}.jsonl" --cache-dir "$IQE_CACHE/_probe.${1##*/}" >/dev/null 2>&1
  local rc=$?; rm -f "$IQE_OUT/qe_open/_probe.${1##*/}.jsonl"; return $rc
}
: "${RUN_QE3B:=1}" "${RUN_QEBIG:=1}" "${RUN_METRICS:=1}" "${RUN_APE:=1}" "${RUN_COLLECT:=1}" "${DO_4SHOT:=1}"

qe(){ # provider model pair shots
  local out="$IQE_OUT/qe_open/${2##*/}.s${4}.${3}.jsonl"
  if grep -q '"_summary"' "$out" 2>/dev/null; then echo "  skip $(basename "$out")"; return; fi
  echo ">>> $(date +%H:%M) QE $1 ${2##*/} $3 s$4"
  python -m indicqe_ape.cli qe --provider "$1" --model "$2" --pair "$3" --subset challenge \
    --shots "$4" --out "$out" --cache-dir "$IQE_CACHE/${2##*/}.s${4}.${3}" || echo "  !! FAIL QE ${2##*/} $3 s$4"
}
ape(){ # provider model pair
  local out="$IQE_OUT/ape_open/${2##*/}.s0.${3}.jsonl"
  if grep -q '"_summary"' "$out" 2>/dev/null; then echo "  skip $(basename "$out")"; return; fi
  echo ">>> $(date +%H:%M) APE $1 ${2##*/} $3"
  python -m indicqe_ape.cli ape --provider "$1" --model "$2" --pair "$3" --subset challenge \
    --shots 0 --no-comet --out "$out" --cache-dir "$IQE_CACHE/ape.${2##*/}.${3}" || echo "  !! FAIL APE ${2##*/} $3"
}
mt(){ # provider model pair  — dedicated-MT retranslate (e.g. sarvam-translate); scored like APE
  local out="$IQE_OUT/ape_open/${2##*/}.mt.${3}.jsonl"
  if grep -q '"_summary"' "$out" 2>/dev/null; then echo "  skip $(basename "$out")"; return; fi
  echo ">>> $(date +%H:%M) MT-retranslate $1 ${2##*/} $3"
  python -m indicqe_ape.cli ape --provider "$1" --model "$2" --pair "$3" --subset challenge \
    --shots 0 --no-comet --mode retranslate --out "$out" --cache-dir "$IQE_CACHE/mt.${2##*/}.${3}" || echo "  !! FAIL MT ${2##*/} $3"
}

if [ "$RUN_QE3B" = 1 ]; then
  echo "######## PHASE 1/6: 3B QE (hf) ########"; activate "$CONDA_IQE"
  for m in "${HF3B[@]}"; do for p in "${PAIRS[@]}"; do qe hf "$m" "$p" 0; done; done
  [ "$DO_4SHOT" = 1 ] && for p in "${PAIRS[@]}"; do qe hf CohereLabs/tiny-aya-fire "$p" 4; done
fi

if [ "$RUN_QEBIG" = 1 ]; then
  echo "######## PHASE 2/6: 24-32B QE (vLLM, TP=1 on the a100) ########"
  activate "$CONDA_VLLM"; export VLLM_TP="${VLLM_TP:-1}"
  for m in "${BIG[@]}"; do
    if ! probe "$m"; then echo "  !! ${m##*/} vLLM load FAILED — skipping its pairs (check arch support / CUDA)"; continue; fi
    for p in "${PAIRS[@]}"; do qe vllm "$m" "$p" 0; done
  done
  [ "$DO_4SHOT" = 1 ] && probe sarvamai/sarvam-m && for p in "${PAIRS[@]}"; do qe vllm sarvamai/sarvam-m "$p" 4; done
fi

if [ "$RUN_METRICS" = 1 ]; then
  echo "######## PHASE 3/6: trained metrics (CometKiwi-da/-xl, XCOMET-XL) ########"; activate "$CONDA_IQE"
  python scripts/run_comet_metrics.py --out "$IQE_OUT/qe_metrics" --gpus 1 --batch-size 16 || echo "  !! metrics failed"
fi

if [ "$RUN_APE" = 1 ]; then
  echo "######## PHASE 4/6: APE generation (finalists x PE pairs) ########"
  activate "$CONDA_IQE"; for p in "${PE[@]}"; do ape hf CohereLabs/tiny-aya-fire "$p"; done
  activate "$CONDA_VLLM"; export VLLM_TP="${VLLM_TP:-1}"
  for m in sarvamai/sarvam-m; do for p in "${PE[@]}"; do ape vllm "$m" "$p"; done; done
  # dedicated Indic-MT specialist (from-scratch retranslate) — the NMT point on the system spectrum
  for p in "${PE[@]}"; do mt vllm sarvamai/sarvam-translate "$p"; done
  echo "######## PHASE 5/6: CometKiwi over post-edits + MT-retranslate ########"; activate "$CONDA_IQE"
  python scripts/run_comet_metrics.py --ape "$IQE_OUT/ape_open" --out "$IQE_OUT/ape_metrics" --gpus 1 || echo "  !! ape-comet failed"
fi

if [ "$RUN_COLLECT" = 1 ]; then
  echo "######## PHASE 6/6: collect + analyse ########"; activate "$CONDA_IQE"
  python scripts/analyse_challenge.py --root "$IQE_OUT" --out "$IQE_OUT/analysis" || true
  ls -la "$IQE_OUT/analysis" 2>/dev/null
fi
echo "######## ALL PHASES DONE $(date) ########"
