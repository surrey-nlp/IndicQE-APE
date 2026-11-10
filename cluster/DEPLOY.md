# Running on a Slurm cluster

The scripts in `cluster/` produced the paper's GPU results. Nothing site-specific is hard-coded:
`cluster/env.sh` (sourced by every job) reads its settings from environment variables, and every
`.sbatch` ships with its partition line disabled (`##SBATCH --partition=...`), so you pass `-p`
at submit time. Each script's header says how much GPU memory it needs.

## 1. Environment variables

| Variable | Meaning |
|---|---|
| `SCRATCH` | Compute-visible scratch root. **Required inside Slurm jobs.** Data snapshot, HF cache, outputs and caches live under it. |
| `IQE_REPO` | This repository's checkout on the cluster (normally `$SCRATCH/indicqe-ape`). The `r*` and `xcomet_*` jobs `cd` here. |
| `PART_ITER` / `PART_BIG` / `PART_SMOKE` | Your partitions for >=48 GB, 80 GB and >=24 GB GPUs. Empty by default. |
| `FLAKY_NODE` | Optional node list that `submit.sh` passes to `--exclude`. |
| `CONDA_IQE` / `CONDA_VLLM` / `CONDA_ALOPE` | Conda env names. Defaults: `indicqe`, `indicqe-vllm`, `alope-qe`. |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Needed only for the closed-model (API) runs. |
| `PY` | Interpreter for the workstation (non-Slurm) scripts. Default: `python3`. |
| `IQE_SHARED` | Optional shared results directory to copy outputs into. |

`IQE_ROOT`, `INDICQEAPE_DATA`, `IQE_OUT`, `IQE_CACHE`, `ALOPE_REPO` and `HF_HOME` are all derived
from `SCRATCH` and can be overridden. When `SCRATCH` is unset outside a Slurm job, `env.sh` falls
back to a repo-local `.local-scratch/`. That fallback is enough for the API-only scripts.

## 2. One-time setup (on the login node, which has network access)

```bash
export SCRATCH=/path/to/scratch
cd "$SCRATCH" && git clone <this-repo-url> indicqe-ape && cd indicqe-ape
export IQE_REPO="$PWD"
source cluster/env.sh
bash cluster/00_setup.sh             # conda env + pip install -e . + dataset snapshot + sanity check
bash cluster/01_prefetch_models.sh   # cache open-weight models + COMET metrics (HF login, gated licences)
bash cluster/rebuild_vllm.sh         # separate vLLM env, only needed for the 24-32B models
git clone https://github.com/surrey-nlp/ALOPE.git "$ALOPE_REPO"   # ALOPE stage only
```

- Compute jobs run with `HF_HUB_OFFLINE=1`, so every model has to be cached beforehand.
- Install vLLM in its **own** env (`CONDA_VLLM`). Installing it with `pip` replaces torch, which
  would break the `indicqe` env (torch 2.4.1+cu121, required by COMET).
- The MetricX jobs (`r4_*`, `r4b_*`, `r5_metricx_gap`) need a separate `metricx` env
  (transformers 4.44.2) and the MetricX repo at `$SCRATCH/metricx-repo`.
- The full-population jobs (`xcomet_both_modes`, `r2_*`, `r4b_*`) read the HF `ape` config parquet
  files from `$SCRATCH/indicqeape/ape_parquet/`. Stage them there on the login node.

## 3. Pre-flight

```bash
python3 cluster/validate_deploy.py                            # syntax, array sizes, pair coverage, decoding budget
cluster/submit.sh <script> -p <partition> --dry-run           # sbatch --test-only + env/data checks
cluster/submit.sh <script> -p <partition> [sbatch args...]    # submit after the same checks
sbatch -p "$PART_SMOKE" cluster/smoke_vllm.sbatch             # check that the vLLM env loads and generates
```

## 4. Jobs, in run order

| Stage | Script(s) | What | GPU |
|---|---|---|---|
| QE, closed (API) | `23_qe_rerun_closed.sh` (`STAGE=probe`, then `enml`, then `full4`); `run_gpt55_closed.sh` | GPT-5.5 QE (+ APE), 512-token budget | none |
| QE, open (prompted) | `24_qe_rerun_tiered.sbatch` (`IQE_TIER=small\|large\|xl`) | `models_rerun.txt` x 9 pairs x {0,4}-shot | 11 / 48 / 80 GB by tier |
| QE, trained metrics | `10_qe_metrics.sbatch` | CometKiwi-DA, CometKiwi-XL, XCOMET-XL on the challenge test (+ APE pass) | 80 GB |
| Full-population metrics | `xcomet_both_modes.sbatch` (R1), `r2_cometkiwi_full.sbatch` (R2), `r4b_metricx_full.sbatch` (R4b) | 73,829-row `ape` config | 48 GB |
| en-gu / en-te gap | `r5_build_qeda_gap.py` (local), then `r5_metricx_gap.sbatch` + `r5_comet_gap.sh` (workstation) | reference-free only | 48 GB / 1 GPU |
| MQM | `50_mqm_zeroshot.sbatch` | COMET metrics vs en-hi MQM gold | 48 GB |
| Severity | `r0_severity_gpu.sbatch` (R0) + `r4_metricx_severity.sbatch` (R4), then `r0_severity_report.py` (CPU) | injected severity set | 48 / 80 GB |
| ALOPE | `30_alope_perpair.sbatch`, `31_alope_multilingual.sbatch` | 4 strategies: per pair (20 tasks), pooled (4 tasks) | >=24 GB |
| APE | `42_ape_regen.sbatch` (`IQE_APE_TIER=core\|xen`), then `ape_comet_local.sh` | tiny-aya / sarvam-m / sarvam-translate post-edits, then CometKiwi | 80 GB / 1 GPU |
| Collect | `90_collect.sh` | aggregate summaries + challenge diagnostics | none |

```bash
IQE_TIER=small bash cluster/24_qe_rerun_tiered.sbatch --dry-run     # prints the task count and --array
sbatch -p "$PART_SMOKE" --array=0-53 --export=ALL,IQE_TIER=small,IQE_PROVIDER=hf   cluster/24_qe_rerun_tiered.sbatch
sbatch -p "$PART_ITER"  --array=0-17 --export=ALL,IQE_TIER=large,IQE_PROVIDER=vllm cluster/24_qe_rerun_tiered.sbatch
sbatch -p "$PART_BIG"   --array=0-8  --export=ALL,IQE_APE_TIER=core cluster/42_ape_regen.sbatch
sbatch -p "$PART_ITER"  cluster/xcomet_both_modes.sbatch
STAGE=probe bash cluster/23_qe_rerun_closed.sh                       # API only; login node or laptop
```

Other scripts:

- **Workstation (no Slurm):** `60_a3_enml_fewshot.sh`, `r5_comet_gap.sh` and `ape_comet_local.sh`
  run on a local GPU machine. Set `PY` and `CUDA_VISIBLE_DEVICES` (or `GPU`) as needed.
- **Earlier launchers, kept for provenance:** `20_qe_prompt_open.sbatch`, `22_qe_rerun_open.sbatch`
  (both read `models_open.txt` and are superseded by `24_*`), `21_qe_prompt_closed.sh`,
  `40_ape_open.sbatch`, `41_ape_closed.sh`, `fewshot_3b.sbatch`, and `run_all_a100.sbatch`
  (a single-job orchestrator).

## Notes

- Every prompted-QE run scores both GEMBA templates (`gemba-da`, `gemba-sqm`) per item. Leave
  `IQE_QE_MAXTOK` unset so the 512-token default applies. A budget of 8 tokens truncates the outputs.
- The vLLM launchers set `IQE_NO_THINK=1` for sarvam-m (a hybrid reasoning model). Set
  `IQE_THINK=1` to opt back in.
- Jobs are resumable: the CLI skips ids already in `--out`, so re-submit after a pre-emption. If you
  change a generation setting, first delete the affected outputs and their cache.
- Jobs that `cd "$IQE_REPO"` write to `results/` inside the checkout. The rest write under
  `$IQE_OUT`. Scratch filesystems are often purged, so copy results off when a job finishes.
