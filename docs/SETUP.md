# Setting up and running IndicQE-APE

Machine setup, provider keys, batch manifests and the model shortlist. The project overview,
results and citation are in the top-level README.

## Setup — for a brand new machine

These commands take you from a fresh checkout to a running smoke test in ~5 minutes (laptop) or ~10 minutes (workstation, GPU model download dominates).

### 0. Prerequisites

- **Python ≥ 3.10** (`python3 --version`)
- **git**
- **macOS users:** install Xcode CLT (`xcode-select --install`) once.
- **Workstation users:** CUDA 12.x toolkit and NVIDIA drivers for `bitsandbytes` 4-bit support.

### 1. Clone

```bash
git clone https://github.com/surrey-nlp/IndicQE-APE.git
cd IndicQE-APE
```

### 2. Create a virtual environment

```bash
# Option A — venv (built-in)
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# Option B — conda
conda create -n indicqe-ape python=3.11 -y
conda activate indicqe-ape
```

### 3. Install

Pick the install profile that matches the machine.

```bash
# Laptop (Mac / no CUDA): API providers + HF local — skip 4-bit
pip install -e ".[openai,anthropic,google,cohere,hf]"

# Workstation (Linux + CUDA): everything including bitsandbytes 4-bit
pip install -e ".[all]"

# Pinned-version reproducible install (either machine)
pip install -r requirements.txt
pip install -e . --no-deps
```

### 4. Configure API keys and HF token

```bash
cp .env.example .env
$EDITOR .env                  # fill in any keys you have
set -a; source .env; set +a   # export everything for this shell

# Or, just for this session:
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export COHERE_API_KEY="..."
export HF_TOKEN="hf_..."      # needed for gated models AND COMET-Kiwi
```

The `HF_TOKEN` matters for two reasons:

1. Gated open-weight models (Llama, Gemma) require it.
2. `Unbabel/wmt22-cometkiwi-da` (COMET-Kiwi) is gated — accept the licence on its [HF page](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) once, then the token grants the eval scripts access. Without it, APE will run but report `comet_kiwi: null`.

### 5. Verify the install

```bash
pytest tests/                                      # 106 tests, a few seconds
indicqe-ape --help                                 # CLI loads
indicqe-ape stats                                  # confirm dataset reachable on HF
```

### 6. Laptop smoke test (no API keys needed)

Runs a tiny 0.5B model against 5 examples to validate the whole pipeline (data loader, HF provider, prompt rendering, parsing, metrics, JSONL output, resumability). On Apple Silicon it lands on MPS automatically.

```bash
python scripts/smoke_test.py
# Or pick a different model / pair / size:
python scripts/smoke_test.py --model qwen2.5-1.5b --pair enhi --limit 5
```

If this prints `✓ smoke test passed`, you're set.

## Running experiments

### One-off

```bash
# Zero-shot QE on en-hi with GPT-5.5, 200 examples
indicqe-ape qe --provider openai --model gpt-5.5 \
  --pair enhi --subset qe-da \
  --shots 0 --limit 200 \
  --out results/gpt55_enhi_zs.jsonl

# 4-shot APE on en-mr with Aya Expanse 32B through the Cohere API
indicqe-ape ape --provider cohere --model c4ai-aya-expanse-32b \
  --pair enmr --subset ape \
  --shots 4 --limit 500 \
  --out results/aya32b_enmr_4s.jsonl

# Zero-shot MQM on the en-hi MQM layer (the `mqm` config)
indicqe-ape mqm --provider anthropic --model <claude-model-id> \
  --pair enhi --subset mqm --shots 0 \
  --out results/claude_enhi_mqm.jsonl
```

The output JSONL is **resumable**: if a run is interrupted, re-issuing the same command continues from where it stopped. The response cache survives interruption.

### Batch (recommended for paper-table production)

```bash
# Laptop manifest — small models, capped at 100 examples per pair
python scripts/run_manifest.py configs/laptop.yaml

# Workstation manifest — mid-size open-weight + closed-weight comparators
python scripts/run_manifest.py configs/workstation.yaml

# Dry-run to preview what would be executed
python scripts/run_manifest.py configs/workstation.yaml --dry-run
```

Each experiment writes a JSONL to `results/{name}.jsonl`. Failed experiments do not stop the manifest; they're logged to `results/_manifest_errors.jsonl`.

### Aggregating results across machines

After any run (laptop or workstation), regenerate the canonical tables:

```bash
python scripts/aggregate_results.py
# → writes results/summary.tsv, summary.jsonl, summary.md
```

The aggregator dedupes by a stable `experiment_id` derived from `(task, subset, pair, shots, provider, model, seed, fewshot_signature)`, so re-running the same config is idempotent and runs from several machines merge cleanly: copy the per-experiment JSONLs into one `results/` directory and re-run the aggregator. `results/` is gitignored apart from the outputs and aggregate CSVs that back the paper.

## Adding a new model provider

Subclass `LLMProvider` in `src/indicqe_ape/models/`:

```python
from indicqe_ape.models import register
from indicqe_ape.models.base import GenerationConfig, LLMProvider

@register
class MyProvider(LLMProvider):
    name = "myprovider"

    def __init__(self, model: str, **kw):
        super().__init__(model=model, **kw)
        # validate keys, init client...

    def complete(self, system: str, user: str, *, config: GenerationConfig | None = None) -> str:
        ...  # call your API / model
```

Add the module name to the lazy-import tuple in `src/indicqe_ape/models/__init__.py` and the CLI picks it up automatically — `--provider myprovider` becomes a valid choice.

## Metrics

- **QE** — Pearson, Spearman, Kendall correlations of model-predicted scores against gold DA mean (and against MQM scores, where available).
- **APE** — sacreBLEU, chrF++, TER, and COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`) against human post-edits.
- **MQM** — segment-level severity-weighted score (minor +1, major +5, critical +10), and span-level macro F1 with Jaccard ≥ 0.5 token overlap and severity match. Both-empty segments are skipped from the macro to prevent inflation by no-error baselines.

All metrics live in `src/indicqe_ape/metrics.py` and are unit-tested. Run `pytest tests/` before any results table.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set in the environment` | `export OPENAI_API_KEY=...` or load `.env` |
| HF model load hangs on Mac | check `device` resolved correctly — set `dtype=float16` if `bfloat16` errors on older MPS |
| `bitsandbytes` import error on Mac | drop `load_in_4bit`; 4-bit needs CUDA |
| COMET-Kiwi reports `null` | accept the licence on the model's HF page and re-export `HF_TOKEN` |
| Resume created duplicate rows | shouldn't happen — `_read_done_ids` is `id`-keyed; report a bug if seen |
| Cache hit on a different config | each `pair`/`shots`/`temperature`/`top_p` change re-keys the hash; if you suspect a stale hit, delete the matching `cache/{task}__{provider}__{model}.jsonl` |

## Reproducibility

- All API responses are cached in `cache/{task}__{provider}__{model}.jsonl` keyed by `(provider, model, prompt, full GenerationConfig, task, shots, fewshot_signature)`.
- Each result JSONL begins with a `_meta` line capturing the full experiment spec; aggregation derives a stable `experiment_id` from this.
- `seed=` is honoured by `pandas.sample`, the OpenAI provider (passed to the API), and HF greedy decoding (deterministic at `temperature=0`). Anthropic and Google APIs are non-deterministic at temperature 0; document this when reporting numbers.

## Data

The benchmark is hosted on the Hugging Face Hub as [`surrey-nlp/IndicQE-APE`](https://huggingface.co/datasets/surrey-nlp/IndicQE-APE) and cached locally on first use; `scripts/download_data.py` makes an offline snapshot. Configurations: `all`, `full`, `challenge`, `qe-da`, `ape`, `word-qe`, `explainable-qe`, `mqm`, plus one per pair (the dataset card lists their sizes). The pre-release subset names `qe_only`, `ape_only`, `ape_qe_combined` and `mqm_en_hi` are still accepted as aliases. Nine directional pairs: `enhi`, `enmr`, `enta`, `engu`, `ente`, `enml`, and `eten` *(Estonian–English, a non-Indic control)*, `neen`, `sien`.

## Recommended Open-Weight Models for Indic QE/APE

A curated registry of open-weight, instruction-tuned causal LMs known to be runnable through `HFProvider` and useful for English↔Indic QE / APE. Every entry below has been verified to exist on HuggingFace at the time of writing. Definitions live in `src/indicqe_ape/models/indic_models.py`; pass the alias directly to `--model` or use the helper:

```python
from indicqe_ape.models.indic_models import resolve, recommend_for_laptop
resolve("qwen2.5-7b")          # -> "Qwen/Qwen2.5-7B-Instruct"
recommend_for_laptop()         # -> "qwen2.5-1.5b"
```

### Laptop (≤2B)

| alias | HF path | size | quant-friendly? | licence | best for |
| --- | --- | --- | --- | --- | --- |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | n/a (already tiny) | apache-2.0 | sanity checks, CI, CPU-only smoke tests |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | yes (4-bit ~1 GB) | apache-2.0 | **recommended laptop baseline** — Apple Silicon MPS, multilingual |
| `llama-3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | yes | llama3.2 | gated; en-hi only on Indic side |
| `sarvam-1` | `sarvamai/sarvam-1` | 2.5B | yes | other (non-commercial) | broadest 11-language Indic coverage at this size |

### Workstation single-GPU (3-14B, runs bf16 on a single A5000 24GB)

| alias | HF path | size | quant-friendly? | licence | best for |
| --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | 3.1B | yes | qwen-research (NC) | small mid-tier multilingual |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | yes | llama3.2 | en-hi only |
| `airavata` | `ai4bharat/Airavata` | 6.9B | yes | llama2 | AI4Bharat Hindi specialist (en-hi) |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | yes | apache-2.0 | strong open multilingual workhorse |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8.0B | yes | llama3.1 | gated; Hindi-aware general baseline |
| `llama-3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | 8.0B | yes | llama3 | English-only control |
| `aya-23-8b` | `CohereLabs/aya-23-8B` | 8.0B | yes | cc-by-nc-4.0 | Cohere Aya, 23-language coverage incl. Hindi |
| `gemma-2-9b` | `google/gemma-2-9b-it` | 9.2B | yes | gemma | gated; broad multilingual |
| `navarasa-7b` | `Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0` | 8.5B | yes | gemma | broadest 15-Indic-language fine-tune in this tier |

### Workstation multi-GPU / quantized (≥14B; needs 2× A5000 24GB or 4-bit)

| alias | HF path | size | quant-friendly? | licence | best for |
| --- | --- | --- | --- | --- | --- |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` | 14B | yes | apache-2.0 | bf16 across 2× 24 GB; strongest Apache mid-large |
| `sarvam-m-24b` | `sarvamai/sarvam-m` | 24B | yes (4-bit fits 1× 24 GB) | apache-2.0 | **strongest Indic-tuned chat model that fits a single A5000 in 4-bit** |
| `gemma-2-27b` | `google/gemma-2-27b-it` | 27B | yes | gemma | gated; 4-bit single-GPU or bf16 across both |
| `qwen2.5-32b` | `Qwen/Qwen2.5-32B-Instruct` | 32B | yes | apache-2.0 | 4-bit single-GPU; bf16 needs both |
| `aya-23-35b` | `CohereLabs/aya-23-35B` | 35B | yes | cc-by-nc-4.0 | non-commercial; 4-bit on 2× 24 GB |
| `llama-3-70b` | `meta-llama/Meta-Llama-3-70B-Instruct` | 70B | required (4-bit) | llama3 | English-tuned upper-bound control |
| `llama-3.1-70b` | `meta-llama/Llama-3.1-70B-Instruct` | 70B | required (4-bit) | llama3.1 | gated; Hindi-aware upper-bound |
| `nemotron-70b` | `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | 70B | required (4-bit) | llama3.1 | NVIDIA RLHF-tuned, strong instruction-following |
| `qwen2.5-72b` | `Qwen/Qwen2.5-72B-Instruct` | 72B | required (4-bit) | qwen (other) | 4-bit just fits 2× 24 GB; best open multilingual ceiling |

### Recommendation for a two-GPU (2× 24 GB) workstation

Start with **`sarvam-m-24b`** (`sarvamai/sarvam-m`). It is the strongest open Indic-tuned chat model that fits comfortably on a single A5000 in 4-bit (Mistral-Small-3.1 24B base, Apache-2.0, fine-tuned on en + 10 Indian languages: bn, hi, kn, gu, mr, ml, or, pa, ta, te). For an Apache-2.0 baseline alongside it, run **`qwen2.5-14b`** in bf16 across both GPUs — together they give a clean Indic-specialist vs. general-multilingual contrast for the QE/APE tables. Push to **`qwen2.5-72b`** or **`llama-3.1-70b`** in 4-bit only when you need the upper-bound numbers.

## Acknowledgements

**IndicQE-APE** was made possible by its data sponsors and collaborating institutions:

- **People-Centred AI Institute, University of Surrey** — https://www.surrey.ac.uk/artificial-intelligence
- **Computation for Indic Language Technology (CFILT), IIT Bombay** — https://www.cfilt.iitb.ac.in
- **European Association for Machine Translation (EAMT)** — https://eamt.org

We further acknowledge the upstream data providers whose released corpora this benchmark
consolidates: the WMT Quality Estimation shared tasks (2020, 2022, 2023, 2024) and the WMT
Automatic Post-Editing shared tasks, the MLQE-PE dataset, and the annotation teams behind the
English–Hindi MQM and English–Malayalam QE data. We acknowledge the efforts of Rajen Chatterjee
(Apple Inc., US) towards creating the APE data, and thank the data annotation agencies Zibanka
Media and Techliebe for their help with human data annotation over the years.
