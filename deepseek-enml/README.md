# DeepSeek-V4-Pro en-ml generator (synthetic descriptions + APE)

One script, `generate_enml_deepseek.py`, that completes the `surrey-nlp/IndicQE-APE` **en-ml** data
with DeepSeek-V4-Pro: it **reuses what already exists and generates only what's genuinely missing.**

## Background — the state before this script ran (18 Jul 2026)

en-ml has 10,000 rows. The ALOPE-RL source held descriptions for 5,000 of them in **two variants
(DeepSeek + Gemini)**; only 2,960 DeepSeek ones had been published. We recovered the rest and pushed
them to HF, so the dataset now has:

- `synthetic_description` (+ `error_categories`) — DeepSeek, for **5,000** rows
- `synthetic_description_gemini` (+ `error_categories_gemini`) — Gemini, for the same **5,000** rows
- the other **5,000** rows (tech/sports/finance domain-expansion, not in ALOPE) — **no description**

The script has since been run: in the released dataset all **10,000** en-ml rows carry an error
explanation (Table 1 of the paper).

Every row has a **human `post_edit`** (the APE reference — the model never sees it).

## What the script does

| Path                          | Rows                         | Given to the model                                                                               | Model produces                                                                   |
| ----------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **A** (no description)  | ~5,000 (tech/sports/finance) | source, MT, domain,`tqr`                                                                       | `synthetic_description` + `error_categories` + **`post_edit` (APE)** |
| **B** (has description) | ~5,000 (recovered ALOPE)     | source, MT, domain,**existing** `synthetic_description` + `error_categories` + `tqr` | **`post_edit` (APE)** only                                               |

So Path A fills only the genuine 5,000 gaps; Path B reuses the recovered descriptions. It branches
automatically on `has_synth_expl`, so **no manual filtering is needed** — just run it.

**We generate a DeepSeek description only for the 5,000 Path-A rows — no Gemini variant for those**
(Gemini exists only for the original ALOPE 5,000). A DeepSeek APE hypothesis is produced for all 10K.

## Setup

```bash
pip install openai pandas huggingface_hub pyarrow
export DEEPSEEK_API_KEY=sk-...        # https://platform.deepseek.com/api_keys
```

## Run

```bash
python generate_enml_deepseek.py --smoke        # 3 rows/path, prints prompts + outputs
python generate_enml_deepseek.py --workers 32    # full 10,000 -> enml_deepseek.jsonl (resumable)
python generate_enml_deepseek.py --only A        # just the ~5,000 that need a new description
python generate_enml_deepseek.py --split test    # only the 1,436 test rows
```

Pulls en-ml straight from the Hub (public), so no data prep. `--input file.parquet|.jsonl` for a
local copy.

## API settings (baked in, per api-docs.deepseek.com)

- model `deepseek-v4-pro`, base_url `https://api.deepseek.com` (OpenAI-compatible)
- **thinking on, `reasoning_effort="high"`** (thinking mode ignores temperature/top_p → not set)
- **`max_tokens=65536`** (model max 384,000; raise via `--max-tokens` if you see `finish_reason="length"`)
- strict JSON via `response_format={"type":"json_object"}`, with retry + robust parse
- retries with backoff on 429/500/503; aborts a row on 400/401/402/422
- 16 workers by default (account concurrency limit 500; `--workers 32` is comfortable)

## Output — `enml_deepseek.jsonl` (one line per row)

```json
{
  "uid": "...", "_split": "train", "path": "A_gen_desc_and_ape",
  "used_existing_synth_desc": false,
  "synthetic_description": "…(generated for A; copied through for B)…",
  "error_categories": "Mistranslation, Fluency Error",
  "deepseek_post_edit": "…Malayalam APE hypothesis…",
  "explanation_model": "deepseek-v4-pro",
  "finish_reason": "stop", "usage": {...}, "error": null
}
```

Downstream: `deepseek_post_edit` is the DeepSeek APE hypothesis (`hyp_pe`) — MATRA-score it against
the human `post_edit` (`ref_pe`), like the other APE systems. The Path-A `synthetic_description` /
`error_categories` fill the 5,000 domain-expansion gaps. Join back on `uid`.

## Design choices (flag if you'd prefer otherwise)

- **The human `post_edit` is never shown to the model** — it's the held-out reference, so DeepSeek is
  a genuine APE system.
- **`tqr` (human quality remarks) is given in both paths** where present. In Path A the model still
  writes its own description; `tqr` is offered as remarks to consider. `--path-a-no-tqr` if you want
  Path A descriptions to be purely model-derived (flag not yet wired — ask and I'll add it).
- Reasoning traces aren't saved by default (`--save-reasoning` to keep them).
- Rough cost at high effort for the ~5,000 Path-A + 10,000 APE ≈ **$10–20**.
