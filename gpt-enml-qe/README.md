# GPT-5.5 en-ml QE — prompt-identical to the open-LLM runs

`generate_enml_gpt_qe.py` scores the **en-ml challenge test split (1,436 rows)** with GPT-5.5, the
en-ml cell of the paper's QE table (Table 4); `generate_enml_gpt_ape.py` does the same for APE. Both
need an `OPENAI_API_KEY` in the environment.

## Why it's consistent

It does **not** re-write the prompt. It drives the package's own pipeline, so the only variable vs
the other rows of `tab:qe` is the model:

- prompt   → `indicqe_ape.prompts.render_qe` (canonical **GEMBA-DA + GEMBA-SQM**)
- parser   → `indicqe_ape.prompts.parse_qe_score` (same 0-100 extractor)
- items    → `run_qe` on `challenge/test/enml` — same ids, same order, same schema
- GPT call → `indicqe_ape.models.OpenAIProvider` (gpt-5.x = reasoning: no temp/top_p,
  `max_completion_tokens`, `reasoning_effort`)

Pinned to the 8-pair GPT-5.5 runs: `shots=0`, `seed=0`, `max_tokens=512`, `reasoning_effort="none"`.
Output `gpt-5.5.s0.enml.jsonl` matches the other en-ml QE files, so it folds straight into the table.

## Run

```bash
export OPENAI_API_KEY=sk-...
python generate_enml_gpt_qe.py --smoke     # 3 items: prints the exact prompt + raw + parsed score
python generate_enml_gpt_qe.py             # full 1,436 (resumable) -> gpt-5.5.s0.enml.jsonl
```

Reads the dataset from the public Hub (`surrey-nlp/IndicQE-APE`); pass `--data <snapshot>` to use a
local parquet copy. Both templates are scored in one pass (2 calls/item ≈ 2,872 calls); the paper
column is **gemba-da**. On finish it prints P/ρ/τ for both templates.

## Notes

- `--model gpt-5.5` is the default label; override if OpenAI's exact id differs (you'll get a 404 to
  tell you). The reasoning path triggers on any `gpt-5*`/`o1/o3/o4` id.
- Keep `--effort none` unless you re-run the **whole** GPT column at a new effort — otherwise en-ml
  wouldn't be comparable to the other 8 pairs.
- Resumable: re-running continues from the last written id. Expected first line is a `_meta` record,
  then one line per item, then a `_summary` line.
