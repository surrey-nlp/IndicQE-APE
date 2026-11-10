#!/usr/bin/env python3
"""
generate_enml_gpt_qe.py — GPT-5.5 Quality-Estimation on the en-ml challenge test split.

PROMPT CONSISTENCY BY CONSTRUCTION. This does not re-implement the QE prompt; it drives the
*exact* package pipeline the open LLMs used (aya-32B, sarvam-m, Llama-3.2-3B, BharatGPT-3B,
tiny-aya). Concretely it reuses:
  * indicqe_ape.prompts.render_qe        — the canonical GEMBA-DA + GEMBA-SQM templates
  * indicqe_ape.prompts.parse_qe_score   — the same 0-100 score parser
  * indicqe_ape.tasks.qe.run_qe          — the same item set (challenge/test/enml), ids,
                                            resumability, and output schema
  * indicqe_ape.models.OpenAIProvider    — the same GPT call (reasoning family: no temp/top_p,
                                            max_completion_tokens, reasoning_effort)
so the only thing that changes versus the other rows of tab:qe is the model. Output is written as
`gpt-5.5.s0.enml.jsonl` with records {id, pair, task, route, is_curated, gold_da_mean,
pred:{gemba-da, gemba-sqm}, raw} — identical to the other en-ml QE files, so it folds straight in.

Settings pinned to the 8-pair GPT-5.5 runs:
  shots = 0,  seed = 0,  templates = (gemba-da, gemba-sqm),  max_tokens = 512,
  reasoning_effort = "none"   (the 8-pair GPT column was produced at the provider default; keep it
                               so en-ml is comparable — override with --effort only if you re-run
                               the whole GPT column at the new effort).

Usage
-----
  export OPENAI_API_KEY=sk-...
  python generate_enml_gpt_qe.py --smoke      # 3 items: prints the exact prompt + raw + parsed score
  python generate_enml_gpt_qe.py              # full 1,436 -> gpt-5.5.s0.enml.jsonl (resumable)
  python generate_enml_gpt_qe.py --model gpt-5.5   # override if OpenAI's exact id differs
  python generate_enml_gpt_qe.py --data /path/to/snapshot   # use a local parquet snapshot instead of the Hub

The dataset is read from the public Hub repo (surrey-nlp/IndicQE-APE) unless --data is given.
Both templates are scored in one pass (2 calls/item); the paper's tab:qe column is gemba-da.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the in-repo package importable no matter where this is run from.
REPO = Path(__file__).resolve().parents[1]          # .../IndicQE-paper
SRC = REPO / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def _die(msg: str) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT-5.5 en-ml QE (prompt-identical to the open-LLM runs)")
    ap.add_argument("--model", default="gpt-5.5",
                    help="OpenAI model id (default gpt-5.5; the OpenAIProvider treats gpt-5.x as reasoning)")
    ap.add_argument("--pair", default="enml")
    ap.add_argument("--subset", default="challenge")
    ap.add_argument("--split", default="test")
    ap.add_argument("--effort", default=os.environ.get("IQE_REASONING_EFFORT", "none"),
                    help="GPT-5.x reasoning effort; 'none' matches the 8-pair paper runs")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("IQE_QE_MAXTOK", "512")),
                    help="completion budget (512 matches the other QE runs)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "gpt-5.5.s0.enml.jsonl"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "cache"))
    ap.add_argument("--data", default=None,
                    help="local parquet snapshot dir (sets INDICQEAPE_DATA); default reads the Hub")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N items (debug)")
    ap.add_argument("--smoke", action="store_true",
                    help="score 3 items, print the exact prompt + raw response + parsed score, then exit")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        _die("OPENAI_API_KEY is not set in the environment.")

    # Provider reads these from the env (exactly as the package does).
    os.environ["IQE_REASONING_EFFORT"] = args.effort
    os.environ["IQE_QE_MAXTOK"] = str(args.max_tokens)
    # Read en-ml from the public Hub unless a local snapshot is given (a stale local snapshot may
    # predate en-ml, so we clear it by default — same reason we `unset INDICQEAPE_DATA` on the cluster).
    if args.data:
        os.environ["INDICQEAPE_DATA"] = args.data
    else:
        os.environ.pop("INDICQEAPE_DATA", None)

    try:
        from indicqe_ape.models.openai_provider import OpenAIProvider
        from indicqe_ape.tasks.qe import run_qe
    except Exception as e:  # pragma: no cover
        _die(f"could not import the indicqe_ape package from {SRC} ({e}). "
             f"Run this from inside the IndicQE-paper repo, or `pip install -e .` there.")

    prov = OpenAIProvider(model=args.model, seed=0)

    if args.smoke:
        from indicqe_ape.data import DataLoader
        from indicqe_ape.languages import parse_pair
        from indicqe_ape.models.base import GenerationConfig
        from indicqe_ape.prompts import parse_qe_score, render_qe

        cfg = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0)
        pair_obj = parse_pair(args.pair)
        items = list(DataLoader().iter_items(args.subset, split=args.split, pair=args.pair, limit=3, seed=0))
        if not items:
            _die("no items loaded — check HF access to surrey-nlp/IndicQE-APE (or pass --data).")
        print(f"# smoke: {args.model}  effort={args.effort}  max_tokens={args.max_tokens}  "
              f"{args.subset}/{args.split}/{args.pair}\n")
        for it in items:
            for tkey in ("gemba-da", "gemba-sqm"):
                sys_msg, usr_msg = render_qe(pair_obj, it.source, it.mt, template=tkey)
                raw = prov.complete(sys_msg, usr_msg, config=cfg)
                print("=" * 72)
                print(f"[{tkey}] id={it.id}  gold_da={it.da_mean}")
                print("SYSTEM:", sys_msg)
                print("USER  :", usr_msg)
                print("RAW   :", repr(raw), "-> parsed:", parse_qe_score(raw))
        print("\n# smoke OK — prompts above are the package's render_qe output verbatim.")
        return

    summary = run_qe(
        prov,
        pair=args.pair,
        subset=args.subset,
        split=args.split,
        shots=0,
        limit=args.limit,
        seed=0,
        out=args.out,
        cache_dir=args.cache_dir,
    )
    da = summary["by_template"]["gemba-da"]
    sqm = summary["by_template"]["gemba-sqm"]
    print(f"[done] {args.model} en-ml QE  n={da['n']}  parsed={da['n_parsed']}/{da['n']}  "
          f"failures={da['n_parse_failures']}")
    print(f"       gemba-da   P={da['pearson']:.3f}  rho={da['spearman']:.3f}  tau={da['kendall']:.3f}   (paper column)")
    print(f"       gemba-sqm  P={sqm['pearson']:.3f}  rho={sqm['spearman']:.3f}  tau={sqm['kendall']:.3f}")
    print(f"       out={args.out}")


if __name__ == "__main__":
    main()
