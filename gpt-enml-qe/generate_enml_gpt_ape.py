#!/usr/bin/env python3
"""
generate_enml_gpt_ape.py — GPT-5.5 automatic post-editing (APE) on the en-ml challenge test split.

Companion to generate_enml_gpt_qe.py. Same prompt-consistency guarantee: it drives the package's
own APE pipeline, so a GPT-5.5 row drops straight into Table~\\ref{tab:ape} next to sarvam-m,
sarvam-translate and tiny-aya. Reuses:
  * indicqe_ape.prompts.render_ape / parse_ape_response  — the APE prompt + output cleaner
  * indicqe_ape.tasks.ape.run_ape (mode='postedit')      — same challenge/test/enml items, ids, schema
  * indicqe_ape.models.OpenAIProvider                    — the GPT call (reasoning family)

Output `gpt-5.5.postedit.enml.jsonl` carries {id, pair, task, route, source, mt, ref_pe, hyp_pe,
raw_response} — identical to the other APE files, so `results/enml-runs/enml_ape_score.py` scores it
(MATRA / MATRA++ / char-TER / chrF++) with no changes.

Note on effort: run_ape caps the completion budget at 512 tokens, which is ample for a post-edit at
`reasoning_effort=none` (the default here). Raising effort would need more tokens, so leave it at none
unless you also raise the cap in tasks/ape.py.

Usage
-----
  export OPENAI_API_KEY=sk-...
  python generate_enml_gpt_ape.py --smoke     # 3 items: prints prompt + raw + cleaned post-edit
  python generate_enml_gpt_ape.py             # full 1,436 -> gpt-5.5.postedit.enml.jsonl (resumable)
Then MATRA-score it:
  cp gpt-5.5.postedit.enml.jsonl ../results/enml-runs/enml_bundle/ape/ && \
    python ../results/enml-runs/enml_ape_score.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT-5.5 en-ml APE (prompt-identical to the open APE runs)")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--pair", default="enml")
    ap.add_argument("--subset", default="challenge")
    ap.add_argument("--split", default="test")
    ap.add_argument("--effort", default=os.environ.get("IQE_REASONING_EFFORT", "none"),
                    help="GPT-5.x reasoning effort; keep 'none' (run_ape caps tokens at 512)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "gpt-5.5.postedit.enml.jsonl"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "cache"))
    ap.add_argument("--data", default=None, help="local parquet snapshot dir (default: Hub)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-comet", action="store_true", help="skip COMET-Kiwi in the summary (faster)")
    ap.add_argument("--smoke", action="store_true", help="post-edit 3 items, print prompt+raw+clean, exit")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        _die("OPENAI_API_KEY is not set in the environment.")

    os.environ["IQE_REASONING_EFFORT"] = args.effort
    if args.data:
        os.environ["INDICQEAPE_DATA"] = args.data
    else:
        os.environ.pop("INDICQEAPE_DATA", None)

    try:
        from indicqe_ape.models.openai_provider import OpenAIProvider
        from indicqe_ape.tasks.ape import run_ape
    except Exception as e:  # pragma: no cover
        _die(f"could not import indicqe_ape from {SRC} ({e}). Run inside the repo or `pip install -e .`.")

    prov = OpenAIProvider(model=args.model, seed=0)

    if args.smoke:
        from indicqe_ape.data import DataLoader
        from indicqe_ape.languages import parse_pair
        from indicqe_ape.models.base import GenerationConfig
        from indicqe_ape.prompts import parse_ape_response, render_ape

        cfg = GenerationConfig(max_tokens=512, temperature=0.0)
        pair_obj = parse_pair(args.pair)
        items = [it for it in DataLoader().iter_items(args.subset, split=args.split, pair=args.pair,
                                                      limit=8, seed=0) if it.pe][:3]
        if not items:
            _die("no PE-bearing en-ml items loaded — check Hub access (or pass --data).")
        print(f"# smoke: {args.model} APE  effort={args.effort}  {args.subset}/{args.split}/{args.pair}\n")
        for it in items:
            sys_msg, usr_msg = render_ape(pair_obj, it.source, it.mt)
            raw = prov.complete(sys_msg, usr_msg, config=cfg)
            print("=" * 72)
            print(f"id={it.id}")
            print("SYSTEM:", sys_msg)
            print("USER  :", usr_msg)
            print("RAW   :", repr(raw))
            print("CLEAN hyp_pe:", parse_ape_response(raw))
            print("ref_pe:", it.pe)
        print("\n# smoke OK — prompt above is the package's render_ape output verbatim.")
        return

    summary = run_ape(
        prov, pair=args.pair, subset=args.subset, split=args.split, shots=0,
        limit=args.limit, seed=0, out=args.out, cache_dir=args.cache_dir,
        with_comet=not args.no_comet, mode="postedit",
    )
    print(f"[done] {args.model} en-ml APE  ->  {args.out}")
    print(f"       summary: {summary}")
    print("       next: MATRA-score with results/enml-runs/enml_ape_score.py (see header).")


if __name__ == "__main__":
    main()
