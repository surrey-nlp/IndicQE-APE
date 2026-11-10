"""Single CLI entrypoint: `indicqe-ape <task> ...`.

Usage examples
--------------
indicqe-ape qe   --provider openai    --model gpt-4o                   --pair enhi --subset qe_only        --shots 0 --limit 200 --out results/gpt4o_enhi_zs.jsonl
indicqe-ape ape  --provider hf        --model meta-llama/Meta-Llama-3-70B-Instruct --pair enmr --subset ape_only --shots 4 --limit 500 --out results/llama3_enmr_4s.jsonl
indicqe-ape mqm  --provider anthropic --model claude-opus-4-6                                  --shots 0          --out results/claude_enhi_mqm.jsonl
indicqe-ape stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import DataLoader, SUBSETS
from .languages import all_pair_codes
from .models import get_provider, list_providers
from .tasks import run_ape, run_mqm, run_qe


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", required=True, choices=list_providers())
    p.add_argument("--model", required=True)
    p.add_argument("--pair", required=True, choices=all_pair_codes())
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="test", choices=("train", "validation", "test"),
                   help="which split of the config to evaluate (default: test)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-dir", default="cache", type=Path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indicqe-ape")
    sub = parser.add_subparsers(dest="task", required=True)

    # --subset takes a unified config (all/full/challenge/qe-da/ape/<pair>); legacy
    # subset names (qe_only/ape_only/...) are accepted and mapped in data.py.
    p_qe = sub.add_parser("qe", help="Run QE evaluation (DA correlation).")
    _add_common_args(p_qe)
    p_qe.add_argument("--subset", default="challenge",
                      help="config to evaluate (default: challenge)")
    p_qe.add_argument("--templates", default="gemba-da,gemba-sqm",
                      help="comma-separated QE prompt variants to score per item "
                           "(default: gemba-da,gemba-sqm). First is the primary/baseline.")
    p_qe.add_argument("--gold", default="da", choices=("da", "mqm"),
                      help="label to correlate against. 'da' everywhere except --subset mqm, "
                           "where 2,016 of 2,639 test rows carry MQM and no DA. Both labels are "
                           "written per instance regardless, so a run can be rescored either way.")

    p_ape = sub.add_parser("ape", help="Run APE evaluation (BLEU / chrF / COMET-Kiwi vs PE).")
    _add_common_args(p_ape)
    p_ape.add_argument("--subset", default="challenge",
                       help="config to evaluate; rows without a post-edit are skipped")
    p_ape.add_argument("--no-comet", action="store_true", help="Skip COMET-Kiwi (faster).")
    p_ape.add_argument("--mode", default="postedit", choices=("postedit", "retranslate"),
                       help="postedit = LLM corrects the MT; retranslate = dedicated-MT system "
                            "(e.g. sarvam-translate) translates from scratch. Both write hyp_pe.")

    p_mqm = sub.add_parser("mqm", help="Run MQM span generation on the en-hi MQM config.")
    _add_common_args(p_mqm)
    # was "mqm_en_hi", a pre-June schema name that no longer resolved, so this subcommand
    # raised on every invocation. The released config is "mqm".
    p_mqm.add_argument("--subset", default="mqm")

    sub.add_parser("stats", help="Print row counts per subset / pair.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.task == "stats":
        df = DataLoader().stats()
        print(df.to_string(index=False))
        return 0

    provider = get_provider(args.provider, model=args.model)

    if args.task == "qe":
        templates = tuple(t.strip() for t in args.templates.split(",") if t.strip())
        summary = run_qe(
            provider, pair=args.pair, subset=args.subset, split=args.split,
            shots=args.shots, limit=args.limit, seed=args.seed,
            out=args.out, cache_dir=args.cache_dir, templates=templates,
            gold=args.gold,
        )
    elif args.task == "ape":
        summary = run_ape(
            provider, pair=args.pair, subset=args.subset, split=args.split,
            shots=args.shots, limit=args.limit, seed=args.seed,
            out=args.out, cache_dir=args.cache_dir,
            with_comet=not args.no_comet, mode=args.mode,
        )
    elif args.task == "mqm":
        summary = run_mqm(
            provider, pair=args.pair, subset=args.subset, split=args.split,
            shots=args.shots, limit=args.limit, seed=args.seed,
            out=args.out, cache_dir=args.cache_dir,
        )
    else:  # pragma: no cover
        raise SystemExit(f"Unknown task: {args.task}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
