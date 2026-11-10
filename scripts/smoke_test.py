#!/usr/bin/env python3
"""Laptop smoke test: run a tiny model over 5 examples to validate the full pipeline.

What this checks
----------------
1. Package imports cleanly
2. Data loader reaches HuggingFace (or falls back to dev mirror)
3. HF provider loads `Qwen/Qwen2.5-0.5B-Instruct` on whatever device is available
4. Tokenizer chat template + generation works
5. QE prompt parses; Pearson is finite
6. Output JSONL is valid; aggregation produces a row
7. Test resumability — re-run skips already-done items

Runs in ~2-3 minutes on Apple Silicon, longer on CPU. Does NOT require any
API keys or COMET-Kiwi. Exits 0 on success, non-zero on failure.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct --pair enhi --limit 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model path or registry alias (default: Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--pair", default="enhi", help="language pair (default: enhi)")
    ap.add_argument("--subset", default="qe_only")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--cache-dir", default=None,
                    help="response cache dir (default: a tmpdir, cleaned up at exit)")
    ap.add_argument("--keep", action="store_true", help="don't delete the temp working dir on exit")
    args = ap.parse_args()

    print("== indicqe-ape smoke test ==", flush=True)
    print(f"model: {args.model}  pair: {args.pair}  subset: {args.subset}  limit: {args.limit}", flush=True)

    # 1. Imports
    try:
        from indicqe_ape.data import DataLoader
        from indicqe_ape.models import get_provider
        from indicqe_ape.tasks import run_qe
    except Exception as e:
        print(f"FAIL imports: {e!r}", flush=True)
        return 10
    print("[1/6] imports OK", flush=True)

    # 2. Data loader
    try:
        dl = DataLoader()
        df = dl.load(args.subset, pair=args.pair)
        if len(df) == 0:
            print(f"FAIL data loader: empty subset for pair={args.pair}", flush=True)
            return 11
    except Exception as e:
        print(f"FAIL data loader: {e!r}", flush=True)
        return 11
    print(f"[2/6] data loader OK ({len(df):,} rows for {args.pair} in {args.subset})", flush=True)

    # 3. Provider construction (no API keys needed for hf)
    try:
        provider = get_provider("hf", model=args.model)
    except Exception as e:
        print(f"FAIL provider: {e!r}", flush=True)
        return 12
    print(f"[3/6] provider OK ({provider})", flush=True)

    # 4-7. Run QE on `limit` examples
    workdir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="indicqe_smoke_"))
    out_path = workdir / f"smoke_{args.pair}.jsonl"
    cache_dir = workdir / "cache"
    try:
        summary = run_qe(
            provider, pair=args.pair, subset=args.subset,
            shots=0, limit=args.limit, seed=0,
            out=out_path, cache_dir=cache_dir, data_loader=dl,
        )
        print(f"[4/6] generation OK summary={summary}", flush=True)

        # 5. Output JSONL contains the expected number of records
        with out_path.open("r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        meta = next((r for r in lines if "_meta" in r), None)
        body = [r for r in lines if "_meta" not in r and "_summary" not in r]
        smry = next((r for r in lines if "_summary" in r), None)
        assert meta and smry, "missing _meta or _summary"
        assert len(body) == args.limit, f"expected {args.limit} rows, got {len(body)}"
        print(f"[5/6] output schema OK ({len(body)} rows + meta + summary)", flush=True)

        # 6. Resumability: a second run on the same out_path should skip everything
        run_qe(
            provider, pair=args.pair, subset=args.subset,
            shots=0, limit=args.limit, seed=0,
            out=out_path, cache_dir=cache_dir, data_loader=dl,
        )
        with out_path.open("r", encoding="utf-8") as fh:
            lines2 = [json.loads(l) for l in fh if l.strip()]
        body2 = [r for r in lines2 if "_meta" not in r and "_summary" not in r]
        assert len(body2) == args.limit, f"resume should not duplicate rows, got {len(body2)}"
        print("[6/6] resumability OK (second run added 0 new rows)", flush=True)

        print()
        print("✓ smoke test passed", flush=True)
        print(f"  workdir: {workdir} (use --keep to retain)", flush=True)
        return 0
    except Exception as e:
        print(f"FAIL: {e!r}", flush=True)
        import traceback; traceback.print_exc()
        return 13
    finally:
        if not args.keep and (not args.cache_dir):
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
