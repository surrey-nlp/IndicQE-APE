#!/usr/bin/env python3
"""Run a YAML manifest of experiments sequentially. Skips already-completed runs.

A manifest is a list of experiment specs. Each spec has the same shape as a
single CLI invocation:

```yaml
defaults:
  cache_dir: cache
  seed: 0
  limit: null
experiments:
  - name: qwen2.5-1.5b_qe_enhi_zs
    task: qe
    provider: hf
    model: qwen2.5-1.5b
    pair: enhi
    subset: qe_only
    shots: 0
    limit: 200
  - name: gpt4o_qe_enhi_zs
    task: qe
    provider: openai
    model: gpt-4o
    pair: enhi
    subset: qe_only
    shots: 0
    limit: 200
```

Output paths are auto-derived as ``results/{name}.jsonl`` unless ``out`` is set
explicitly. Failures in one experiment do not stop the manifest — they are
logged to ``results/_manifest_errors.jsonl`` and the next experiment runs.

Usage:
    python scripts/run_manifest.py configs/laptop.yaml
    python scripts/run_manifest.py configs/workstation.yaml --results /scratch/results
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

from indicqe_ape.models import get_provider
from indicqe_ape.tasks import run_ape, run_mqm, run_qe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path, help="YAML manifest file")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--dry-run", action="store_true", help="print what would run, do nothing")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    defaults = manifest.get("defaults", {}) or {}
    experiments = manifest.get("experiments", []) or []

    args.results.mkdir(parents=True, exist_ok=True)
    err_log = args.results / "_manifest_errors.jsonl"

    n_ok = n_fail = n_skip = 0
    for i, spec in enumerate(experiments, 1):
        full = {**defaults, **spec}
        name = full.get("name", f"exp_{i:03d}")
        out = Path(full.get("out") or args.results / f"{name}.jsonl")
        full["out"] = out
        full.setdefault("cache_dir", args.cache_dir)
        task = full["task"]

        print(f"[{i}/{len(experiments)}] {name} → {out}", flush=True)
        if args.dry_run:
            print(f"  spec: {full}", flush=True)
            continue

        try:
            provider = get_provider(full["provider"], model=full["model"])
            kwargs = {k: full[k] for k in ("pair", "subset", "shots", "limit", "seed", "out", "cache_dir") if k in full}
            if task == "qe":
                summary = run_qe(provider, **kwargs)
            elif task == "ape":
                summary = run_ape(provider, with_comet=full.get("with_comet", True), **kwargs)
            elif task == "mqm":
                summary = run_mqm(provider, **kwargs)
            else:
                raise ValueError(f"unknown task: {task!r}")
            print(f"   OK {summary}", flush=True)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            tb = traceback.format_exc()
            print(f"   FAIL {e!r}", flush=True)
            with err_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"name": name, "spec": _jsonable(full), "error": str(e), "traceback": tb}) + "\n")

    print(f"\nManifest done: {n_ok} ok, {n_fail} failed, {n_skip} skipped.", flush=True)
    return 0 if n_fail == 0 else 1


def _jsonable(d: dict) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


if __name__ == "__main__":
    raise SystemExit(main())
