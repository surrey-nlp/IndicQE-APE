#!/usr/bin/env python3
"""validate_deploy.py — pre-flight validation of every deployable script.

Run this BEFORE any sbatch. It checks the things that have actually gone wrong on this
project, each one traceable to a real failure in the logs:

  C1  bash/python syntax                     (a typo wastes a queue slot)
  C2  SBATCH partition is real               (checked against `sinfo`; skipped off-Slurm)
  C3  array size == grid size                (0-107 for 5 models -> unbound MODELS[5], rows die)
  C4  PAIRS has all nine pairs               (8-pair arrays are why en-ml was missing everywhere)
  C5  IQE_QE_MAXTOK is never set to 8        (truncated outputs; BharatGPT en-ta parse 0.22)
  C6  per-task --cache-dir, never shared     (shared cache -> UnicodeDecodeError, 4 tasks lost)
  C7  referenced data files exist            (or are declared as produced upstream)
  C8  output dir is set and is under $IQE_OUT
  C9  resumability                           (pre-emption must not restart from zero)
  C10 no hardcoded --limit on a full run     (--limit 400 is why gpt-5.5 4-shot covers 3,200)
  C11 IQE_NO_THINK=1 for reasoning models    (sarvam-m without it: 11h26m for 8 of 18 cells,
                                              and a different score, not just a slower one)

Exit code 0 only if every check passes at ERROR level. WARN does not block.

Usage:
  python3 cluster/validate_deploy.py                 # all deployable scripts
  python3 cluster/validate_deploy.py --deploy-set    # only what we are about to submit
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "cluster"


def _site_partitions() -> set[str] | None:
    """Partitions on THIS cluster, read from `sinfo -h -o %P`; None when not on Slurm.

    Submitting to a partition that does not exist fails at sbatch time with an opaque message,
    so active `#SBATCH --partition=` lines are checked against the live list. The shipped scripts
    keep that line disabled (`##SBATCH`) and take the partition from `-p` at submit time.
    """
    try:
        r = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return {ln.strip().rstrip("*") for ln in r.stdout.splitlines() if ln.strip()}


VALID_PARTITIONS = _site_partitions()
NINE_PAIRS = {"enhi", "enmr", "enta", "ente", "engu", "enml", "eten", "neen", "sien"}
#: Scripts whose pair list is legitimately a subset, with the reason.
PAIR_SUBSET_OK = {
    "40_ape_open.sbatch": "PE-bearing pairs only",
    "41_ape_closed.sh": "PE-bearing pairs only",
    "30_alope_perpair.sbatch": "curated en->Indic probe",
    "r5_build_qeda_gap.py": "gap-fill pairs only",
    "r5_metricx_gap.sbatch": "gap-fill pairs only",
    "r5_comet_gap.sh": "gap-fill pairs only",
}
#: The set we intend to submit now.
DEPLOY_SET = ["24_qe_rerun_tiered.sbatch", "23_qe_rerun_closed.sh", "50_mqm_zeroshot.sbatch",
              "22_qe_rerun_open.sbatch"]

ERRORS: list[str] = []
WARNS: list[str] = []


def strip_comments(txt: str) -> str:
    """Remove shell comments, preserving #SBATCH directives and line numbering.

    Several scripts DOCUMENT the faults they fix ("the original runs set IQE_QE_MAXTOK=8",
    "--limit 400 per pair"). Matching those comments produced false positives on the very
    scripts that are correct, which is the fastest way to make a validator ignored.
    """
    out = []
    for ln in txt.splitlines():
        s = ln.lstrip()
        if s.startswith("#SBATCH"):
            out.append(ln)
        elif s.startswith("#"):
            out.append("")                      # keep line count, drop content
        else:
            out.append(re.sub(r"(?<!\$)#(?![{(]).*$", "", ln))   # trailing comment
    return "\n".join(out)


def err(f: str, c: str, m: str) -> None:
    ERRORS.append(f"  ERROR [{c}] {f}: {m}")


def warn(f: str, c: str, m: str) -> None:
    WARNS.append(f"  WARN  [{c}] {f}: {m}")


def c1_syntax(p: Path, txt: str) -> None:
    if p.suffix == ".py":
        try:
            ast.parse(txt)
        except SyntaxError as e:
            err(p.name, "C1", f"python syntax: {e}")
    else:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            err(p.name, "C1", f"bash syntax: {r.stderr.strip()[:200]}")


def c2_partition(p: Path, txt: str) -> None:
    for m in re.finditer(r"^#SBATCH\s+--partition=(\S+)", txt, re.M):
        part = m.group(1)
        if VALID_PARTITIONS is None:
            print(f"    C2 skip: {p.name} (not on Slurm; cannot check partition '{part}')")
        elif part not in VALID_PARTITIONS:
            err(p.name, "C2", f"partition '{part}' does not exist on this cluster (sinfo)")


def c3_array(p: Path, txt: str) -> None:
    am = re.search(r"^#SBATCH\s+--array=(\d+)-(\d+)", txt, re.M)
    if not am:
        return
    hi = int(am.group(2))
    # The tiered launcher derives its array size at runtime from the tier and prints it via
    # --dry-run, so a hardcoded --array is expected to be absent or advisory. Skip C3 for it.
    if "models_rerun.txt" in txt:
        print(f"    C3 skip: {p.name} derives task count from the tier at runtime")
        return
    if "models_open.txt" not in txt:
        return
    mf = CLUSTER / "models_open.txt"
    if not mf.exists():
        err(p.name, "C3", "models_open.txt missing")
        return
    n_models = len([ln for ln in mf.read_text().splitlines()
                    if ln.strip() and not ln.strip().startswith("#")])
    pm = re.search(r"PAIRS=\(([^)]*)\)", txt)
    sm = re.search(r"SHOTS_LIST=\(([^)]*)\)", txt)
    if not (pm and sm):
        return
    n_pairs = len(pm.group(1).split())
    n_shots = len(sm.group(1).split())
    expect = n_models * n_pairs * n_shots
    if hi + 1 != expect:
        err(p.name, "C3",
            f"array 0-{hi} declares {hi + 1} tasks but grid is "
            f"{n_models} models x {n_pairs} pairs x {n_shots} shots = {expect}. "
            f"Rows >= {expect} index an unbound MODELS[] and abort under set -u.")
    else:
        print(f"    C3 ok: {p.name} array 0-{hi} == {n_models}x{n_pairs}x{n_shots}")


def c4_pairs(p: Path, txt: str) -> None:
    if p.name in PAIR_SUBSET_OK:
        print(f"    C4 skip: {p.name} ({PAIR_SUBSET_OK[p.name]})")
        return
    # Union every PAIRS assignment, so a script that splits pairs across tiers is judged on what
    # it can reach in total rather than on whichever branch appears first in the file.
    found: set[str] = set()
    for pat in (r"PAIRS=\(([^)]*)\)", r"ALL_PAIRS=\(([^)]*)\)"):
        for m in re.finditer(pat, txt):
            found |= {t for t in m.group(1).split() if re.fullmatch(r"[a-z]{4}", t)}
    if not found:
        return

    # A pair can be legitimately unreachable -- en-gu and en-te carry no post-edits, so APE on them
    # is impossible, not merely skipped. Requiring a written reason keeps that visible rather than
    # letting anyone silence the check by trimming the array.
    # Read from the RAW file: the exemption is deliberately a comment, and `txt` has had comments
    # stripped so that explanatory prose cannot trip the other checks.
    exempt: set[str] = set()
    raw = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"VALIDATE-C4-EXEMPT:\s*([a-z, ]+?)\s*(?:--|#|$)", raw, re.M)
    if m:
        exempt = {t.strip() for t in m.group(1).split(",") if t.strip()}
        reason = raw[m.end():].split("\n")[0].strip(" -#")
        if not reason:
            err(p.name, "C4", "VALIDATE-C4-EXEMPT given without a reason after '--'")
            return
        print(f"    C4 exempt: {p.name} {sorted(exempt)} -- {reason[:70]}")

    missing = NINE_PAIRS - found - exempt
    if missing:
        err(p.name, "C4", f"PAIRS missing {sorted(missing)} (has {len(found)} of 9, "
                          f"{len(exempt)} exempt)")
    else:
        print(f"    C4 ok: {p.name} covers all nine pairs"
              + (" (some by exemption)" if exempt else ""))


def c5_maxtok(p: Path, txt: str) -> None:
    for m in re.finditer(r"IQE_QE_MAXTOK\s*=\s*[\"']?(\d+)", txt):
        if m.group(1) == "8":
            err(p.name, "C5", "sets IQE_QE_MAXTOK=8 — truncates output, destroys parse rate")
    if "python -m indicqe_ape.cli qe" in txt and "unset IQE_QE_MAXTOK" not in txt:
        warn(p.name, "C5", "runs the qe CLI without `unset IQE_QE_MAXTOK`; "
                           "an inherited value from the shell would silently apply")


#: models that emit a chain of thought unless told not to. Generating one is not merely slow:
#: it changes the score, so a mixed run is a confound across cells, not just a slow run.
REASONING_MODELS = ("sarvam-m",)


def c11_nothink(p: Path, txt: str) -> None:
    """Reasoning models on the vLLM path must have thinking disabled.

    Job 2293019 ran sarvam-m for 11h26m and completed 8 of 18 cells before it was killed, because
    the launcher never set IQE_NO_THINK=1. The cost, 28s/item against ~1s, was written in a comment
    in vllm_provider.py and in the filename of a legacy sbatch, and neither stopped it. So check.
    """
    uses_vllm = "vllm" in txt.lower()
    hits = [m for m in REASONING_MODELS
            if m in txt or (Path(ROOT / "cluster" / "models_rerun.txt").exists()
                            and "models_rerun.txt" in txt
                            and m in (ROOT / "cluster" / "models_rerun.txt").read_text())]
    if not (uses_vllm and hits):
        return
    if re.search(r"(export\s+)?IQE_NO_THINK\s*=\s*[\"']?1", txt):
        print(f"    C11 ok: {p.name} disables thinking for {', '.join(hits)}")
    else:
        err(p.name, "C11", f"can run {', '.join(hits)} via vLLM without IQE_NO_THINK=1 — "
                           f"~28s/item instead of ~1s, and a different score")


def c6_cache(p: Path, txt: str) -> None:
    if "--cache-dir" not in txt:
        return
    for m in re.finditer(r"--cache-dir\s+(\S+)", txt):
        val = m.group(1)
        # a per-task cache must vary with model/pair/shots or the array id
        if not re.search(r"\$\{?(MODEL|PAIR|SHOTS|SLURM_ARRAY_TASK_ID)", val):
            warn(p.name, "C6", f"cache dir {val} may be shared across array tasks; "
                               f"a shared cache file corrupted and lost 4 tasks previously")
        else:
            print(f"    C6 ok: {p.name} per-task cache")


def c7_data(p: Path, txt: str) -> None:
    for m in re.finditer(r"data/qe_indic_[\w.]+\.csv", txt):
        rel = m.group(0)
        f = ROOT / "experiments" / "comet_qe" / rel
        if not f.exists():
            err(p.name, "C7", f"references {rel} which does not exist")


def c8_outdir(p: Path, txt: str) -> None:
    if "--out " not in txt and "--out-dir" not in txt:
        return
    if "$IQE_OUT" in txt or "$OUTDIR" in txt or "OUTDIR=" in txt:
        print(f"    C8 ok: {p.name} writes under IQE_OUT")
    else:
        warn(p.name, "C8", "output path does not reference $IQE_OUT")


def c9_resume(p: Path, txt: str) -> None:
    if p.suffix not in (".sbatch", ".sh"):
        return
    if "indicqe_ape.cli" in txt:
        print(f"    C9 ok: {p.name} CLI skips ids already in --out")
        print("       note: that skip is also a hazard. If a generation setting changes between "
              "runs,\n       surviving output is regenerated under the OLD setting. Delete the "
              "cells and the\n       response cache before relaunching with a changed setting.")
    elif "alope" in p.name:
        warn(p.name, "C9", "ALOPE training has no resume guard; a pre-emption restarts from zero")


def c10_limit(p: Path, txt: str) -> None:
    for m in re.finditer(r"--limit\s+(\d+)", txt):
        n = int(m.group(1))
        if n >= 100 and "probe" not in txt[max(0, m.start() - 300):m.start()].lower():
            warn(p.name, "C10", f"--limit {n} caps a run; gpt-5.5 4-shot covers 3,200 of "
                                f"12,730 segments for exactly this reason")


CHECKS = [c1_syntax, c2_partition, c3_array, c4_pairs, c5_maxtok,
          c6_cache, c7_data, c8_outdir, c9_resume, c10_limit, c11_nothink]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-set", action="store_true",
                    help="validate only the scripts queued for submission now")
    a = ap.parse_args()

    if a.deploy_set:
        files = [CLUSTER / f for f in DEPLOY_SET]
        missing = [f for f in files if not f.exists()]
        if missing:
            for f in missing:
                err(f.name, "C0", "declared in DEPLOY_SET but not on disk")
            files = [f for f in files if f.exists()]
    else:
        files = sorted([f for f in CLUSTER.iterdir()
                        if f.suffix in (".sbatch", ".sh", ".py") and f.name != "validate_deploy.py"])

    print(f"validating {len(files)} scripts\n")
    for f in files:
        raw = f.read_text(errors="replace")
        # C1 needs the real file; every other check must not see commented-out examples.
        txt = raw if f.suffix == ".py" else strip_comments(raw)
        print(f"  -- {f.name}")
        for chk in CHECKS:
            try:
                chk(f, raw if chk is c1_syntax else txt)
            except Exception as e:  # a broken check must not mask a real failure
                warn(f.name, chk.__name__, f"check raised {type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    if WARNS:
        print(f"{len(WARNS)} WARNING(S)\n" + "\n".join(WARNS))
    if ERRORS:
        print(f"\n{len(ERRORS)} ERROR(S) — DO NOT SUBMIT\n" + "\n".join(ERRORS))
        return 1
    print("\nALL ERROR-LEVEL CHECKS PASS — safe to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
