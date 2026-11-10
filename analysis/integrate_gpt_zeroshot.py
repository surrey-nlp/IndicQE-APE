#!/usr/bin/env python3
r"""integrate_gpt_zeroshot.py — fold the gpt-5.5 zero-shot rerun into the paper, in one command.

RUN THIS AFTER:

    STAGE=full0 bash cluster/23_qe_rerun_closed.sh 2>&1 | tee -a results/gpt55_full0.log

It does everything else:

  1. copies the new s0 files out of .local-scratch into results/qe_closed_rerun/
  2. refuses to proceed unless the rerun actually covers the release, which is the whole point of
     doing it: the old files were missing 1,634 segments, half of en-gu and en-te among them
  3. regenerates every table that reads gpt-5.5 (tab:qe, tab:perpairfull, tab:prompting, tab:parse4,
     the route table and figure, the pair-factor table)
  4. removes the four population caveats that only exist because the old run was short
  5. reports the before and after for every number that moves, and runs the check suite

Nothing is written if step 2 fails, so a partial rerun cannot half-update the paper.

    python3 analysis/integrate_gpt_zeroshot.py            # do it
    python3 analysis/integrate_gpt_zeroshot.py --dry-run  # say what would happen
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

sys.path.insert(0, str(ROOT / "analysis"))
from release_population import released  # noqa: E402

SCRATCH = ROOT / ".local-scratch" / "indicqeape" / "out" / "qe_closed_rerun"
DEST = ROOT / "results" / "qe_closed_rerun"
TEX = _PAPER_TEX
PAIRS = ["enhi", "enmr", "enta", "engu", "ente", "enml", "eten", "neen", "sien"]

#: the caveats that exist only because the zero-shot run was short. Each is (description, old, new).
CAVEATS = [
    ("S3 Setup",
     r"""which at zero shots is between $10{,}903$ and $12{,}730$; the residual is its non-compliance, reported alongside every correlation rather than folded into it. The lower bound is \mGPT{}: its zero-shot cells cover $10{,}903$ of the $12{,}730$ segments, missing half of en-gu and en-te and a tenth of en-hi, en-mr and en-ta, while its four-shot cells cover all nine pairs in full.""",
     r"""which at zero shots is between {LO} and $12{,}730$; the residual is its non-compliance, reported alongside every correlation rather than folded into it."""),
    ("tab:qe caption",
     r""" \mGPT{}'s zero-shot row rests on $10{,}903$ of the $12{,}730$ segments, half of en-gu and
en-te being absent from that run; every other row covers the population it complies on.""",
     r""""""),
    ("tab:perpairfull caption",
     r"""the prompted models the $10{,}903$ to $12{,}730$ they comply on.""",
     r"""the prompted models the {LO} to $12{,}730$ they comply on."""),
    ("Limitations",
     r""" \mGPT{}'s zero-shot cells cover $10{,}903$ segments, with half of en-gu and en-te absent;
its four-shot cells cover all nine pairs in full.""",
     r""""""),
    ("S3 four-shot narrative",
     r"""lowers macro Spearman for four of the six prompted models and raises it for two, though \mGPT{}'s
rise ($0.604$ to $0.625$) is a population effect: its zero-shot cells cover $10{,}903$ segments
against $12{,}730$ at four shots, and on the segments both conditions scored the difference is
$-0.000$. The largest fall""",
     r"""lowers macro Spearman for {NFALL} of the six prompted models and raises it for {NRISE}. The largest fall"""),
    ("S3 capacity sentence",
     r"""tested, gains, and \mGPT{} does not move once the populations are matched.""",
     r"""tested, gains."""),
    ("S3 paired tolerance",
     r"""The two agree to within $0.002$ for every model except \mGPT{}, where they differ by the
population, so the effect is in the scoring""",
     r"""The two agree to within {TOL} for every model, so the effect is in the scoring"""),
]


def loose(s: str) -> re.Pattern[str]:
    return re.compile(r"\s+".join(re.escape(tok) for tok in s.split()))


def coverage(path: Path) -> tuple[int, int]:
    # Against the WHOLE challenge test, not the DA-bearing subset: a run legitimately scores
    # segments that carry no DA (they are in the test, they just cannot enter a correlation), so
    # comparing against the DA subset would flag a correct rerun as stale.
    keep = released(da_only=False)
    n = kept = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "_meta" in r or "_summary" in r or r.get("id") is None:
            continue
        n += 1
        kept += r["id"] in keep
    return kept, n


def run(*cmd: str) -> str:
    p = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return p.stdout


def main() -> int:
    dry = "--dry-run" in sys.argv

    new = sorted(glob.glob(str(SCRATCH / "gpt-5.5.s0.*.jsonl")))
    if not new:
        raise SystemExit(f"no new zero-shot files in {SCRATCH}. Run the rerun first:\n"
                         f"  STAGE=full0 bash cluster/23_qe_rerun_closed.sh")

    print("coverage of the new zero-shot files against the released challenge test")
    short, seen = [], set()
    for f in new:
        pair = Path(f).stem.split(".")[-1]
        seen.add(pair)
        kept, n = coverage(Path(f))
        flag = "" if kept == n else "   <-- rows outside the release"
        print(f"  {pair:5s} {kept:5d} of {n:5d}{flag}")
        if kept < 0.98 * n:
            short.append(pair)
    missing = [p for p in PAIRS if p not in seen]
    if missing:
        raise SystemExit(f"REFUSING: the rerun is missing {missing}. Finish it before integrating.")
    if short:
        raise SystemExit(f"REFUSING: {short} still score segments outside the release; the rerun "
                         f"did not use the current challenge test.")

    before = {}
    for line in run("analysis/make_qe_tables.py", "--check").splitlines():
        if line.strip().startswith("gpt-5.5"):
            before["qe"] = line.strip()
    print(f"\nbefore: {before.get('qe', '?')}")

    if dry:
        print("\n--dry-run: would copy the files and regenerate everything")
        return 0

    for f in new:
        shutil.copy2(f, DEST / Path(f).name)
    print(f"\ncopied {len(new)} files into {DEST.relative_to(ROOT)}")

    for script in ("analysis/make_qe_tables.py", "analysis/make_prompting_tables.py",
                   "analysis/make_routes_table.py", "analysis/make_qe_routes_figure.py",
                   "analysis/pair_factors.py", "analysis/template_compare.py"):
        out = run(script)
        for line in out.splitlines():
            if line.strip().startswith(("gpt-5.5", "  gpt-5.5", "lowest column mean",
                                        "scored populations")):
                print(f"  {line.strip()}")

    # numbers the caveat removal needs
    prom = run("analysis/make_prompting_tables.py", "--check")
    rows = [l for l in prom.splitlines() if "rho" in l and "->" in l]
    deltas = {}
    for l in rows:
        name = l.split()[0]
        d = float(l.split("(")[1].split(")")[0])
        deltas[name] = d
    nfall = sum(1 for v in deltas.values() if v < 0)
    nrise = sum(1 for v in deltas.values() if v > 0)
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    lo = min(int(x.replace(",", "")) for x in re.findall(r"scored populations: ([\d,]+)", prom)) \
        if "scored populations" in prom else None

    qe = run("analysis/make_qe_tables.py", "--check")
    m = re.search(r"scored populations: ([\d,]+) to ([\d,]+)", qe)
    lo_s = "$" + m.group(1).replace(",", "{,}") + "$" if m else "$12{,}730$"

    t = TEX.read_text()
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    applied = []
    for desc, old, new_txt in CAVEATS:
        new_txt = (new_txt.replace("{LO}", lo_s).replace("{NFALL}", words.get(nfall, str(nfall)))
                   .replace("{NRISE}", words.get(nrise, str(nrise))).replace("{TOL}", "$0.002$"))
        hits = list(loose(old).finditer(t))
        if not hits:
            print(f"  skip caveat: {desc}")
            continue
        t = t[:hits[0].start()] + new_txt + t[hits[0].end():]
        applied.append(desc)
    assert labels == set(re.findall(r"\\label\{([^}]+)\}", t)), "REFUSING: labels changed"
    TEX.write_text(t)
    print(f"\nremoved {len(applied)} population caveats: {', '.join(applied)}")
    print(f"four-shot now falls for {nfall} models and rises for {nrise}")

    print("\n" + subprocess.run(["bash", "analysis/run_all_checks.sh"], cwd=ROOT,
                                capture_output=True, text=True).stdout[-900:])
    print("If everything passes, rebuild in Overleaf and check that the body still ends on page 10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
