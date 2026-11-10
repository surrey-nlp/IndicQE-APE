#!/usr/bin/env python3
r"""export_evidence_bundle.py — the per-instance predictions behind every number in the paper.

WHY
Appendix J says "All analysis is CPU-only and runs from saved per-instance predictions, so every
number in this paper can be recomputed without a GPU." That is true for us and false for everyone
else: the predictions live in results/, which is 300 MB of scoring JSONL and is not published.
Nobody outside the group can re-derive a single cell.

This builds the artefact that makes the claim true. It is attached to a tagged GitHub release
rather than committed, because it is tens of megabytes and it is an output, not source.

WHAT GOES IN
Only what an analysis needs, which is the segment identifier and the score:

  qe/<model>.<shots>.<pair>.jsonl.gz    one {"id", "v"} per segment, plus the run's own _meta
                                        record so the decoding settings travel with the scores
  head/<run>.predictions_test.csv       the COMET-encoder head, per instance, as saved
  tables/*.csv                          the aggregate CSVs the generators splice from
  MANIFEST.tsv                          every file with its sha256, row count and source path
  README.md                             which file backs which table in the paper

`v` is the parsed score for a trained metric and the raw model reply for a prompted one, because
the parse rule is part of the measurement (S3) and a reader checking compliance needs the text
that was parsed, not our verdict on it.

WHAT STAYS OUT
Source and hypothesis text. Those are the dataset, they are on the Hub under their own licence,
and duplicating them here would fork the release. Every id in this bundle is a `uid` in
surrey-nlp/IndicQE-APE, so the two join on one column.

    python3 analysis/export_evidence_bundle.py                # build dist/
    python3 analysis/export_evidence_bundle.py --dry-run      # sizes only, write nothing
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
STAMP = date.today().isoformat()

#: (source glob, destination prefix). Order is the order of the manifest.
QE_DIRS = ["results/qe_closed_rerun", "results/qe_rerun512", "results/qe_metrics_ext",
           "results/qe_metrics_full", "results/ape_regen", "results/ape_open"]
HEAD_GLOB = "experiments/comet_qe/results/campaign/outputs/*/predictions_test.csv"
TABLE_GLOB = "results/analysis/*.csv"
SKIP_TABLE = ("_perinstance.csv", "llm_judge_raw", "domain_compare", "joint_qe_ape",
              "wmt_meta_all")

README = """# IndicQE-APE: evidence bundle

The per-instance predictions behind every number in the paper. Everything here is CPU-only to
re-analyse; nothing in this bundle needs a GPU.

Join key: every `id` is a `uid` in the `surrey-nlp/IndicQE-APE` dataset on the Hugging Face Hub.
Fetch the dataset for source, hypothesis, post-edit, direct assessment and the difficulty route,
then join on that one column.

## Layout

    qe/<model>.<shots>.<pair>.jsonl.gz   one record per segment: {"id", "v"}
                                         plus the run's own _meta record, first line, carrying the
                                         model snapshot and the decoding settings
    head/<run>.predictions_test.csv      COMET-encoder head, per instance (item_id, language_pair,
                                         gold, pred)
    tables/*.csv                         the aggregate CSVs the paper's generators splice from
    MANIFEST.tsv                         sha256, rows and original path for every file

`v` is the parsed score for a trained metric and the raw reply for a prompted one. The parse rule
is part of the measurement, so the text that was parsed travels with the score rather than only
our verdict on it.

## Which file backs which table

| Paper | Source in this bundle |
|---|---|
| Table 4, QE over nine pairs | `qe/*.s0.*` for the prompted models, `qe/cometkiwi-*`, `qe/xcomet-*` |
| Table 7, four-shot effect and compliance | `qe/*.s0.*` and `qe/*.s4.*` |
| Table 15, per-pair four-shot compliance | `qe/*.s4.*` |
| Table 16, per-pair Kendall | the same files as Table 4 |
| Table 5 and Table 22, APE | `tables/ape_metrics_table.csv`, `ape_comet_table.csv`, `ape_do_nothing.csv` |
| Table 8 and Table 9, the COMET head | `head/*` and `tables/seed_stability.csv` |
| Table 18, full population | `qe/cometkiwi-xl-reffree.*`, `xcomet-xl-reffree.*`, `metricx24-xl-qe.*` |
| Table 6, cross-lingual fits | `tables/recalibration_9pair.csv` |
| Table 17, direction scope | `tables/xling_direction_scope.csv` |
| Table 21, MQM by batch | `tables/mqm_provenance_split.csv` |
| Table 12, pair factors | `tables/pair_factors.csv` |

The scripts that consume these live in `analysis/` in the code repository. `bash
analysis/run_all_checks.sh` re-derives the generated tables and fails on drift; `python3
analysis/audit_all_numbers.py` traces every numeric literal in the paper back to a file here.
"""


def slim(path: Path) -> tuple[bytes, int]:
    """One {"id","v"} per scored segment, with the run's _meta kept as the first line."""
    buf = io.StringIO()
    rows = 0
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in r:
                buf.write(json.dumps(r, separators=(",", ":")) + "\n")
                continue
            if "_summary" in r or r.get("id") is None:
                continue
            v = r.get("pred_score")
            if v is None:
                raw = r.get("raw") or {}
                v = raw or (r.get("pred") or None)
            buf.write(json.dumps({"id": r["id"], "v": v}, separators=(",", ":")) + "\n")
            rows += 1
    return gzip.compress(buf.getvalue().encode()), rows


def main() -> int:
    dry = "--dry-run" in sys.argv
    entries: list[tuple[str, bytes, int, str]] = []      # arcname, payload, rows, source

    for d in QE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.glob("*.jsonl")):
            payload, rows = slim(f)
            if rows == 0:
                continue
            entries.append((f"qe/{f.stem}.jsonl.gz", payload, rows, str(f.relative_to(ROOT))))

    for f in sorted(ROOT.glob(HEAD_GLOB)):
        run = f.parent.name
        data = f.read_bytes()
        entries.append((f"head/{run}.predictions_test.csv", data,
                        data.count(b"\n") - 1, str(f.relative_to(ROOT))))

    for f in sorted(ROOT.glob(TABLE_GLOB)):
        if any(s in f.name for s in SKIP_TABLE):
            continue
        data = f.read_bytes()
        entries.append((f"tables/{f.name}", data, data.count(b"\n") - 1,
                        str(f.relative_to(ROOT))))

    total = sum(len(p) for _, p, _, _ in entries)
    print(f"{len(entries)} files, {total/1e6:.1f} MB before tar compression")
    for kind in ("qe/", "head/", "tables/"):
        sel = [e for e in entries if e[0].startswith(kind)]
        print(f"  {kind:8s} {len(sel):4d} files  {sum(len(p) for _, p, _, _ in sel)/1e6:7.1f} MB  "
              f"{sum(r for _, _, r, _ in sel):,} rows")
    if dry:
        print("\n--dry-run: nothing written")
        return 0

    man = io.StringIO()
    man.write("sha256\tbytes\trows\tpath_in_bundle\tsource_in_repo\n")
    for arc, payload, rows, src in entries:
        man.write(f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{rows}\t{arc}\t{src}\n")

    OUT.mkdir(exist_ok=True)
    tgz = OUT / f"indicqe-ape-evidence-{STAMP}.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        add("README.md", README.encode())
        add("MANIFEST.tsv", man.getvalue().encode())
        for arc, payload, _, _ in entries:
            add(arc, payload)

    sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
    print(f"\nwrote {tgz.relative_to(ROOT)}  {tgz.stat().st_size/1e6:.1f} MB")
    print(f"sha256 {sha}")
    (OUT / f"{tgz.name}.sha256").write_text(f"{sha}  {tgz.name}\n")
    print("\nAttach to a tagged release:")
    print(f"  gh release create v1.0-paper {tgz} {tgz}.sha256 \\")
    print(f"     --title 'IndicQE-APE evidence bundle' --notes-file <release notes>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
