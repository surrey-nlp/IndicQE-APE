#!/usr/bin/env python3
"""Integrate the en-hi MQM data (WMT24 + EAMT-QE MQM annotations) into HuggingFace.

The raw vendor MQM exports this script reads (dataset/mqm_enhi_raw/*.tsv) are not redistributed
in this repository; the processed result is the released `mqm` configuration and the mqm_*
columns of `full`. The script is kept for provenance.

Design decisions (2026-07-25):
  * Bucket A (2,319 exact + 8 loose = 2,327 segments that match existing en-hi instances on
    (source, mt)) -> MQM as COLUMNS on those instances in the `full` config. Row count of `full`
    is UNCHANGED; only the new mqm_* columns are added, null everywhere except the 2,327 en-hi uids.
  * Bucket B (2,013 WMT24 general-MT + 147 CS-lecture = 2,160 novel segments) -> NEW instances that
    live ONLY in a new `mqm` configuration, so no other config's per-pair counts move.
  * The `mqm` config = the 2,327 Bucket-A rows (their full row data + mqm columns) + the 2,160
    Bucket-B rows = 4,487 en-hi MQM-annotated segments.

Merge rules:
  * mqm_score = MIN across annotation passes (worst-case; user decision). 6 segments have >1 pass;
    only "Ayrshire" (91.24 vs 99.65) is a real disagreement.
  * mqm_spans = list of {typology_error, severity, start_index, end_index, annotator, comment},
    union of all error rows across passes. Also stored as mqm_spans_json (a JSON string) -- both.
  * mqm_minor/major/critical = error counts by severity. mqm_batch, mqm_annotator, mqm_source
    carry provenance.
  * Bucket B provenance: WMT24 domains -> sources=wmt24-genmt-mqm, domain in
    {literary,news,social,speech}; lecture -> sources=eamt-qe-mqm, domain=lecture. All test split.
  * Bucket B uids are minted as md5("mqm|source|mt|en-hi")[:16] and ASSERTED not to collide with
    any existing uid.

Six gates, dry-run first, re-gate against the live repo after push. Same discipline as
integrate_deepseek_v4pro.py.

Usage:
    python3 scripts/integrate_mqm_enhi.py --dry-run
    HF_TOKEN=... python3 scripts/integrate_mqm_enhi.py --push
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, re, sys
from collections import defaultdict
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

REPO = "surrey-nlp/IndicQE-APE"
RAW = os.path.join(os.path.dirname(__file__), "..", "dataset", "mqm_enhi_raw")
WORK = "/tmp/iqe_mqm"
SPLITS = ("train", "validation", "test")

N_BUCKET_A = 2327      # 2319 exact + 8 loose
N_BUCKET_B = 2163      # WMT24 general-MT + CS-lecture novel segments (4490 - 2327)
N_MQM_TOTAL = 4490     # every unique (source,target) MQM segment
FULL_ROWS = 126754
FULL_ENHI = 21092

MQM_COLS = ["mqm_score", "mqm_spans", "mqm_spans_json", "mqm_minor", "mqm_major",
            "mqm_critical", "mqm_batch", "mqm_annotator", "mqm_source"]

DOMAIN_OF_BATCH = {
    "WMT24_English-Hindi_Literary": ("literary", "wmt24-genmt-mqm"),
    "WMT24_English-Hindi_News": ("news", "wmt24-genmt-mqm"),
    "WMT24_English-Hindi_Social": ("social", "wmt24-genmt-mqm"),
    "WMT24_English-Hindi_Speech1": ("speech", "wmt24-genmt-mqm"),
    "WMT24_English-Hindi_Speech2": ("speech", "wmt24-genmt-mqm"),
    "WMT24_English-Hindi_Speech3": ("speech", "wmt24-genmt-mqm"),
}


def norm(s): return re.sub(r"\s+", " ", (s or "").strip())
def h(s): return hashlib.md5(norm(s).encode()).hexdigest()[:12]
def hloose(s):
    s = norm(s).lower(); s = re.sub(r"[\"'`.,;:!?()\-–—]", "", s)
    return hashlib.md5(re.sub(r"\s+", " ", s).strip().encode()).hexdigest()[:12]
def mint_uid(src, mt): return hashlib.md5(f"mqm|{norm(src)}|{norm(mt)}|en-hi".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. parse the raw TSVs to one record per (source, target) segment
# ---------------------------------------------------------------------------
def parse_mqm():
    seg = {}  # key (src,tgt) -> record
    for f in sorted(glob.glob(os.path.join(RAW, "*.tsv"))):
        if "concat_errors" in f:
            continue                     # WMT24-only consolidation; per-batch files cover everything
        batch_default = os.path.basename(f).replace(".tsv", "")
        for r in csv.DictReader(open(f, encoding="utf-8"), delimiter="\t"):
            src = norm(r.get("source_text")); tgt = norm(r.get("target_text"))
            if not src or not tgt:
                continue
            k = (src, tgt)
            rec = seg.setdefault(k, {"source": src, "mt": tgt, "scores": [], "spans": [],
                                     "batches": set(), "annotators": set()})
            try:
                rec["scores"].append(float(r["mqm"]))
            except (TypeError, ValueError):
                pass
            rec["batches"].add(r.get("batch_name") or batch_default)
            if norm(r.get("annotator")):
                rec["annotators"].add(norm(r.get("annotator")))
            typ = norm(r.get("typology_error"))
            if typ:                      # an error row; record the span
                span = {"typology_error": typ, "severity": norm(r.get("severity")),
                        "start_index": _int(r.get("start_index")), "end_index": _int(r.get("end_index")),
                        "annotator": norm(r.get("annotator")), "comment": norm(r.get("annotator_comment"))}
                rec["spans"].append(span)
    # finalise
    out = {}
    for k, rec in seg.items():
        spans = rec["spans"]
        sev = [s["severity"] for s in spans]
        out[k] = {
            "source": rec["source"], "mt": rec["mt"],
            "mqm_score": min(rec["scores"]) if rec["scores"] else None,   # MIN = worst case
            "mqm_spans": spans,
            "mqm_spans_json": json.dumps(spans, ensure_ascii=False),
            "mqm_minor": sev.count("minor"), "mqm_major": sev.count("major"),
            "mqm_critical": sev.count("critical"),
            "mqm_batch": ",".join(sorted(rec["batches"])),
            "mqm_annotator": ",".join(sorted(rec["annotators"])),
        }
    return out


def _int(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None


# ---------------------------------------------------------------------------
# 2. match against HF en-hi (exact then loose) -> Bucket A (uid) / Bucket B (novel)
# ---------------------------------------------------------------------------
def load_full():
    d = {}
    for sp in SPLITS:
        p = hf_hub_download(REPO, f"data/full/{sp}.parquet", repo_type="dataset")
        d[sp] = pd.read_parquet(p)
    return d


def classify(mqm, full):
    hi = pd.concat([v[v.language_pair == "en-hi"] for v in full.values()], ignore_index=True)
    exact = {(h(s), h(m)): u for s, m, u in zip(hi.source, hi.mt, hi.uid)}
    loose = {(hloose(s), hloose(m)): u for s, m, u in zip(hi.source, hi.mt, hi.uid)}
    bucketA = {}   # uid -> mqm record
    bucketB = []   # list of mqm records (novel)
    for (src, tgt), rec in mqm.items():
        u = exact.get((h(src), h(tgt))) or loose.get((hloose(src), hloose(tgt)))
        if u is not None:
            bucketA[u] = rec           # last write wins; uids are unique in exact/loose maps
        else:
            bucketB.append(rec)
    return bucketA, bucketB


# ---------------------------------------------------------------------------
# 3. build the two artefacts: full-with-columns, and the mqm config
# ---------------------------------------------------------------------------
def blank_mqm():
    return {"mqm_score": None, "mqm_spans": None, "mqm_spans_json": None, "mqm_minor": None,
            "mqm_major": None, "mqm_critical": None, "mqm_batch": None, "mqm_annotator": None,
            "mqm_source": None}


def add_columns(df, bucketA, source_tag_map=None):
    """Left-join mqm columns onto df by uid. source_tag defaults to the batch's tag."""
    df = df.reset_index(drop=True)          # guarantee a clean, unique index for axis=1 concat
    recs = []
    for uid in df["uid"]:
        r = bucketA.get(uid)
        if r is None:
            recs.append(blank_mqm())
        else:
            m = {k: r[k] for k in MQM_COLS if k != "mqm_source"}
            # source tag: eamt-qe-mqm for the Research batch, else the batch tag
            tag = "eamt-qe-mqm" if "EAMT" in (r["mqm_batch"] or "") or "Research" in (r["mqm_batch"] or "") \
                  else "wmt24-genmt-mqm"
            m["mqm_source"] = tag
            recs.append(m)
    add = pd.DataFrame(recs).reset_index(drop=True)
    # assign per-column: `mqm_score`/`mqm_spans` are RESERVED (already-empty) columns in the
    # schema, so we populate them in place; the rest are genuinely new columns. Either way this
    # never creates a duplicate column.
    for c in MQM_COLS:
        df[c] = add[c].values
    return df


def build_bucketB_rows(bucketB, template_cols, existing_uids):
    rows = []
    lecture_srcset = None
    for rec in bucketB:
        batch = (rec["mqm_batch"] or "").split(",")[0]
        if batch in DOMAIN_OF_BATCH:
            domain, tag = DOMAIN_OF_BATCH[batch]
        else:                            # the 147 CS-lecture (Research_EAMT_QE novel)
            domain, tag = "lecture", "eamt-qe-mqm"
        uid = mint_uid(rec["source"], rec["mt"])
        if uid in existing_uids:
            raise SystemExit(f"ABORT: minted uid {uid} collides with an existing uid")
        row = {c: None for c in template_cols}
        row.update({
            "uid": uid, "language_pair": "en-hi", "src_lang": "en", "tgt_lang": "hi",
            "source": rec["source"], "mt": rec["mt"], "reference": None,
            "domain": domain, "domain_provenance": "annotated",
            "sources": tag, "cadence_id": f"WMT24MQM-ENHI-{uid[:8]}",
            "is_curated": False,
            "mqm_score": rec["mqm_score"], "mqm_spans": rec["mqm_spans"],
            "mqm_spans_json": rec["mqm_spans_json"], "mqm_minor": rec["mqm_minor"],
            "mqm_major": rec["mqm_major"], "mqm_critical": rec["mqm_critical"],
            "mqm_batch": rec["mqm_batch"], "mqm_annotator": rec["mqm_annotator"],
            "mqm_source": tag,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gates(full_before, full_after, mqm_cfg, bucketA, label):
    print(f"\n--- validation gates ({label}) ---")
    ok = True
    def chk(n, desc, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] gate {n}: {desc}" + (f"  ({detail})" if detail else ""))

    b = pd.concat(full_before.values(), ignore_index=True) if isinstance(full_before, dict) else full_before
    a = pd.concat(full_after.values(), ignore_index=True) if isinstance(full_after, dict) else full_after

    # 1 row count of full unchanged
    chk(1, f"full row count unchanged at {FULL_ROWS}", len(a) == FULL_ROWS, f"actual={len(a)}")
    # 2 uid set of full unchanged
    chk(2, "full uid set unchanged (no rows added/removed)", set(a.uid) == set(b.uid),
        f"added={len(set(a.uid)-set(b.uid))} removed={len(set(b.uid)-set(a.uid))}")
    # 3 every non-mqm column byte-identical on full (sort by uid; null-safe string compare,
    #    robust to list-valued cells like da_scores/wtag and to pandas index quirks)
    common = [c for c in b.columns if c not in MQM_COLS and c != "uid"]
    bb = b.sort_values("uid").reset_index(drop=True)
    aa = a.sort_values("uid").reset_index(drop=True)
    diffs = []
    if not bb["uid"].equals(aa["uid"]):
        diffs.append("uid-order")
    SENT = "\0<null>"
    for c in common:
        x = bb[c].astype(str).where(bb[c].notna(), SENT)
        y = aa[c].astype(str).where(aa[c].notna(), SENT)
        n = int((x.values != y.values).sum())
        if n:
            diffs.append(f"{c}:{n}")
    chk(3, "all pre-existing full columns byte-identical", len(diffs) == 0,
        "; ".join(diffs[:5]))
    # 4 mqm_score populated on exactly the 2,327 en-hi uids, null elsewhere (numpy: no align)
    nn = a["mqm_score"].notna().to_numpy()
    is_enhi = (a["language_pair"] == "en-hi").to_numpy()
    enhi_nn = int((nn & is_enhi).sum())
    other_nn = int((nn & ~is_enhi).sum())
    chk(4, f"mqm_score non-null on exactly {N_BUCKET_A} en-hi rows, 0 elsewhere",
        enhi_nn == N_BUCKET_A and other_nn == 0, f"en-hi={enhi_nn}, other={other_nn}")
    # 5 mqm config: size, membership, Bucket-A rows match full
    chk(5, f"mqm config has {N_MQM_TOTAL} rows", len(mqm_cfg) == N_MQM_TOTAL, f"actual={len(mqm_cfg)}")
    inA = set(mqm_cfg.uid) & set(a.uid)
    novel = set(mqm_cfg.uid) - set(a.uid)
    chk(6, f"mqm config = {N_BUCKET_A} existing (Bucket A) + {N_BUCKET_B} novel (Bucket B), no uid collision",
        len(inA) == N_BUCKET_A and len(novel) == N_BUCKET_B and mqm_cfg.uid.nunique() == len(mqm_cfg),
        f"existing={len(inA)}, novel={len(novel)}, unique={mqm_cfg.uid.nunique()}")
    print(f"--- {'ALL GATES PASS' if ok else 'GATES FAILED'} ({label}) ---")
    return ok


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    print("parsing raw MQM TSVs ...")
    mqm = parse_mqm()
    print(f"  {len(mqm)} unique (source,target) MQM segments")

    print("downloading full and classifying ...")
    full = load_full()
    existing_uids = set(pd.concat(full.values(), ignore_index=True).uid)
    bucketA, bucketB = classify(mqm, full)
    print(f"  Bucket A (columns on existing): {len(bucketA)}")
    print(f"  Bucket B (novel, -> mqm config): {len(bucketB)}")
    if len(bucketA) != N_BUCKET_A or len(bucketB) != N_BUCKET_B:
        print(f"  !! expected A={N_BUCKET_A} B={N_BUCKET_B}; investigate before proceeding")

    # build full-with-columns (per split; no _split column is introduced -- the split is the file)
    full_after = {sp: add_columns(df, bucketA) for sp, df in full.items()}
    template_cols = list(full_after["train"].columns)

    # build the mqm config SPLIT-AWARE: Bucket A rows keep their original split (they come from
    # the corresponding full_after[sp]); Bucket B is test-only.
    b_rows = build_bucketB_rows(bucketB, template_cols, existing_uids).reindex(columns=template_cols)
    mqm_parts = {}
    for sp in SPLITS:
        aA = full_after[sp][full_after[sp].uid.isin(bucketA)]
        mqm_parts[sp] = pd.concat([aA, b_rows], ignore_index=True) if sp == "test" else aA.reset_index(drop=True)
    mqm_cfg = pd.concat(mqm_parts.values(), ignore_index=True)   # 2327 A + 2163 B = 4490

    if not gates(full, full_after, mqm_cfg, bucketA, "pre-push, local build"):
        print("\nABORT — gates failed."); return 1

    # write artefacts
    for sp, df in full_after.items():
        df.to_parquet(f"{WORK}/full_{sp}.parquet", index=False)
    for sp, part in mqm_parts.items():
        part.to_parquet(f"{WORK}/mqm_{sp}.parquet", index=False)
    print(f"\nbuilt artefacts in {WORK}")
    print(f"  mqm split sizes: {{ {', '.join(f'{sp}:{len(p)}' for sp,p in mqm_parts.items())} }}")

    if not a.push:
        print("\ndry run — nothing pushed. Re-run with --push (HF_TOKEN set).")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — cannot push."); return 1
    api = HfApi(token=token)
    for sp in SPLITS:
        api.upload_file(path_or_fileobj=f"{WORK}/full_{sp}.parquet",
                        path_in_repo=f"data/full/{sp}.parquet", repo_id=REPO, repo_type="dataset",
                        commit_message="full: add en-hi MQM columns (WMT24 + EAMT-QE; Bucket A)")
        api.upload_file(path_or_fileobj=f"{WORK}/mqm_{sp}.parquet",
                        path_in_repo=f"data/tasks/mqm/{sp}.parquet", repo_id=REPO, repo_type="dataset",
                        commit_message="add mqm config (en-hi MQM subset: Bucket A + Bucket B)")
        print(f"  pushed {sp}")

    print("\nre-downloading to re-gate against the live repo ...")
    full2 = load_full()
    full2_after = full2  # already has columns now
    mqm2 = pd.concat([pd.read_parquet(hf_hub_download(REPO, f"data/tasks/mqm/{sp}.parquet", repo_type="dataset"))
                      for sp in SPLITS], ignore_index=True)
    # rebuild "before" without mqm cols for the byte-identity gate
    before = {sp: df.drop(columns=[c for c in MQM_COLS if c in df.columns]) for sp, df in full.items()}
    bA2 = {u for u in bucketA}
    if not gates(before, {sp: df for sp, df in full2.items()}, mqm2, bucketA, "post-push, live repo"):
        print("\nPOST-PUSH GATES FAILED — investigate."); return 1
    print("\nMQM integration complete and verified against the live repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
