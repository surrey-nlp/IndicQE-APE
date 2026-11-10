#!/bin/bash
#SBATCH --job-name=iqe-r5-metricx-gap
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r5-metricx-gap-%j.out

# R5 — MetricX-24 XL on the two pairs missing from the full-population run.
#
#   en-gu 9,109 + en-te 13,157 = 22,266 rows, all DA-bearing.
#
# QE MODE ONLY. These two pairs carry no post-edits (has_pe == 0 in the qe-da config), which is
# exactly why they are absent from the `ape` config that R4b scored. With no reference there is no
# reference-based mode to run; the script asserts this rather than silently producing nonsense.
#
# Completing them takes the full-population DA table from 4 en->Indic pairs to 6.
#
# Input is the pre-built, pre-validated jsonl so that this job and the workstation COMET job
# (r5_comet_gap.sh) score the IDENTICAL rows. Build it locally with r5_build_qeda_gap.py, then copy
# results/qe_metrics_full/_r5_qeda_gap_input.jsonl to $IQE_REPO/r5_qeda_gap_input.jsonl on the
# cluster (or point R5_IN at it).
#
# Env `metricx` (transformers 4.44.2), NOT `indicqe`: MetricX-24 needs the pre-4.45 mT5 mask
# convention and fails inside the decoder under 4.57.

set -euo pipefail
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
mkdir -p logs results/qe_metrics

source cluster/env.sh
activate metricx
export PYTHONPATH="$SCRATCH/metricx-repo:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

IN=${R5_IN:-r5_qeda_gap_input.jsonl}
[ -f "$IN" ] || { echo "FATAL: $IN missing -- copy it to the repo root (or set R5_IN) first"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[input] $IN : $(wc -l < "$IN") rows"

python - "$IN" <<'PY'
import json, os, sys, tempfile, torch, transformers
from metricx24 import models, predict as mxpredict

IN, OUT = sys.argv[1], "results/qe_metrics"
REPO, MAXLEN = "google/metricx-24-hybrid-xl-v2p6", 1536
BATCH = int(os.environ.get("R5_BATCH", "16"))
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
assert len(rows) == 22266, f"expected 22266 rows, got {len(rows)} -- input changed"
assert all(r.get("gold_da_mean") is not None for r in rows), "every row must carry a gold DA"
assert not any("post_edit" in r for r in rows), "input carries a reference; this job is QE-only"
print(f"[data] {len(rows):,} rows over {sorted({r['pair'] for r in rows})}", flush=True)

tok = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False, use_fast=True)
device = torch.device("cuda")
model = models.MT5ForRegression.from_pretrained(
    REPO, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval()


def score(sub):
    """Explicit batching, mirroring R4b. NOT transformers.Trainer: get_dataset tokenises with
    padding=False and keeps the raw text columns, so the stock collator pads input_ids and
    attention_mask to different widths and mT5 cross-attention fails one position short. Here both
    tensors take the same `width`, so they cannot disagree. Sorting by length keeps each batch
    near-uniform; the original order is restored via the index, so no number changes."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        for r in sub:
            # QE mode: reference is the empty string, per the MetricX-24 hybrid input contract
            tf.write(json.dumps({"source": r["source"], "hypothesis": r["mt"],
                                 "reference": ""}, ensure_ascii=False) + "\n")
        tmp = tf.name
    ds = mxpredict.get_dataset(tmp, tok, MAXLEN, device, True)["test"]   # is_qe=True
    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["input_ids"]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    vals = [None] * len(ds)
    with torch.no_grad():
        for st in range(0, len(order), BATCH):
            idx = order[st:st + BATCH]
            seqs = [ds[i] for i in idx]
            width = max(len(f["input_ids"]) for f in seqs)
            ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
            att = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
            for j, f in enumerate(seqs):
                a = torch.as_tensor(f["input_ids"], dtype=torch.long)
                m = torch.as_tensor(f["attention_mask"], dtype=torch.long)
                assert len(a) == len(m), f"ragged pair: {len(a)} vs {len(m)}"
                ids[j, :len(a)] = a.to(device); att[j, :len(m)] = m.to(device)
            mo = model(input_ids=ids, attention_mask=att)      # not `out` -- name collision
            v = getattr(mo, "predictions", None)
            if v is None:
                v = mo.logits
            v = v.float().reshape(-1).cpu().tolist()
            assert len(v) == len(idx), f"{len(v)} scores for {len(idx)} inputs"
            for i, x in zip(idx, v):
                vals[i] = float(x)
            if st % (BATCH * 200) == 0:
                print(f"      {st + len(idx)}/{len(order)}", flush=True)
    os.unlink(tmp)
    assert all(x is not None for x in vals), "some segments never scored"
    return vals


for pair in sorted({r["pair"] for r in rows}):
    dst = f"{OUT}/metricx24-xl-qe.{pair}.jsonl"
    if os.path.exists(dst):
        print(f"[skip] {dst}", flush=True)
        continue
    sub = [r for r in rows if r["pair"] == pair]
    print(f"[run ] qe {pair} n={len(sub)}", flush=True)
    vals = score(sub)
    lo, hi = min(vals), max(vals)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"metric": "metricx24-xl", "mode": "qe", "model": REPO,
                                       "pair": pair, "direction": "error_lower_is_better",
                                       "range": [0, 25], "reference": None, "n": len(sub)}},
                            ensure_ascii=False) + "\n")
        for r, v in zip(sub, vals):
            fh.write(json.dumps({"id": r["id"], "pair": r["pair"],
                                 "gold_da_mean": r["gold_da_mean"],
                                 "pred_score": v}, ensure_ascii=False) + "\n")
    print(f"[done] {dst}  n={len(sub):,}  range {lo:.3f}-{hi:.3f}", flush=True)
    if hi <= 0.001:
        print(f"  WARN {pair}: all scores ~0 -- check the input template", flush=True)

print("[ALL DONE]", flush=True)
PY
echo "=== exit=$? $(date) ==="
