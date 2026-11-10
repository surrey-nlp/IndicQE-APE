#!/bin/bash
#SBATCH --job-name=r4b-mx
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/r4b-mx-%j.out

# R4b — MetricX-24 XL over the FULL ape config (73,829 rows), QE and reference-based.
#
# Companion to R4 (severity set). The two answer different questions and neither substitutes
# for the other:
#
#   R4  (6,372 injected)  -> severity ORDERING. Needs char-matched pairs with known MQM
#                            severities. mqm_score is 100% null in this release, so the MQM
#                            prior exists only on the synthetic set -- this is the only
#                            population where severity PA is computable at all.
#   R4b (73,829 real)     -> DA CORRELATION. Puts MetricX on the same footing as XCOMET-XL
#                            (R1) and CometKiwi-XL (R2), which both ran the full population.
#                            Without it MetricX is simply missing from that table.
#
# XL only, not XXL: 73,829 x 2 modes = 147,658 inferences, and XXL is ~4x slower. XXL earns its
# place on the severity set, where the capacity argument matters. Here it would cost a day of
# GPU to answer a question XL already answers.
#
# Post-edit is the reference (HTER sense) -- no independent references exist. MetricX emits an
# ERROR score in [0,25], LOWER = better, so correlation with DA is expected NEGATIVE.

set -euo pipefail
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
mkdir -p logs results/qe_metrics

source cluster/env.sh
activate metricx                      # NOT indicqe -- see the version note in r4_metricx_severity
export PYTHONPATH="$SCRATCH/metricx-repo:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export IQE_PARQUET="$SCRATCH/indicqeape/ape_parquet"
[ -d "$IQE_PARQUET" ] || { echo "FATAL: $IQE_PARQUET missing"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import json, os, tempfile, torch, transformers
import pandas as pd
from metricx24 import models, predict as mxpredict

SRC, OUT = os.environ["IQE_PARQUET"], "results/qe_metrics"
REPO, MAXLEN = "google/metricx-24-hybrid-xl-v2p6", 1536
BATCH = int(os.environ.get("R4_BATCH", "16"))
os.makedirs(OUT, exist_ok=True)

df = pd.concat([pd.read_parquet(f"{SRC}/{sp}.parquet")
                for sp in ("train", "validation", "test")], ignore_index=True)
df = df[df.mt.astype(str).str.strip().ne("") & df.post_edit.astype(str).str.strip().ne("")]
print(f"[data] {len(df)} rows over {sorted(df.language_pair.unique())}", flush=True)
assert len(df) == 73829, f"expected 73829 rows, got {len(df)} -- HF release changed"

tok = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False, use_fast=True)
device = torch.device("cuda")
model = models.MT5ForRegression.from_pretrained(
    REPO, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval()


def score(sub, is_qe):
    """Explicit batching -- NOT transformers.Trainer. get_dataset tokenises with padding=False
    and keeps the raw text columns, so the stock collator pads input_ids and attention_mask to
    different widths. Here both come from one `width`, so they cannot disagree."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        for s, m, r in zip(sub.source, sub.mt, sub.post_edit):
            tf.write(json.dumps({"source": str(s), "hypothesis": str(m),
                                 "reference": str(r)}, ensure_ascii=False) + "\n")
        tmp = tf.name
    ds = mxpredict.get_dataset(tmp, tok, MAXLEN, device, is_qe)["test"]
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


for mode, is_qe in (("qe", True), ("refbased", False)):
    for pair, sub in df.groupby("language_pair"):
        dst = f"{OUT}/metricx24-xl-{mode}.{pair.replace('-', '')}.jsonl"
        if os.path.exists(dst):
            print(f"[skip] {dst}", flush=True)
            continue
        print(f"[run ] {mode} {pair} n={len(sub)}", flush=True)
        vals = score(sub, is_qe)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"metric": "metricx24-xl", "mode": mode, "model": REPO,
                                           "pair": pair, "direction": "error_lower_is_better",
                                           "range": [0, 25],
                                           "reference": "post_edit" if not is_qe else None,
                                           "n": len(sub)}}, ensure_ascii=False) + "\n")
            for uid, oid, sc, gold in zip(sub.uid, sub.original_id, vals, sub.da_mean):
                fh.write(json.dumps({"id": str(uid), "original_id": str(oid), "pair": pair,
                                     "gold_da_mean": None if pd.isna(gold) else float(gold),
                                     "pred_score": sc}, ensure_ascii=False) + "\n")
        mean = sum(vals) / len(vals)
        print(f"[done] {dst} mean={mean:.4f}", flush=True)
        if mean < 0.01 or mean > 24.9:
            print("  WARN saturated mean -- check the input template", flush=True)
PY

echo "DONE. Bring back: results/qe_metrics/metricx24-xl-*.jsonl"
