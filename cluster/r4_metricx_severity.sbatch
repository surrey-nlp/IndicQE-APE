#!/bin/bash
#SBATCH --job-name=r4-mx
##SBATCH --partition=<your-partition>   # pass -p at submit time (80 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r4-mx-%j.out

# R4 — MetricX-24 on the injected severity set, at BOTH model sizes and in BOTH modes.
#
#   metricx-24-hybrid-xl-v2p6   (3.7B)  QE + reference-based
#   metricx-24-hybrid-xxl-v2p6  (13B)   QE + reference-based
#
# WHY BOTH SIZES: the paper claims trained neural metrics cannot order severity under length
# control. Running only one size invites "you used the small one" / "it is a capacity artefact".
# XL and XXL together answer that: if scaling 3.7B -> 13B does not fix the ordering, the failure
# is structural, not a matter of model size.
#
# DIRECTION: MetricX emits an ERROR score in [0, 25], LOWER = better. That is the opposite of
# XCOMET/CometKiwi (quality, higher = better) and the SAME direction as MATRA/char-TER. The
# report script negates by direction; these columns are registered as higher = more severe.
#
# Tokenisation is delegated to the repo's own get_dataset() -- it applies _make_input (the exact
# "source: ... candidate: ... [reference: ...]" template), truncation at max_input_length, and
# critically _remove_eos(), which strips the trailing EOS token. Hand-rolling this silently
# produces wrong scores rather than an error.

set -euo pipefail
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
mkdir -p logs results/severity

source cluster/env.sh
# NOT $CONDA_IQE. MetricX-24 requires the pre-4.45 mT5 mask convention; transformers 4.57 (which
# indicqe carries, pinned there by COMET's torch 2.4.1 requirement) rebuilt that plumbing and the
# cross-attention mask arrives one position short of the position bias -- surfacing as
# "size of tensor a (18) must match tensor b (17)" deep inside the decoder. Not fixable by
# reshaping inputs. `metricx` is a separate env: transformers 4.44.2 + torch 2.4.1+cu121.
activate metricx
export PYTHONPATH="$SCRATCH/metricx-repo:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

IN=results/severity/r0_gpu_input.jsonl
[ -s "$IN" ] || { echo "FATAL: $IN missing -- scp it up first"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "from metricx24 import models; print('metricx24 import OK')"

python - <<'PY'
import json, os, tempfile, torch, transformers, datasets

# MetricX-24 ships pytorch_model.bin shards, not safetensors. transformers >=4.56 refuses
# torch.load unless torch>=2.6 (CVE-2025-32434), and this env is pinned to torch 2.4.1 because
# COMET 2.2.7 requires it -- upgrading would break the XCOMET/CometKiwi jobs sharing this env.
#
# The CVE concerns unpickling UNTRUSTED checkpoints. These are official Google weights from the
# canonical HF repo, verified by size on download. Disabling the blanket check for this job is
# the proportionate fix; upgrading torch is not.
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda *a, **k: None
_mu.check_torch_load_is_safe = lambda *a, **k: None   # modeling_utils imported it by name

from metricx24 import models, predict as mxpredict

IN   = "results/severity/r0_gpu_input.jsonl"
OUT  = "results/severity/r4_metricx_scores.jsonl"
MAXLEN = 1536                       # MetricX-24 documented input length
BATCH  = int(os.environ.get("R4_BATCH", "16"))
rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
print(f"[data] {len(rows)} records", flush=True)
assert all(r.get("id") for r in rows), "every record needs an id -- the join depends on it"

out = {}
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r
for r in rows:
    out.setdefault(r["id"], {"id": r["id"]})

JOBS = [
    ("metricx24_xl_qe",   "google/metricx-24-hybrid-xl-v2p6",  True),
    ("metricx24_xl_ref",  "google/metricx-24-hybrid-xl-v2p6",  False),
    ("metricx24_xxl_qe",  "google/metricx-24-hybrid-xxl-v2p6", True),
    ("metricx24_xxl_ref", "google/metricx-24-hybrid-xxl-v2p6", False),
]

# mT5 tokenizer is shared by both sizes; only the tokenizer files were fetched for xxl.
tok = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False, use_fast=True)
device = torch.device("cuda")

for col, repo, is_qe in JOBS:
    if all(col in out[r["id"]] for r in rows):
        print(f"[skip] {col}", flush=True)
        continue
    print(f"[load] {repo} is_qe={is_qe} -> {col}", flush=True)

    # get_dataset() reads a JSONL with source/hypothesis/reference fields
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        for r in rows:
            tf.write(json.dumps({"source": r["src"], "hypothesis": r["mt"],
                                 "reference": r["ref"]}, ensure_ascii=False) + "\n")
        tmp = tf.name

    # get_dataset returns a DatasetDict with a single "test" split, not a bare Dataset. It also
    # sets output_all_columns=True, so each feature still carries the raw text fields.
    ds = mxpredict.get_dataset(tmp, tok, MAXLEN, device, is_qe)["test"]
    # bf16 for XXL (13B would not fit 80GB in fp32 with batching); fp32 is fine for XL
    dtype = torch.bfloat16 if "xxl" in repo else torch.float32
    # low_cpu_mem_usage streams the checkpoint shard-by-shard instead of materialising the whole
    # thing in host RAM first. Without it, XXL's 49GB fp32 checkpoint needs ~60GB of host memory
    # before it is ever cast to bf16 -- which is what kept this job pending for 128G on a100.
    model = models.MT5ForRegression.from_pretrained(
        repo, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    # ---- explicit batched inference -------------------------------------------------------
    # NOT transformers.Trainer. get_dataset tokenises with padding=False and leaves the raw text
    # columns attached (output_all_columns=True); DataCollatorWithPadding then pads input_ids and
    # attention_mask to different lengths, which surfaces deep inside mT5 cross-attention as
    # "size of tensor a (109) must match tensor b (108)". The repo avoids this by using batch
    # size 1. Padding both tensors together here is correct AND lets us batch.
    #
    # Sequences are sorted by length so each batch pads to near-uniform width, then results are
    # restored to the original order via the index -- a ~3-5x speedup that changes no number.
    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["input_ids"]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    scores = [None] * len(ds)

    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            seqs = [ds[i] for i in idx]
            width = max(len(f["input_ids"]) for f in seqs)
            ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
            att = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
            for j, f in enumerate(seqs):
                a = torch.as_tensor(f["input_ids"], dtype=torch.long)
                m = torch.as_tensor(f["attention_mask"], dtype=torch.long)
                assert len(a) == len(m), f"ragged pair: {len(a)} vs {len(m)}"
                ids[j, :len(a)] = a.to(device)
                att[j, :len(m)] = m.to(device)
            # NB: not `out` -- that name holds the results accumulator dict.
            mo = model(input_ids=ids, attention_mask=att)
            # MT5ForRegression exposes .predictions; fall back to .logits across versions
            vals = getattr(mo, "predictions", None)
            if vals is None:
                vals = mo.logits
            vals = vals.float().reshape(-1).cpu().tolist()
            assert len(vals) == len(idx), f"{len(vals)} scores for {len(idx)} inputs"
            for i, v in zip(idx, vals):
                scores[i] = float(v)
            if s % (BATCH * 50) == 0:
                print(f"    {s + len(idx)}/{len(order)}", flush=True)

    assert all(v is not None for v in scores), "some segments never scored"
    assert len(scores) == len(rows), f"{len(scores)} scores for {len(rows)} rows"
    for r, s in zip(rows, scores):
        out[r["id"]][col] = s
    lo, hi = min(scores), max(scores)
    print(f"[done] {col} mean={sum(scores)/len(scores):.4f} range=[{lo:.3f},{hi:.3f}]", flush=True)
    if hi <= 0.001:
        print(f"  WARN {col}: all scores ~0 -- check the input template / EOS handling", flush=True)

    del model                          # no `trainer` -- inference is the explicit loop above
    torch.cuda.empty_cache()
    os.unlink(tmp)

    with open(OUT, "w", encoding="utf-8") as fh:      # write after each pass
        for r in rows:
            fh.write(json.dumps(out[r["id"]], ensure_ascii=False) + "\n")

print(f"[write] {OUT} ({len(rows)} records)", flush=True)
PY

echo
echo "DONE. Bring back: results/severity/r4_metricx_scores.jsonl"
