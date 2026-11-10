#!/usr/bin/env bash
# Run ONCE on the login node: pre-download open-weight model weights into the scratch HF
# cache, so compute jobs (which run HF_HUB_OFFLINE=1) read from cache without network.
# Needs `huggingface-cli login` (gated models: Llama, Gemma) accepted on your HF account.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh
export HF_HUB_OFFLINE=0
activate "$CONDA_IQE"

# Open-weight roster (see cluster/models_open.txt). Gated models (BharatGPT, tiny-aya)
# need their HF licence accepted on your account first.
mapfile -t MODELS < <(grep -vE '^\s*#|^\s*$' cluster/models_open.txt | awk '{print $1}')
# dedicated Indic-MT specialist for the retranslate/quality track (4B gemma3, GPL-3.0, ungated)
MODELS+=( sarvamai/sarvam-translate )
# COMET encoder backbones — REQUIRED for the trained metrics to load OFFLINE. download_model()
# fetches the checkpoint but the encoder (tokenizer/vocab) is a separate HF repo; without these in
# cache the metric dies with "Unable to load vocabulary from file". cometkiwi-da uses InfoXLM-large;
# cometkiwi-da-xl + XCOMET-XL use XLM-R-XL. (Both ungated.)
MODELS+=( microsoft/infoxlm-large facebook/xlm-roberta-xl )
# add the ALOPE backbone(s) you intend to fine-tune
MODELS+=( meta-llama/Llama-3.2-3B-Instruct "${ALOPE_BACKBONE:-CoRover/BharatGPT-3B-Indic}" )
for M in "${MODELS[@]}"; do
  echo ">> caching $M"
  huggingface-cli download "$M" --quiet || hf download "$M" || true
done
# ---- trained QE metrics (gated cc-by-nc-sa: accept the 3 Unbabel licences on your HF account first)
# Resolve + SAVE local checkpoint paths so the offline compute job loads them directly
# (download_model() can bypass the cache under HF_HUB_OFFLINE=1).
echo ">> caching COMET metrics + writing $IQE_CACHE/comet_paths.json"
mkdir -p "$IQE_CACHE"
python - "$IQE_CACHE/comet_paths.json" <<'PY' || echo "   [comet] prefetch failed — accept the 3 Unbabel licences + huggingface-cli login, then re-run"
import sys, json
from comet import download_model
out = sys.argv[1]; paths = {}
for m in ("Unbabel/wmt22-cometkiwi-da", "Unbabel/wmt23-cometkiwi-da-xl", "Unbabel/XCOMET-XL"):
    try:
        p = download_model(m); paths[m] = p; print("   cached", m, "->", p)
    except Exception as e:
        print("   FAILED", m, e)
json.dump(paths, open(out, "w")); print("   wrote", out, "(%d models)" % len(paths))
PY

echo "Models + metrics cached under $HF_HOME. Compute jobs can now run offline."
echo "NOTE: command-a-plus (126B w4a4) + Axion-24B (GGUF) are heavy/special — fetch separately if used."
