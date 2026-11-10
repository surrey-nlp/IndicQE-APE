# tiny-aya-fire quality estimation on IndicQE-APE

The aggregate evaluation outputs of the layer-probe experiment (ALOPE + `CohereLabs/tiny-aya-fire`,
§4 and Tables 24–25 of the paper), read by `analysis/make_alope_table.py` and
`analysis/probe_vs_prompt.py`. This folder holds only the summary CSVs; the training and inference
scripts listed below belong to the experiment's own codebase
([ALOPE](https://github.com/surrey-nlp/ALOPE)) and are not included here.

The model is trained jointly on the six English-to-Indic pairs of the
IndicQE-APE `all` split (Gujarati, Hindi, Malayalam, Marathi, Tamil, Telugu) and
evaluated on the matched `challenge` split.

Results are included for layers -5, -7, -11, -16, -20 and -24.

## Layout

Data preparation, training and inference:

- `setup_data_aya_indic.py` — downloads IndicQE-APE and writes the train and
  challenge TSVs.
- `single-layer-regression_train_aya_indic.py` — trains the probe for one layer
  on all six pairs.
- `single-layer-regression_inference_aya_indic.py` — runs inference for one
  layer and one language pair.

Evaluation:

- `evaluate.py` — aggregate metrics over a whole test file.
- `evaluate_challenge.py` — per-route metrics and the drop table. Marks results
  whose `easy_control` baseline failed its sanity check.
- `diagnose_challenge_floor_aya_indic.py` — diagnostics for a route that looks
  wrong: ground-truth spread, prediction spread, prompt lengths, the length
  correlations, and length-controlled correlations.

Queue scripts, which run the above sequentially on a server:

- `run_queue_aya.sh` — training, one run per layer.
- `infer_queue_aya_indic.sh` — inference, every layer crossed with every pair.
- `eval_queue_aya_indic.sh` — evaluation, one run per pair, then a combined
  summary.

Data and results:

- `data/indicqe-ape/` — the challenge TSVs the results were produced against.
- `results/tiny-aya-fire_indic_challenge_layer_*/` — per-pair inference output.
- `results/evaluation_challenge_summary_*.csv` — per-route summaries and drop
  tables.

## A note on the test files

The challenge TSVs are included in this repository rather than being treated as
a regenerable input. Prediction files are joined to routes by row position.

## NOTE en-ml negative correlation with easy_control

The en-ml control route correlates negatively, which might be related to how DA scores correlate
with segment length; `analysis/make_alope_table.py` therefore excludes en-ml from the drop analysis.
