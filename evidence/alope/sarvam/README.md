# sarvam-m: predecessor experiment

The earlier generation of this quality-estimation work, run on `sarvamai/sarvam-m`
before the pipeline moved to `CohereLabs/tiny-aya-fire`. It is here for context —
the read position (last token of the raw prompt, no chat template) and the layer
choice used in the main experiment both come out of these runs.

Everything below is derived from the prediction files in `results/`; you do not
need to run anything. `sarvam-m_run_summary.csv` holds the same table.

## What the runs show

All runs are en-hi. `gt_std` is the spread of the human scores, `pred_std` the
spread of the model's predictions — the gap between them is the main story.

| run_tag | layer | n | gt_std | pred_std | pearson | spearman | mae |
|---|---|---|---|---|---|---|---|
| (none) | -5 | 1842 | 14.72 | 2.42 | -0.073 | -0.067 | 11.24 |
| (none) | -11 | 1842 | 14.72 | 1.53 | 0.121 | 0.069 | 11.35 |
| (none) | -12 | 1842 | 14.72 | 1.89 | 0.060 | 0.025 | 11.51 |
| (none) | -16 | 1842 | 14.72 | 1.24 | 0.143 | 0.082 | 11.35 |
| (none) | -20 | 1842 | 14.72 | 0.36 | 0.046 | 0.013 | 10.56 |
| (none) | -24 | 1842 | 14.72 | 3.29 | 0.394 | 0.330 | 10.36 |
| (none) | -28 | 1842 | 14.72 | 0.21 | 0.230 | 0.129 | 11.19 |
| (none) | -32 | 1842 | 14.72 | 3.14 | 0.345 | 0.273 | 11.14 |
| (none) | -40 | 1842 | 14.72 | 3.73 | 0.335 | 0.208 | 10.72 |
| enhi_claude | -5 | 1842 | 14.72 | 1.87 | 0.142 | 0.077 | 12.58 |
| **enhi_lnorm** | **-20** | **1842** | 14.72 | **6.85** | **0.560** | **0.539** | **9.67** |
| enhi_only | -20 | 1842 | 14.72 | 0.36 | 0.046 | 0.013 | 10.56 |
| layer_norm | -20 | 1842 | 14.72 | 1.91 | 0.158 | 0.081 | 12.46 |
| full | -5 | 401 | 10.38 | 5.15 | 0.134 | 0.305 | 7.54 |
| multilingual_rank | -5 | 401 | 10.38 | 2.30 | 0.252 | 0.277 | 7.55 |

Four things to take from it:

**Most of these runs collapsed to predicting the mean.** Human scores vary with
a standard deviation of 14.7; most runs predict within a band of 1–4 points, and
two of them (layers -20 and -28) within 0.2–0.4 points. A model emitting an
almost constant value cannot be tracking quality, and the near-zero correlations
follow from that. These are not weak results, they are non-results.

**One configuration broke out of it.** `enhi_lnorm` at layer -20 predicts with a
spread of 6.85 and reaches pearson 0.560 / spearman 0.539, with the lowest error
in the set. It is the only run whose predictions vary enough to be meaningful.
Note that `layer_norm` at the same layer reached only 0.158 — two attempts at the
same idea with very different outcomes, so the difference is in the details of
those two scripts rather than in the idea itself.

**Two runs are on a different test set and are not comparable.** `full` and
`multilingual_rank` have n=401 against everyone else's n=1842, and their ground
truth has a different distribution (mean 80.4, sd 10.4 versus 73.2 and 14.7).
Their much lower MAE (~7.5 against ~11) reflects that narrower target range, not
a better model. Any comparison across those two groups is meaningless.

**Two directories hold identical predictions.** `sarvam-m_layer_-20` and
`sarvam-m_enhi_only_layer_-20` are byte-identical files under different names.
Count them as one run.

**On layer choice**, reading the untagged sweep alone: the shallow layers (-5 to
-16) are poor, and the best untagged results sit at -24, -32 and -40 (0.33–0.39).
This is the evidence behind the layer range used in the main experiment.

## Which script produced which run

Run directories are named `sarvam-m{_RUN_TAG}_layer_{LAYER}`, and `RUN_TAG` lives
in the config file rather than in the training script, so this mapping is inferred
from the naming convention:

| results directory | training script |
|---|---|
| `sarvam-m_layer_-5` … `-40` | `single-layer-regression_train_sarvam.py` |
| `sarvam-m_full_layer_-5` | `..._train_sarvam_full.py` |
| `sarvam-m_layer_norm_layer_-20` | `..._train_sarvam_layer_norm.py` |
| `sarvam-m_multilingual_rank_layer_-5` | `..._train_sarvam_multilingual_rank.py` |
| `sarvam-m_enhi_lnorm_layer_-20` | `train_sarvam_enhi_lnorm_-20.py` |
| `sarvam-m_enhi_only_layer_-20` | TO CONFIRM (duplicate of the untagged -20 run) |
| `sarvam-m_enhi_claude_layer_-5` | TO CONFIRM |

The variants `_gradient`, `_loss` and `_norm` have no results directory here.

## Also in this folder

`frozen_feature_probe.py` fits ridge regression on frozen pooled hidden states
with no LoRA and no trained head — a floor showing how much signal is linearly
available before any training. The five `*_diagnostic/` directories hold
`en-hi_train_diagnostic.csv` from the layer-scan script rather than predictions,
and are in a different format from the tables above; check their provenance
before using them, since an earlier version of that script read the wrong token
position.

Do not compare any number here with the tiny-aya-fire results in `../results/` —
different model, different dataset, different test files.
