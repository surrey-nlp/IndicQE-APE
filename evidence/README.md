# evidence/

Read-only aggregate inputs that the analysis scripts read and the paper cites, copied from the
experiments that produced them. Do not edit anything here by hand: if a number changes, the fix
belongs in the experiment that produced it.

| subdir | what it holds | read by |
|---|---|---|
| `alope/` | per-pair, per-layer challenge-split summaries of the tiny-aya-fire layer probe (ALOPE), §4 and Tables 24–25 | `analysis/make_alope_table.py`, `analysis/probe_vs_prompt.py` |
| `alope/sarvam/` | the predecessor probe on sarvam-m, kept for context (read position and layer choice) | — |

Per-instance model outputs are not here: they live under `results/` and
`experiments/comet_qe/results/`. `analysis/export_evidence_bundle.py` packs the aggregate inputs and
those outputs into a single archive for archiving alongside the paper.
