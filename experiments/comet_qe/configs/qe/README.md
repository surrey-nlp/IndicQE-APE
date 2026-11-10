# QE experiment configs

One YAML per run, named `qe-<arm>-<variant>[-s<seed>].yaml`. Every config except
`qe-smoke-tiny.yaml` is written by `scripts/gen_qe_configs.py` from a single reference recipe
(`qe-A-cometkiwi-full`) with exactly one field changed, so edit the generator rather than the
YAML. The arms are described in `../../README.md`.

- `qe-smoke-tiny.yaml` — offline end-to-end smoke (tiny random encoder, 600-row slice). Build the
  inputs first: `python scripts/make_tiny_encoder.py` and
  `python scripts/build_qe_data.py --variant enindic --per-pair 40`, then
  `python src/train.py --config configs/qe/qe-smoke-tiny.yaml`.
- `python scripts/gen_qe_configs.py --check` parse-validates every generated config without
  writing.

The original BEA reference configs are in `../templates/`.
