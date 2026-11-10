#!/usr/bin/env python3
"""gen_qe_configs.py — emit the CORE comet-QE ablation configs from one reference.

Design: "one variable at a time from a strong reference" (the arms are described in README.md).
Every hybrid-model run (reference A-full, A0-min, E1 encoder set, E2 features, E3 funnelling,
E4 loss, E5 target scale) is the reference config with exactly one field changed, so the configs
are a generated artefact — never hand-edited. B (native CometKiwi FT), C (COMET-features+MLP) and
Z (zero-shot) are separate pipelines and are NOT emitted here.

    python scripts/gen_qe_configs.py            # write all CORE configs to configs/qe/
    python scripts/gen_qe_configs.py --check    # parse-validate every emitted config, no write
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import yaml

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "qe"

# ── Reference config (Strategy A-full) ────────────────────────────────────────
REFERENCE = {
    "task": "qe",
    "model_name": "models/cometkiwi-infoxlm",
    "data_path": "data/qe_indic_enindic.csv",
    "report_to": "none",
    # QE data schema
    "split_column": "subset", "dev_split": "dev", "test_split": "test",
    "item_id_column": "item_id", "label_column": "target",
    "group_column": "language_pair",
    "source_column": "source", "mt_column": "mt", "segment_order": "src_first",
    # architecture
    "use_hybrid_model": True, "deep_mlp": True,
    "multi_layer_funneling": True, "num_funneling_layers": 4,   # -> [12,16,20,-1] on 24L
    # feature recipe (A-full)
    "use_attention": True,
    "use_target_source_alignment": True,   # TSA
    "use_target_source_entropy": True,     # TSE
    "use_length_ratio": True,
    "use_lp": False,                       # A4 (joint-only) is EXT, off by default
    # loss + selection (QE convention)
    "loss_function": "huber", "loss_kwargs": {"delta": 1.0},
    "metric_for_best_model": "pearson", "greater_is_better": True,
    # target scale: per-pair z is handled in qe_data; raw/minmax are E5 variants
    "target_scale": "perpair_z",
    # training. batch 8 + accum 2 (eff. 16) and max_length 256 keep the 565M encoder with
    # output_attentions (manual-attention path, memory-heavy) inside a 24 GB A5000; 512/16 OOMs.
    "learning_rate": 3e-5, "train_batch_size": 8, "eval_batch_size": 16,
    "gradient_accumulation_steps": 2, "num_train_epochs": 4,
    "max_length": 256, "warmup_steps": 100, "weight_decay": 0.01,
    "logging_steps": 50, "save_total_limit": 1, "fp16": True, "seed": 42,
}

# A0 minimal = reference with the feature recipe stripped to CLS+funnelling only
A0_OFF = {"use_target_source_alignment": False, "use_target_source_entropy": False,
          "use_length_ratio": False, "use_lp": False}


def a0(**over):
    c = copy.deepcopy(REFERENCE); c.update(A0_OFF); c.update(over); return c


def ref(**over):
    c = copy.deepcopy(REFERENCE); c.update(over); return c


# ── The CORE grid ─────────────────────────────────────────────────────────────
CONFIGS = {
    # reference + minimal
    "qe-A-cometkiwi-full": ref(),
    "qe-A0-cometkiwi-min": a0(),
    # E1 — encoder init (A0 recipe, swap backbone)
    "qe-E1-xlmr":      a0(model_name="FacebookAI/xlm-roberta-large"),
    "qe-E1-infoxlm":   a0(model_name="microsoft/infoxlm-large"),
    "qe-E1-cometkiwi": a0(model_name="models/cometkiwi-infoxlm"),   # == A0-min, the star
    "qe-E1-cometda":   a0(model_name="models/comet-xlmr"),
    # E2 — feature recipe (additive on cometkiwi)
    "qe-E2-A0": a0(),
    "qe-E2-A1": a0(use_target_source_alignment=True),
    "qe-E2-A2": a0(use_target_source_alignment=True, use_target_source_entropy=True),
    "qe-E2-A3": a0(use_target_source_alignment=True, use_target_source_entropy=True,
                   use_length_ratio=True),
    "qe-E2-Afull": ref(),
    # E3 — funnelling layers
    "qe-E3-none":   ref(multi_layer_funneling=False),
    "qe-E3-auto4":  ref(funneling_layers=[12, 16, 20, -1]),
    "qe-E3-upper":  ref(funneling_layers=[16, 20, -1]),
    "qe-E3-buggy":  ref(funneling_layers=[8, 10, -1]),
    # E4 — loss
    "qe-E4-huber1":  ref(loss_function="huber", loss_kwargs={"delta": 1.0}),
    "qe-E4-huber05": ref(loss_function="huber", loss_kwargs={"delta": 0.5}),
    "qe-E4-mse":     ref(loss_function="mse", loss_kwargs={}),
    "qe-E4-ccc":     ref(loss_function="ccc", loss_kwargs={}),
    # E5 — target scale (build-time: build_qe_data.py --target-scale writes the scale-suffixed CSV)
    "qe-E5-z":      ref(target_scale="perpair_z", data_path="data/qe_indic_enindic.csv"),
    "qe-E5-raw":    ref(target_scale="raw", data_path="data/qe_indic_enindic_raw.csv"),
    "qe-E5-minmax": ref(target_scale="minmax", data_path="data/qe_indic_enindic_minmax.csv"),
}

# ── Multi-seed reruns of the four single-seed collapses ───────────────────────
# E1-infoxlm (-0.001), E2-A1 (0.085), E3-none (0.097) and E4-huber05 (0.089) each collapsed to
# near-mean prediction at seed 42 while their neighbours in the same sweep trained normally. One
# seed cannot separate an unhelpful setting from an unlucky initialisation, so each is repeated at
# two further seeds. A cell is reported as a collapse only where the collapse reproduces.
COLLAPSED = {
    "qe-E1-infoxlm": a0(model_name="microsoft/infoxlm-large"),
    "qe-E2-A1": a0(use_target_source_alignment=True),
    "qe-E3-none": ref(multi_layer_funneling=False),
    "qe-E4-huber05": ref(loss_function="huber", loss_kwargs={"delta": 0.5}),
}
SEEDS = {f"{name}-s{seed}": {**copy.deepcopy(cfg), "seed": seed}
         for name, cfg in COLLAPSED.items() for seed in (43, 44)}

# ── Disentangling the two confounded cells ────────────────────────────────────
# qe-E2-A1 and qe-E4-huber05 collapsed at seed 42 and recovered at seeds 43/44, but the two runs
# straddled a rebuild of qe_indic_enindic.csv: seed 42 trained on 64,416 segments, seeds 43/44 on
# 64,671. Seed and data therefore changed together and the recovery cannot be attributed to
# either. These two runs repeat seed 42 on the CURRENT build, holding the seed fixed and changing
# only the data. If they still collapse, the cause is the seed. If they do not, the original
# collapse was a property of the superseded data. E1-infoxlm needs no such run: its three seeds
# all used one build already, and E3-none reproduced across both.
CONFOUNDED = {
    "qe-E2-A1-s42new": {**copy.deepcopy(COLLAPSED["qe-E2-A1"]), "seed": 42},
    "qe-E4-huber05-s42new": {**copy.deepcopy(COLLAPSED["qe-E4-huber05"]), "seed": 42},
}

# ── Six-pair and X->en arms ───────────────────────────────────────────────────
# The campaign ran on an `enindic` frame built from dataset/release_build/hf_repo, which predates
# both the addition of en-ml to qe-da and the challenge-test extension. Those runs therefore have
# five pairs and an 8,160-row test set, against 8,294 on the current release. JOINT re-runs the
# reference recipe on the six-pair frame; XEN is the X->en reproduction check against published
# CometKiwi numbers, the only external validity check available for this pipeline.
JOINT = {
    "qe-J-ref-joint":  ref(data_path="data/qe_indic_joint_enml.csv"),
    "qe-J-A0-joint":   a0(data_path="data/qe_indic_joint_enml.csv"),
    "qe-X-ref-xen":    ref(data_path="data/qe_indic_xen.csv"),
    "qe-X-A0-xen":     a0(data_path="data/qe_indic_xen.csv"),
}

# EXT (documented, emitted so they're ready but flagged in the header comment)
EXT = {
    "qe-E1-cometkiwi-xl": a0(model_name="models/cometkiwi-xl-enc", fp16=True,
                             train_batch_size=8, gradient_accumulation_steps=2),
    "qe-E6-lora":  ref(use_lora=True, lora_r=16, lora_target_modules=["query", "value"]),
    "qe-E7-mtfirst": ref(segment_order="mt_first"),
    "qe-A4-lp-joint": ref(use_lp=True, data_path="data/qe_indic_joint_enml.csv"),
}

HEADER = ("# GENERATED by scripts/gen_qe_configs.py — do not hand-edit.\n"
          "# One-variable-at-a-time from qe-A-cometkiwi-full. Arms: experiments/comet_qe/README.md.\n")

REQUIRED = {"task", "model_name", "data_path", "use_hybrid_model", "loss_function",
            "metric_for_best_model", "label_column", "source_column", "mt_column"}


def write_all():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for tier, grid in (("CORE", CONFIGS), ("SEED", SEEDS), ("CONFOUNDED", CONFOUNDED),
                       ("JOINT", JOINT), ("EXT", EXT)):
        for name, cfg in grid.items():
            cfg = {"experiment_name": name, **cfg}
            body = HEADER + f"# tier: {tier}\n" + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
            (CFG_DIR / f"{name}.yaml").write_text(body)
            n += 1
    print(f"wrote {n} configs to {CFG_DIR}")


def check_all():
    ok = True
    for f in sorted(CFG_DIR.glob("qe-*.yaml")):
        try:
            c = yaml.safe_load(f.read_text())
        except Exception as e:
            print(f"FAIL parse {f.name}: {e}"); ok = False; continue
        miss = REQUIRED - set(c)
        if miss:
            print(f"FAIL keys {f.name}: missing {miss}"); ok = False
        elif c["task"] != "qe":
            print(f"FAIL {f.name}: task != qe"); ok = False
        else:
            print(f"ok   {f.name}  ({c['model_name']}, loss={c['loss_function']})")
    print("ALL PARSE + REQUIRED-KEYS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check_all())
    write_all()
    sys.exit(check_all())
