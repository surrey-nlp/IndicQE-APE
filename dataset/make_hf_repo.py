#!/usr/bin/env python3
"""Assemble the push-ready HuggingFace repo for surrey-nlp/IndicQE-APE.

FULL PUBLIC RELEASE (PI decision, 16 Jul 2026 — no more gating): every instance, every
pair (incl. En-Ml), every task (incl. word-QE + explainable-QE) is public. En-Ml is a
first-class curated pair; the challenge/diagnostic test is EXTENDED to include En-Ml's
difficulty routes. When paper 2602.08600 is accepted the data repo will simply point at
this page.

Inputs: curated splits (train/dev/test) from curate_splits.py (carry every unified column
+ split/route/difficulty). Domains are finalised here (normalise gold + classifier-infer
the unlabelled + provenance), so every released row carries a domain.

Layout (under --out-dir):
  README.md                          card + YAML `configs:`
  data/all/{train,validation,test}   curated En->Indic (incl En-Ml)  [default config]
  data/full/{...}                    everything incl X->En MLQE-PE passthrough
  data/tasks/qe-da/{...}             rows with DA
  data/tasks/ape/{...}               rows with a human post-edit (incl En-Ml)
  data/tasks/word-qe/{...}           rows with word-level OK/BAD tags
  data/tasks/explainable-qe/{...}    rows with human remarks (TQR) or synthetic descriptions
  data/pairs/<pair>/{...}            per language pair (9, incl en-ml)
  data/challenge/{test,test_blind}   extended diagnostic test (incl En-Ml routes)
  load_test.py, PUSH.md

Usage:
  python3 make_hf_repo.py --curated release_build/curated --out-dir release_build/hf_repo
"""
import argparse, json, os
import pandas as pd
import normalize_domain as nd

ALL_PAIRS = ["en-hi", "en-mr", "en-ta", "en-ml", "en-te", "en-gu", "et-en", "ne-en", "si-en"]
CURATED = {"en-hi", "en-mr", "en-ta", "en-ml", "en-te", "en-gu"}   # Surrey-curated En->Indic
# clean rich release schema (drops internal scratch: src_key, is_public, _config, bad_density,
# hard_error, raw axis booleans, source_dataset, orig_split, gated_rl)
RELEASE_COLS = ["uid", "cadence_id", "sources", "original_id",
                "language_pair", "src_lang", "tgt_lang",
                "domain", "domain_raw", "domain_provenance", "domain_confidence",
                "source", "mt", "post_edit", "reference",
                "da_scores", "da_mean", "da_z_scores", "da_z_mean", "n_annotators",
                "da_conflict", "da_std", "human_ter",
                "route", "difficulty", "n_axes", "is_curated",
                "wtag", "mt_tokens", "tqr", "synthetic_description",
                "error_categories", "explanation_model", "mqm_spans", "mqm_score",
                "has_da", "has_pe", "has_domain", "has_wtag", "has_tqr", "has_synth_expl", "has_mqm"]
SPLIT_FILE = {"train": "train", "dev": "validation", "test": "test"}


def select(df):
    cols = [c for c in RELEASE_COLS if c in df.columns]
    return df[cols + ["split"]]


def load_curated(curated):
    parts = []
    for split in ["train", "dev", "test"]:
        d = pd.read_parquet(os.path.join(curated, f"{split}.parquet"))
        d["split"] = split
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def write_splits(df, dest):
    os.makedirs(dest, exist_ok=True)
    for split in ["train", "dev", "test"]:
        sub = df[df["split"] == split]
        if len(sub):
            sub.drop(columns=["split"], errors="ignore").reset_index(drop=True).to_parquet(
                os.path.join(dest, f"{SPLIT_FILE[split]}.parquet"), index=False)


YAML = """---
license: cc-by-nc-4.0
task_categories: [translation, text-classification]
task_ids: [text-scoring]
language: [en, hi, mr, ta, ml, te, gu, et, ne, si]
multilinguality: [multilingual]
annotations_creators: [expert-generated, machine-generated]
size_categories: [100K<n<1M]
pretty_name: IndicQE-APE
tags: [quality-estimation, automatic-post-editing, machine-translation, low-resource, indic, dravidian, akshara, challenge-set, direct-assessment]
configs:
{cfg}---
"""


def cfg_block():
    def blk(name, path, default=False):
        s = f"  - config_name: {name}\n"
        if default:
            s += "    default: true\n"
        s += "    data_files:\n"
        for split, fn in [("train", "train"), ("validation", "validation"), ("test", "test")]:
            s += f"      - split: {split}\n        path: {path}/{fn}.parquet\n"
        return s
    def blk_test(name, path):
        return (f"  - config_name: {name}\n    data_files:\n"
                f"      - split: test\n        path: {path}/test.parquet\n")
    out = [blk("all", "data/all", default=True), blk("full", "data/full"),
           blk("qe-da", "data/tasks/qe-da"), blk("ape", "data/tasks/ape"),
           blk("word-qe", "data/tasks/word-qe"), blk("explainable-qe", "data/tasks/explainable-qe")]
    for p in ALL_PAIRS:
        out.append(blk(p, f"data/pairs/{p}"))
    out.append(blk_test("challenge", "data/challenge"))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-name", default="surrey-nlp/IndicQE-APE")
    args = ap.parse_args()
    out = args.out_dir

    df = load_curated(args.curated)
    # finalise domains for the WHOLE release: normalise gold + classifier-infer unlabelled +
    # provenance (annotated/inferred/fallback). Guarantees every released row carries a domain.
    # If the curated splits were already finalised (finalize_domains_step.py), skip the slow refit.
    if "domain_provenance" in df.columns:
        dom_rep = nd.DomainReport(n_rows=len(df),
                                  provenance_counts=df.domain_provenance.value_counts().to_dict())
        print("domains already finalised in curated input — skipping classifier refit")
    else:
        df, dom_rep = nd.finalize_domains(df, keep_raw=True, eval_holdout=True)
    df["is_curated"] = df["language_pair"].isin(CURATED)
    # route is a diagnostic-TEST concept; non-test pool rows carry '' -> null it for a clean schema
    if "route" in df.columns:
        df.loc[df["route"].astype(str) == "", "route"] = None

    rel = select(df)
    cur = rel[rel.is_curated]

    # ---- task + slice configs (all public) ----
    write_splits(cur, os.path.join(out, "data", "all"))
    write_splits(rel, os.path.join(out, "data", "full"))
    write_splits(cur[cur.has_da], os.path.join(out, "data", "tasks", "qe-da"))
    # ape = every row with a human post-edit (curated En->Indic + X->En MLQE-PE)
    write_splits(rel[rel.has_pe], os.path.join(out, "data", "tasks", "ape"))
    write_splits(rel[rel.has_wtag.fillna(False)], os.path.join(out, "data", "tasks", "word-qe"))
    write_splits(rel[rel.has_tqr.fillna(False) | rel.has_synth_expl.fillna(False)],
                 os.path.join(out, "data", "tasks", "explainable-qe"))
    for pair in ALL_PAIRS:
        write_splits(rel[rel.language_pair == pair], os.path.join(out, "data", "pairs", pair))

    # ---- extended challenge = every curated diagnostic-test row (incl En-Ml) + X->En test ----
    os.makedirs(os.path.join(out, "data", "challenge"), exist_ok=True)
    ctest = rel[rel.split == "test"].drop(columns=["split"]).reset_index(drop=True)
    ctest.to_parquet(os.path.join(out, "data", "challenge", "test.parquet"), index=False)
    ctest[["uid", "cadence_id", "language_pair", "source", "mt", "domain", "route"]].to_parquet(
        os.path.join(out, "data", "challenge", "test_blind.parquet"), index=False)

    # ---- README (YAML configs + coverage) ----
    body_f = os.path.join(os.path.dirname(__file__), "templates", "README_body.md")
    body = open(body_f).read() if os.path.exists(body_f) else "\n# IndicQE-APE\n"
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(YAML.format(cfg=cfg_block()))
        fh.write("\n" + body)

    # ---- coverage ----
    cov = rel.groupby("language_pair").agg(
        n=("uid", "size"), train=("split", lambda s: (s == "train").sum()),
        dev=("split", lambda s: (s == "dev").sum()), test=("split", lambda s: (s == "test").sum()),
        has_da=("has_da", "sum"), has_pe=("has_pe", "sum"), has_wtag=("has_wtag", "sum"),
        has_tqr=("has_tqr", "sum")).astype(int)
    cov.to_csv(os.path.join(out, "public_coverage.csv"))
    open(os.path.join(out, "domain_report.json"), "w").write(dom_rep.to_json())
    print("TOTAL released rows:", len(rel))
    print(cov.to_string())
    print("\ndomain provenance:", df.domain_provenance.value_counts().to_dict())
    print("domain classes:", df.domain.value_counts().to_dict())
    print(f"domain classifier holdout: acc={dom_rep.holdout_accuracy} macroF1={dom_rep.holdout_macro_f1}")
    print(f"\nrepo assembled at {out}")


if __name__ == "__main__":
    main()
