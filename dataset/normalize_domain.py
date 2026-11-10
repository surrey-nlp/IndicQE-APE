"""Domain-label normalisation + lightweight inference for IndicQE-APE.

Two jobs, decided with DK (4 Jul):
  1. NORMALISE the redundant raw labels (two source lineages tagged differently — lowercase
     from the domain-QE repos, capitalised from ape-combined) into a 6-way canonical taxonomy.
  2. INFER a domain for every unlabelled row (all of X->En + some En->X: ~33.8k rows) with a
     simple TF-IDF + logistic-regression classifier on the *English side* of each pair
     (the source for En->X, the MT for X->En), so any sample from any subset carries a domain.

Design (all locked with DK):
  * Canonical taxonomy: health, legal, general, tourism, education, other.
  * `domain` is normalised AND filled for every inferrable row.
  * Provenance is explicit: `domain_provenance` in {annotated, inferred, none} and
    `domain_confidence` (NaN for annotated; classifier max-probability for inferred).
  * `has_domain` KEEPS its old meaning — gold-annotated only (== provenance == "annotated") —
    so existing filters that select human-labelled rows are unchanged.
  * `domain_raw` preserves the original label for audit.

Classifier: word⊕character TF-IDF union → Logistic Regression (max settings; see build_pipeline).
Validated on the released `full` config (holdout): accuracy ~0.837, macro-F1 ~0.803
(health 0.91, tourism 0.84, education 0.84, legal 0.82, general 0.79; `other` ~0.61 is the weak catch-all).
Every instance is guaranteed a domain: gold → normalised; unlabelled → inferred; any residual
(empty English side) → 'other' with provenance 'fallback'. No subset of the release has a NULL domain.
Because X->En rows have no gold domain to check and their English side is MT output, their
inferred labels carry more risk — hence provenance is marked, never silently merged.

Usage
-----
Integration (in build_unified.py, after `has_domain` is set):
    from normalize_domain import finalize_domains
    out, dom_report = finalize_domains(out)

Standalone on a parquet snapshot:
    python dataset/normalize_domain.py in.parquet out.parquet --report domain_report.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- taxonomy
CANONICAL_DOMAINS = ("health", "legal", "general", "tourism", "education",
                     "news", "finance", "tech", "sports", "other")

# raw label -> canonical. Extend here if new source lineages appear.
# News/finance/tech/sports promoted to first-class domains (En-Ml 10K, 16 Jul 2026): the
# Anuvaad-sourced En-Ml pair carries these as gold labels. `General/News` (domain-QE compound
# lineage) stays `general`; a bare `news` label now maps to the first-class `news`.
RAW_TO_CANON = {
    "health": "health", "Health": "health",
    "legal": "legal", "Legal": "legal",
    "general": "general", "General": "general", "General/News": "general",
    "tourism": "tourism", "Tourism": "tourism", "Tourism/Culture": "tourism",
    "education": "education", "Education": "education",
    "news": "news", "News": "news",
    "finance": "finance", "Finance": "finance", "financial": "finance",
    "tech": "tech", "Tech": "tech", "technology": "tech", "Technology": "tech",
    "sports": "sports", "Sports": "sports", "sport": "sports",
    "other": "other", "Other": "other",
}


def _norm_one(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return None
    if s in RAW_TO_CANON:
        return RAW_TO_CANON[s]
    # case-insensitive fallback (unseen casing/whitespace variants)
    return RAW_TO_CANON.get(s.lower(), None) or (s.lower() if s.lower() in CANONICAL_DOMAINS else "other")


def english_side(df: pd.DataFrame) -> pd.Series:
    """The English text of each pair: source when src_lang=='en', else the MT (English target)."""
    src = df.get("src_lang")
    is_en_src = (src == "en") if src is not None else pd.Series(True, index=df.index)
    return np.where(is_en_src, df["source"].astype(str), df["mt"].astype(str))


# --------------------------------------------------------------------------- classifier
def build_pipeline():
    """Max-performance production classifier: word + character TF-IDF union → Logistic Regression.

    The word⊕char FeatureUnion is the biggest lever (CV macro-F1 +~2–3 pts over word-only): char
    n-grams capture morphology, named entities and transliteration that word tokens miss. LR is set
    for best attainable fit — C=8 (light regularisation, the domains are lexically separable),
    max_iter=2000 (full convergence), class_weight='balanced' (the `other`/`education` tail is small).
    Full-set holdout: accuracy ~0.837, macro-F1 ~0.803 (vs 0.821/0.780 for the earlier word-only 50k/C4).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           max_features=50_000, sublinear_tf=True)
    return Pipeline([
        ("feats", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", n_jobs=-1)),
    ])


@dataclass
class DomainReport:
    n_rows: int = 0
    canonical: tuple = CANONICAL_DOMAINS
    raw_value_counts: dict = field(default_factory=dict)
    provenance_counts: dict = field(default_factory=dict)
    holdout_accuracy: float | None = None
    holdout_macro_f1: float | None = None
    per_class_f1: dict = field(default_factory=dict)
    inferred_conf_mean: float | None = None
    filled_by_pair: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


def _holdout_metrics(pipe, X, y) -> tuple[float, float, dict]:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    per_class = {k: round(v["f1-score"], 3) for k, v in rep.items() if k in CANONICAL_DOMAINS}
    return (round(accuracy_score(yte, pred), 3),
            round(f1_score(yte, pred, average="macro", zero_division=0), 3), per_class)


# --------------------------------------------------------------------------- main entry
def finalize_domains(df: pd.DataFrame, *, keep_raw: bool = True,
                     eval_holdout: bool = True) -> tuple[pd.DataFrame, DomainReport]:
    """Normalise + infer domains. Returns (df_with_columns, report).

    Adds/overwrites: domain (canonical, filled), domain_provenance, domain_confidence,
    domain_raw (optional), and redefines has_domain = (provenance == 'annotated').
    """
    df = df.copy()
    rep = DomainReport(n_rows=len(df),
                       raw_value_counts=df["domain"].value_counts(dropna=False).to_dict()
                       if "domain" in df.columns else {})

    raw = df["domain"] if "domain" in df.columns else pd.Series([None] * len(df), index=df.index)
    if keep_raw:
        df["domain_raw"] = raw
    canon = raw.map(_norm_one)

    eng = pd.Series(english_side(df), index=df.index).fillna("").astype(str)
    labelled = canon.notna()
    to_infer = (~labelled) & (eng.str.len() > 0)

    provenance = pd.Series(np.where(labelled, "annotated", "none"), index=df.index)
    confidence = pd.Series(np.nan, index=df.index, dtype="float64")
    domain = canon.copy()

    if to_infer.any():
        pipe = build_pipeline()
        X_tr, y_tr = eng[labelled].values, canon[labelled].values
        if eval_holdout and len(set(y_tr)) > 1:
            rep.holdout_accuracy, rep.holdout_macro_f1, rep.per_class_f1 = _holdout_metrics(pipe, X_tr, y_tr)
        pipe.fit(X_tr, y_tr)  # refit on all labelled data for production predictions
        Xi = eng[to_infer].values
        proba = pipe.predict_proba(Xi)
        classes = pipe.named_steps["clf"].classes_
        pred = classes[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        domain.loc[to_infer] = pred
        confidence.loc[to_infer] = conf
        provenance.loc[to_infer] = "inferred"
        rep.inferred_conf_mean = round(float(conf.mean()), 3)

    # ---- coverage guarantee: EVERY instance must carry a domain ----
    # Any row still unset (neither gold nor inferrable — e.g. empty English side) falls back to
    # 'other' with provenance 'fallback', so no NULL domain reaches any subset of the release.
    residual = domain.isna()
    if residual.any():
        domain.loc[residual] = "other"
        provenance.loc[residual] = "fallback"

    df["domain"] = domain
    df["domain_provenance"] = provenance
    df["domain_confidence"] = confidence
    df["has_domain"] = provenance.values == "annotated"   # unchanged gold-only semantics

    assert df["domain"].notna().all(), "domain coverage gap — some rows left unlabelled"
    assert set(df["domain"].unique()) <= set(CANONICAL_DOMAINS), "non-canonical domain value slipped in"

    rep.provenance_counts = provenance.value_counts().to_dict()
    if "language_pair" in df.columns:
        rep.filled_by_pair = (df.assign(filled=(provenance == "inferred").values)
                              .groupby("language_pair")["filled"].sum().astype(int).to_dict())
    return df, rep


def _report_md(rep: DomainReport) -> str:
    lines = ["# IndicQE-APE domain normalisation + inference report", "",
             f"- Rows: **{rep.n_rows:,}**",
             f"- Canonical taxonomy: {', '.join(rep.canonical)}", ""]
    lines += ["## Raw label -> canonical (redundancy collapse)", "",
              "| raw | -> canonical |", "|---|---|"]
    for k in sorted(RAW_TO_CANON):
        lines.append(f"| {k} | {RAW_TO_CANON[k]} |")
    lines += ["", "## Raw value counts (before)", "", "| raw domain | n |", "|---|---:|"]
    for k, v in rep.raw_value_counts.items():
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "## Provenance (after)", "", "| provenance | n |", "|---|---:|"]
    for k, v in rep.provenance_counts.items():
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "## Classifier (English-side TF-IDF + LogReg)", "",
              f"- Holdout accuracy: **{rep.holdout_accuracy}**  |  macro-F1: **{rep.holdout_macro_f1}**",
              f"- Per-class F1: {rep.per_class_f1}",
              f"- Mean confidence on inferred rows: {rep.inferred_conf_mean}", "",
              "## Inferred rows filled, by language pair", "", "| pair | inferred |", "|---|---:|"]
    for k, v in sorted(rep.filled_by_pair.items()):
        lines.append(f"| {k} | {v:,} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Normalise + infer IndicQE-APE domain labels.")
    ap.add_argument("in_parquet")
    ap.add_argument("out_parquet")
    ap.add_argument("--report", default=None, help="write a markdown audit report here")
    ap.add_argument("--no-raw", action="store_true", help="do not keep the domain_raw column")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    out, rep = finalize_domains(df, keep_raw=not args.no_raw)
    out.to_parquet(args.out_parquet, index=False)
    print(f"wrote {args.out_parquet}  ({len(out):,} rows)")
    print(rep.to_json())
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(_report_md(rep))
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
