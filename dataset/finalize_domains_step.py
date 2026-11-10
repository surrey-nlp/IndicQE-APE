#!/usr/bin/env python3
"""Standalone domain-finalisation (split from make_hf_repo to fit one call).

Same design as normalize_domain.finalize_domains — normalise gold + word/char TF-IDF LR
infer the unlabelled + provenance — but fits on a 40k stratified sample of the gold rows so
the vectoriser+LR complete within budget. Inference on the ~34k unlabelled rows is
statistically identical (LR on TF-IDF saturates well below 40k). Writes domain-augmented
curated splits to release_build/curated_dom/ (domain, domain_raw, domain_provenance,
domain_confidence, has_domain redefined = annotated)."""
import time, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import normalize_domain as nd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

CUR, OUT = "release_build/curated", "release_build/curated_dom"
FIT_N = 40_000


def pipe():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                           max_features=12_000, sublinear_tf=True)
    return Pipeline([("feats", FeatureUnion([("word", word), ("char", char)])),
                     ("clf", LogisticRegression(C=8.0, max_iter=350, class_weight="balanced",
                                                solver="lbfgs", n_jobs=-1))])


def main():
    t = time.time()
    parts = []
    for sp in ("train", "dev", "test"):
        d = pd.read_parquet(f"{CUR}/{sp}.parquet"); d["split"] = sp; parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    raw = df["domain"]
    df["domain_raw"] = raw
    canon = raw.map(nd._norm_one)
    eng = pd.Series(nd.english_side(df), index=df.index).fillna("").astype(str)
    labelled = canon.notna()
    to_infer = (~labelled) & (eng.str.len() > 0)

    prov = pd.Series(np.where(labelled, "annotated", "none"), index=df.index)
    conf = pd.Series(np.nan, index=df.index, dtype="float64")
    dom = canon.copy()

    # Inference is CONSERVATIVE: train only on the 6 established domains, so unlabelled rows
    # (mostly X->En MLQE) are inferred into that space. The 4 new domains (news/finance/tech/
    # sports) stay GOLD-ONLY (En-Ml) — first-class in the taxonomy, but never guessed onto
    # other pairs. En-Ml's gold labels are preserved as 'annotated' regardless.
    ORIG6 = {"health", "legal", "general", "tourism", "education", "other"}
    fit_ok = labelled & canon.isin(ORIG6)
    gi = np.where(fit_ok.values)[0]
    yv = canon[fit_ok].values
    if len(gi) > FIT_N:
        gi_fit, _ = train_test_split(gi, train_size=FIT_N, stratify=yv, random_state=0)
    else:
        gi_fit = gi
    clf = pipe()
    clf.fit(eng.iloc[gi_fit].values, canon.iloc[gi_fit].values)
    # quick holdout metric on a disjoint gold sample
    hold = np.setdiff1d(gi, gi_fit)[:15000]
    if len(hold):
        pr = clf.predict(eng.iloc[hold].values)
        acc = round(accuracy_score(canon.iloc[hold].values, pr), 3)
        mf1 = round(f1_score(canon.iloc[hold].values, pr, average="macro", zero_division=0), 3)
    else:
        acc = mf1 = None

    if to_infer.any():
        Xi = eng[to_infer].values
        proba = clf.predict_proba(Xi); classes = clf.named_steps["clf"].classes_
        dom.loc[to_infer] = classes[proba.argmax(1)]
        conf.loc[to_infer] = proba.max(1)
        prov.loc[to_infer] = "inferred"
    resid = dom.isna()
    dom.loc[resid] = "other"; prov.loc[resid] = "fallback"

    df["domain"] = dom.values
    df["domain_provenance"] = prov.values
    df["domain_confidence"] = conf.values
    df["has_domain"] = (prov == "annotated").values

    os.makedirs(OUT, exist_ok=True)
    for sp, g in df.groupby("split"):
        g.drop(columns=["split"]).to_parquet(f"{OUT}/{sp}.parquet")
    print("finalised %d rows in %.1fs | holdout acc=%s macroF1=%s" % (len(df), time.time() - t, acc, mf1))
    print("provenance:", df.domain_provenance.value_counts().to_dict())
    print("domains:", df.domain.value_counts().to_dict())


if __name__ == "__main__":
    main()
