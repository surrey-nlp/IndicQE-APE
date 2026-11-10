"""Evaluation metrics for QE, APE and MQM.

QE  → Pearson, Spearman, Kendall correlations of predicted vs. gold scores.
APE → sacreBLEU, chrF++, COMET-Kiwi (Unbabel/wmt22-cometkiwi-da), TER vs. PE.
MQM → segment-level severity-weighted score; span-level F1 (token overlap).

All implementations are deterministic. COMET-Kiwi loads on first use to avoid
import-time GPU allocation when only string metrics are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- QE
@dataclass(frozen=True)
class CorrelationResult:
    pearson: float
    spearman: float
    kendall: float
    n: int


def correlation(predictions: Sequence[float], gold: Sequence[float]) -> CorrelationResult:
    """Compute Pearson, Spearman and Kendall between paired prediction/gold sequences.

    Items where either side is None / NaN are dropped pairwise.
    """
    p, g = [], []
    for x, y in zip(predictions, gold, strict=True):
        if x is None or y is None:
            continue
        if isinstance(x, float) and np.isnan(x):
            continue
        if isinstance(y, float) and np.isnan(y):
            continue
        p.append(float(x))
        g.append(float(y))
    if len(p) < 2:
        return CorrelationResult(float("nan"), float("nan"), float("nan"), len(p))
    pr = stats.pearsonr(p, g).statistic
    sp = stats.spearmanr(p, g).statistic
    kd = stats.kendalltau(p, g).statistic
    return CorrelationResult(float(pr), float(sp), float(kd), len(p))


def prompt_sensitivity(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, float]:
    """Agreement between two prompt variants' predicted scores on the same items.

    `scores_a`/`scores_b` are aligned predicted-score vectors (e.g. GEMBA-DA vs
    GEMBA-SQM). Pairs where either side is None/NaN are dropped. Returns the
    correlation between the two vectors (how much the choice of prompt moves the
    ranking), and the score-level offset (`mean_diff` = a − b: positive means the
    first variant scores higher). A high Pearson/Spearman with a small mean_diff
    means the metric is robust to the prompt; a large gap is the prompt-sensitivity
    story for the paper.
    """
    a, b = [], []
    for x, y in zip(scores_a, scores_b, strict=True):
        if x is None or y is None:
            continue
        if isinstance(x, float) and np.isnan(x):
            continue
        if isinstance(y, float) and np.isnan(y):
            continue
        a.append(float(x))
        b.append(float(y))
    n = len(a)
    if n < 2:
        nan = float("nan")
        return {"n": n, "pearson": nan, "spearman": nan,
                "mean_diff": nan, "mean_abs_diff": nan, "std_diff": nan}
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    diffs = arr_a - arr_b
    return {
        "n": n,
        "pearson": float(stats.pearsonr(arr_a, arr_b).statistic),
        "spearman": float(stats.spearmanr(arr_a, arr_b).statistic),
        "mean_diff": float(np.mean(diffs)),
        "mean_abs_diff": float(np.mean(np.abs(diffs))),
        "std_diff": float(np.std(diffs)),
    }


# ---------------------------------------------------------------- APE
@dataclass(frozen=True)
class APEResult:
    bleu: float        # sacreBLEU, 13a word tokeniser — Latin-oriented, UNDER-credits Indic morphology
    chrf: float        # chrF++ (char n-grams) — the trustworthy metric for Indic targets
    ter: float         # sacreBLEU TER, word-level — INFLATED for agglutinative Indic (e.g. Tamil)
    chrter: float      # character-level TER — Indic-appropriate edit rate (lower = better)
    comet_kiwi: float | None
    n: int


def _char_ter(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Character-level Translation Edit Rate (%) = total char edit-distance / total ref chars * 100.
    Word-level TER mis-handles Indic morphology (a one-character inflection = a whole-word edit),
    inflating the score; char-TER avoids tokenisation entirely and is the right edit metric here."""
    def edist(a: str, b: str) -> int:
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return max(m, n)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            ai = a[i - 1]
            for j in range(1, n + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            prev = cur
        return prev[n]
    tot_d = tot_r = 0
    for h, r in zip(hypotheses, references, strict=True):
        tot_d += edist(h, r)
        tot_r += max(len(r), 1)
    return 100.0 * tot_d / max(tot_r, 1)


def ape_string_metrics(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """chrF++ (primary, Indic-safe) + char-TER, plus word-level sacreBLEU/TER (kept for reference;
    word tokenisation is not Indic-appropriate — see APEResult field notes)."""
    import sacrebleu  # type: ignore[import-not-found]
    refs = [list(references)]
    bleu = sacrebleu.corpus_bleu(list(hypotheses), refs).score
    chrf = sacrebleu.corpus_chrf(list(hypotheses), refs, word_order=2).score
    ter  = sacrebleu.corpus_ter(list(hypotheses), refs).score
    chrter = _char_ter(list(hypotheses), list(references))
    return {"bleu": float(bleu), "chrf": float(chrf), "ter": float(ter), "chrter": float(chrter)}


def comet_kiwi(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    *,
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    batch_size: int = 8,
    gpus: int = 0,
) -> float:
    """Reference-free COMET-Kiwi system score in [0, 1]."""
    from comet import download_model, load_from_checkpoint  # type: ignore[import-not-found]
    path = download_model(model_name)
    model = load_from_checkpoint(path)
    data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses, strict=True)]
    output = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    return float(output.system_score)


def ape_evaluate(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    with_comet: bool = True,
) -> APEResult:
    string_metrics = ape_string_metrics(hypotheses, references)
    comet = comet_kiwi(sources, hypotheses) if with_comet else None
    return APEResult(
        bleu=string_metrics["bleu"],
        chrf=string_metrics["chrf"],
        ter=string_metrics["ter"],
        chrter=string_metrics["chrter"],
        comet_kiwi=comet,
        n=len(hypotheses),
    )


# ---------------------------------------------------------------- MQM
SEVERITY_WEIGHTS = {"minor": 1, "major": 5, "critical": 10}


def mqm_segment_score(errors: Iterable[dict]) -> float:
    """Sum severity weights for one segment's predicted/gold errors."""
    return float(sum(SEVERITY_WEIGHTS.get(e.get("severity", "").lower(), 0) for e in errors))


def mqm_span_f1(
    predicted: list[dict],
    gold: list[dict],
    *,
    jaccard_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Per-segment span F1 between predicted and gold MQM error sets.

    A predicted span is considered a true positive iff it shares the same
    severity AND its tokenised text has Jaccard overlap >= ``jaccard_threshold``
    with some unmatched gold span. Default 0.5 — i.e. at least half the union
    of tokens is shared. Both-empty (no errors predicted, none in gold) returns
    F1=1.0; one-empty returns 0.0.
    """
    def fingerprint(e: dict) -> tuple[str, frozenset[str]]:
        sev = e.get("severity", "").lower()
        toks = frozenset((e.get("span", "") or "").lower().split())
        return sev, toks

    pred_fps = [fingerprint(e) for e in predicted]
    gold_fps = [fingerprint(e) for e in gold]

    if not pred_fps and not gold_fps:
        return 1.0, 1.0, 1.0

    matched_gold: set[int] = set()
    tp = 0
    for psev, ptoks in pred_fps:
        if not ptoks:
            continue
        for i, (gsev, gtoks) in enumerate(gold_fps):
            if i in matched_gold or psev != gsev or not gtoks:
                continue
            inter = len(ptoks & gtoks)
            if inter == 0:
                continue
            jacc = inter / len(ptoks | gtoks)
            if jacc >= jaccard_threshold:
                tp += 1
                matched_gold.add(i)
                break

    precision = tp / len(pred_fps) if pred_fps else 0.0
    recall    = tp / len(gold_fps) if gold_fps else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def mqm_dataset_score(
    pred_errors: Sequence[list[dict]],
    gold_errors: Sequence[list[dict]],
    *,
    skip_both_empty: bool = True,
    jaccard_threshold: float = 0.5,
) -> dict[str, float]:
    """Aggregate MQM metrics across a dataset.

    When ``skip_both_empty=True`` (default), segments where both predicted and
    gold have zero errors are excluded from the macro span averages — otherwise
    a model that always predicts no errors would get a free 1.0 on every
    error-free segment, inflating F1.
    """
    if len(pred_errors) != len(gold_errors):
        raise ValueError(f"length mismatch: {len(pred_errors)} vs {len(gold_errors)}")
    pred_scores = np.array([mqm_segment_score(e) for e in pred_errors])
    gold_scores = np.array([mqm_segment_score(e) for e in gold_errors])
    seg_corr = correlation(pred_scores.tolist(), gold_scores.tolist())

    p_sum = r_sum = f_sum = 0.0
    n_macro = 0
    for p, g in zip(pred_errors, gold_errors, strict=True):
        if skip_both_empty and not p and not g:
            continue
        pr, rc, fs = mqm_span_f1(p, g, jaccard_threshold=jaccard_threshold)
        p_sum += pr
        r_sum += rc
        f_sum += fs
        n_macro += 1

    n = len(pred_errors)
    return {
        "segment_pearson": seg_corr.pearson,
        "segment_spearman": seg_corr.spearman,
        "span_precision_macro": p_sum / n_macro if n_macro else 0.0,
        "span_recall_macro":    r_sum / n_macro if n_macro else 0.0,
        "span_f1_macro":        f_sum / n_macro if n_macro else 0.0,
        "n": n,
        "n_macro": n_macro,
    }
