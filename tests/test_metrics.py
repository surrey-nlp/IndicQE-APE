"""Unit tests for the metric implementations.

These are the canonical correctness checks. Run before any results table:
    pytest tests/
"""
from __future__ import annotations

import math

import pytest

from indicqe_ape.metrics import (
    SEVERITY_WEIGHTS,
    correlation,
    mqm_dataset_score,
    mqm_segment_score,
    mqm_span_f1,
)
from indicqe_ape.prompts import parse_qe_score, parse_ape_response, parse_mqm_response


# ---------------------------------------------------------------- correlation
class TestCorrelation:
    def test_perfect_positive(self) -> None:
        r = correlation([1, 2, 3, 4], [1, 2, 3, 4])
        assert r.pearson == pytest.approx(1.0)
        assert r.spearman == pytest.approx(1.0)
        assert r.kendall == pytest.approx(1.0)
        assert r.n == 4

    def test_perfect_negative(self) -> None:
        r = correlation([1, 2, 3, 4], [4, 3, 2, 1])
        assert r.pearson == pytest.approx(-1.0)
        assert r.spearman == pytest.approx(-1.0)
        assert r.kendall == pytest.approx(-1.0)

    def test_drops_none(self) -> None:
        r = correlation([1, 2, None, 4], [1, 2, 3, 4])
        assert r.n == 3
        assert r.pearson == pytest.approx(1.0)

    def test_too_few_points(self) -> None:
        r = correlation([1.0], [1.0])
        assert math.isnan(r.pearson)
        assert r.n == 1


# ---------------------------------------------------------------- MQM
class TestMQM:
    def test_severity_weights(self) -> None:
        assert SEVERITY_WEIGHTS == {"minor": 1, "major": 5, "critical": 10}

    def test_segment_score(self) -> None:
        errs = [
            {"severity": "minor"},
            {"severity": "major"},
            {"severity": "critical"},
            {"severity": "MINOR"},  # case-insensitive via .lower() in metrics
        ]
        assert mqm_segment_score(errs) == 17.0  # 1 + 5 + 10 + 1

    def test_span_f1_perfect(self) -> None:
        pred = [{"severity": "major", "span": "the dog"}]
        gold = [{"severity": "major", "span": "the dog"}]
        p, r, f = mqm_span_f1(pred, gold)
        assert p == r == f == 1.0

    def test_span_f1_severity_mismatch(self) -> None:
        pred = [{"severity": "minor", "span": "the dog"}]
        gold = [{"severity": "major", "span": "the dog"}]
        p, r, f = mqm_span_f1(pred, gold)
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_span_f1_partial_overlap_below_threshold(self) -> None:
        # Single-token overlap "barked" → Jaccard 1/4 = 0.25, below default 0.5
        pred = [{"severity": "major", "span": "barked at the cat"}]
        gold = [{"severity": "major", "span": "the dog barked"}]
        p, r, f = mqm_span_f1(pred, gold)
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_span_f1_partial_overlap_above_threshold(self) -> None:
        # Two-token overlap, Jaccard 2/3 = 0.67, above default 0.5
        pred = [{"severity": "major", "span": "the dog"}]
        gold = [{"severity": "major", "span": "the dog barked"}]
        p, r, f = mqm_span_f1(pred, gold)
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_dataset_score_skips_both_empty(self) -> None:
        # If both empty is counted, macro F1 = (1.0 + 0.0)/2 = 0.5.
        # With skip_both_empty=True (default), only the second segment counts → 0.0.
        pred = [[], []]
        gold = [[], [{"severity": "major", "span": "x"}]]
        out = mqm_dataset_score(pred, gold)
        assert out["n_macro"] == 1
        assert out["span_f1_macro"] == 0.0

    def test_span_f1_empty_both(self) -> None:
        # No errors predicted, none in gold = perfect score
        p, r, f = mqm_span_f1([], [])
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_dataset_score_shape(self) -> None:
        pred = [
            [{"severity": "minor", "span": "a"}],
            [],
        ]
        gold = [
            [{"severity": "minor", "span": "a"}],
            [{"severity": "major", "span": "b"}],
        ]
        out = mqm_dataset_score(pred, gold)
        assert out["n"] == 2
        assert "segment_pearson" in out
        assert "span_f1_macro" in out


# ---------------------------------------------------------------- parsers
class TestParsers:
    def test_parse_qe_score_clean(self) -> None:
        assert parse_qe_score("87") == 87.0

    def test_parse_qe_score_with_text(self) -> None:
        assert parse_qe_score("Score: 92/100, the translation is...") == 92.0

    def test_parse_qe_score_clamp(self) -> None:
        assert parse_qe_score("150") == 100.0
        assert parse_qe_score("-20") == 0.0

    def test_parse_qe_score_garbage(self) -> None:
        assert parse_qe_score("no score here") is None
        assert parse_qe_score("") is None

    def test_parse_ape_strips(self) -> None:
        assert parse_ape_response('"corrected text"') == "corrected text"
        assert parse_ape_response("- corrected text") == "corrected text"

    def test_parse_mqm_lines(self) -> None:
        resp = (
            "minor | mistranslation | the dog\n"
            "major | omission | walked\n"
            "garbage line\n"
        )
        out = parse_mqm_response(resp)
        assert len(out) == 2
        assert out[0]["severity"] == "minor"
        assert out[0]["category"] == "mistranslation"
        assert out[0]["span"] == "the dog"
        assert out[1]["severity"] == "major"
