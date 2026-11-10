"""Tests for the prompt template renderers."""
from __future__ import annotations

from indicqe_ape.languages import parse_pair
from indicqe_ape.prompts import Example, render_ape, render_mqm, render_qe


PAIR = parse_pair("enhi")


class TestRenderQE:
    def test_zero_shot_basic(self) -> None:
        # default template is canonical GEMBA-DA
        sys, usr = render_qe(PAIR, "Hello.", "नमस्ते।")
        assert "from English to Hindi" in sys
        assert "no meaning preserved" in sys.lower()
        assert 'English source: "Hello."' in usr
        assert 'Hindi translation: "नमस्ते।"' in usr
        assert usr.rstrip().endswith("Score (0-100):")

    def test_sqm_template_has_rubric(self) -> None:
        sys, _ = render_qe(PAIR, "Hello.", "नमस्ते।", template="gemba-sqm")
        assert "Some meaning preserved" in sys
        assert "Most meaning preserved and few" in sys

    def test_few_shot_includes_label(self) -> None:
        ex = Example("Cat.", "बिल्ली।", "92")
        sys, usr = render_qe(PAIR, "Hello.", "नमस्ते।", fewshot=[ex])
        assert "Cat." in usr and "बिल्ली।" in usr and "92" in usr
        # Both the example and the actual query are present
        assert usr.count("English source:") == 2


class TestRenderAPE:
    def test_zero_shot(self) -> None:
        sys, usr = render_ape(PAIR, "Cat.", "बिल्ली।")
        assert "post-editor" in sys.lower()
        assert "Source (English): Cat." in usr
        assert "Machine translation (Hindi): बिल्ली।" in usr


class TestRenderMQM:
    def test_zero_shot(self) -> None:
        sys, usr = render_mqm(PAIR, "Cat.", "बिल्ली।")
        assert "MQM annotator" in sys
        assert "minor / major / critical" in sys
        assert "SEVERITY | CATEGORY" in sys
