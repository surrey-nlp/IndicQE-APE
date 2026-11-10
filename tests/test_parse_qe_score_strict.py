"""Tests for parse_qe_score_strict.

Every case in REGRESSIONS is a real reply taken from results/qe_rerun512, together with the wrong
value the permissive parser recorded for it. Those three are the reason this parser exists, so
they are asserted explicitly against both parsers: the strict one must refuse or correct, and the
permissive one is asserted to still be wrong so that the test documents the difference rather than
silently passing if someone "fixes" the old parser instead.
"""
from __future__ import annotations

import pytest

from indicqe_ape.prompts import STRICT_MAX_CHARS, parse_qe_score, parse_qe_score_strict

ESSAY = (
    "The Malayalam translation for the given English sentences is evaluated based on accuracy, "
    "grammar, and naturalness. Here are the scores:\n\n1. **First Translation**: The translation "
    "is accurate and conveys the intended meaning well."
)

#: (reply, what the permissive parser recorded, what strict should return)
REGRESSIONS = [
    ("Score (0-100): 88", 0.0, 88.0),      # range start read as the score
    (ESSAY, 1.0, None),                     # list marker read as the score
    ("90\n89\n65\n0", 90.0, None),          # four-shot echo; first number is not the test segment
    ("90, 100, 89, 65", 90.0, None),        # same, comma separated
]


@pytest.mark.parametrize("reply,permissive,strict", REGRESSIONS)
def test_observed_failures(reply, permissive, strict):
    assert parse_qe_score(reply) == permissive, "permissive behaviour changed; update the note"
    assert parse_qe_score_strict(reply) == strict


@pytest.mark.parametrize("reply,expected", [
    ("88", 88.0),
    (" 88 ", 88.0),
    ("88.", 88.0),
    ("88.5", 88.5),
    ("**88**", 88.0),
    ("`88`", 88.0),
    ('"88"', 88.0),
    ("Score: 88", 88.0),
    ("score: 88", 88.0),
    ("Score (0-100): 88", 88.0),
    ("DA score: 7.5", 7.5),
    ("85/100", 85.0),
    ("4/5", 80.0),
    ("9/10", 90.0),
    ("0", 0.0),
    ("100", 100.0),
])
def test_accepts_unambiguous(reply, expected):
    assert parse_qe_score_strict(reply) == pytest.approx(expected)


@pytest.mark.parametrize("reply", [
    None, "", "   ", "\n",
    "N/A", "unable to score", "good", "-",
    "90 89",                                  # two numbers, no label
    "90\n89",
    "88 or 90",
    "0-100",                                  # a scale, not a score
    "(0-100)",
    "The translation is accurate and natural.",  # prose, no number
    "x" * (STRICT_MAX_CHARS + 1),
    "Score: 88 but the second sentence is untranslated so possibly lower",  # over length
])
def test_refuses_ambiguous(reply):
    assert parse_qe_score_strict(reply) is None


def test_clamps_out_of_range():
    assert parse_qe_score_strict("120") == 100.0
    assert parse_qe_score_strict("Score: 999") == 100.0


def test_fraction_on_unrecognised_scale_is_refused():
    # 85/7 is not a QE scale; guessing a normalisation would be inventing a number
    assert parse_qe_score_strict("85/7") is None


def test_length_gate_is_on_the_cleaned_string():
    # markdown is stripped before the length check, so decoration does not push a valid reply over
    assert parse_qe_score_strict("**" * 8 + "88" + "**" * 8) == 88.0


def test_zero_shot_bare_numbers_are_unaffected():
    # the zero-shot cells are all bare numbers; both parsers must agree on them
    for n in ("0", "1", "37", "88", "99", "100"):
        assert parse_qe_score_strict(n) == parse_qe_score(n) == float(n)
