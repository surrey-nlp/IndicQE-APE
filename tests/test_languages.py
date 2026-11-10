"""Tests for the language-pair registry."""
from __future__ import annotations

import pytest

from indicqe_ape.languages import LANGS, PAIRS, all_pair_codes, parse_pair


class TestLanguages:
    def test_nine_pairs(self) -> None:
        # en-ml joined the registry when the Malayalam resource landed; this test asserted
        # eight until 2026-08-06 and had been failing since.
        assert len(PAIRS) == 9
        assert set(all_pair_codes()) == {"enhi", "enmr", "enta", "engu", "ente", "enml",
                                         "eten", "neen", "sien"}

    def test_estonian_is_uralic(self) -> None:
        assert LANGS["et"].family == "Uralic"

    def test_indo_aryan_pairs(self) -> None:
        # 5 Indo-Aryan: hi, mr, gu, ne, si
        indo_aryan = [code for code, lang in LANGS.items() if lang.family == "Indo-Aryan"]
        assert set(indo_aryan) == {"hi", "mr", "gu", "ne", "si"}

    def test_dravidian_pairs(self) -> None:
        # 3 Dravidian: ta, te, ml. Malayalam is Dravidian, not Indo-Aryan, so adding en-ml
        # changed this set as well as the pair count.
        dravidian = [code for code, lang in LANGS.items() if lang.family == "Dravidian"]
        assert set(dravidian) == {"ta", "te", "ml"}

    def test_parse_pair_known(self) -> None:
        p = parse_pair("enhi")
        assert p.src == "en" and p.tgt == "hi"
        assert p.src_lang.name == "English"
        assert p.tgt_lang.name == "Hindi"
        assert "Indic" in p.direction

    def test_parse_pair_unknown(self) -> None:
        with pytest.raises(ValueError):
            parse_pair("xxyy")

    def test_eten_is_estonian_to_english(self) -> None:
        # Regression: eten was previously mis-read as Telugu→English.
        p = parse_pair("eten")
        assert p.src == "et"
        assert p.tgt == "en"
        assert p.src_lang.name == "Estonian"
        assert "Estonian" in p.direction

    def test_display_names(self) -> None:
        assert parse_pair("enhi").display == "English→Hindi"
        assert parse_pair("eten").display == "Estonian→English"
