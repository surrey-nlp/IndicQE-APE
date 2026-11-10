"""Language-pair metadata.

The eight directional pairs and nine languages in the consolidated release.
Used for prompt construction (full language names) and per-pair reporting.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lang:
    code: str          # ISO 639-1
    name: str          # English name
    family: str        # Indo-Aryan / Dravidian / Uralic / Germanic
    script: str        # Devanagari / Tamil / Telugu / Gujarati / Sinhala / Latin


LANGS: dict[str, Lang] = {
    "en": Lang("en", "English",   "Germanic",   "Latin"),
    "hi": Lang("hi", "Hindi",     "Indo-Aryan", "Devanagari"),
    "mr": Lang("mr", "Marathi",   "Indo-Aryan", "Devanagari"),
    "ta": Lang("ta", "Tamil",     "Dravidian",  "Tamil"),
    "gu": Lang("gu", "Gujarati",  "Indo-Aryan", "Gujarati"),
    "te": Lang("te", "Telugu",    "Dravidian",  "Telugu"),
    "ml": Lang("ml", "Malayalam", "Dravidian",  "Malayalam"),
    "et": Lang("et", "Estonian",  "Uralic",     "Latin"),
    "ne": Lang("ne", "Nepali",    "Indo-Aryan", "Devanagari"),
    "si": Lang("si", "Sinhala",   "Indo-Aryan", "Sinhala"),
}


@dataclass(frozen=True)
class Pair:
    code: str          # e.g., "enhi" — first 2 chars source, last 2 chars target
    src: str           # source ISO code
    tgt: str           # target ISO code
    direction: str     # "X→Indic" / "Indic→X" / "Estonian→X"
    wmt_origin: str    # provenance: which WMT edition contributed this pair

    @property
    def src_lang(self) -> Lang:
        return LANGS[self.src]

    @property
    def tgt_lang(self) -> Lang:
        return LANGS[self.tgt]

    @property
    def display(self) -> str:
        return f"{self.src_lang.name}→{self.tgt_lang.name}"


PAIRS: dict[str, Pair] = {
    "enhi": Pair("enhi", "en", "hi", "English→Indic", "WMT 2023/2024 QE"),
    "enmr": Pair("enmr", "en", "mr", "English→Indic", "WMT 2022 QE & APE"),
    "enta": Pair("enta", "en", "ta", "English→Indic", "WMT 2023/2024 QE"),
    "engu": Pair("engu", "en", "gu", "English→Indic", "WMT 2023/2024 QE"),
    "ente": Pair("ente", "en", "te", "English→Indic", "WMT 2023/2024 QE"),
    "enml": Pair("enml", "en", "ml", "English→Indic", "Surrey en-ml (human PE/DA/MQM)"),
    "eten": Pair("eten", "et", "en", "Estonian→English", "WMT 2020/2021 QE (MLQE-PE)"),
    "neen": Pair("neen", "ne", "en", "Indic→English",    "WMT 2020/2021 QE (MLQE-PE)"),
    "sien": Pair("sien", "si", "en", "Indic→English",    "WMT 2020/2021 QE (MLQE-PE)"),
}


# curated En->Indic pairs (Surrey-curated) vs X->En MLQE-PE passthrough (eval-only)
CURATED_PAIRS = ("enhi", "enmr", "enta", "ente", "engu")
XEN_PAIRS = ("eten", "neen", "sien")

# hyphenated <-> 4-char code maps (the unified dataset uses hyphenated 'en-hi')
_HYPHEN = {code: f"{p.src}-{p.tgt}" for code, p in PAIRS.items()}
_FROM_HYPHEN = {v: k for k, v in _HYPHEN.items()}


def to_hyphen(code: str) -> str:
    """4-char code ('enhi') -> hyphenated ('en-hi'). Idempotent on hyphenated input."""
    if "-" in code:
        return code
    if code not in _HYPHEN:
        raise ValueError(f"Unknown language pair: {code}. Known: {sorted(PAIRS)}")
    return _HYPHEN[code]


def from_hyphen(code: str) -> str:
    """Hyphenated ('en-hi') -> 4-char code ('enhi'). Idempotent on 4-char input."""
    return _FROM_HYPHEN.get(code, code)


def parse_pair(code: str) -> Pair:
    """Parse a pair code (4-char 'enhi' or hyphenated 'en-hi') into a Pair."""
    key = from_hyphen(code) if "-" in code else code
    if key not in PAIRS:
        raise ValueError(f"Unknown language pair: {code}. Known: {sorted(PAIRS)}")
    return PAIRS[key]


def all_pair_codes() -> list[str]:
    return list(PAIRS.keys())
