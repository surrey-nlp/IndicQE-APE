"""Prompt templates for QE, APE, and MQM tasks.

QE uses the canonical GEMBA prompts (Kocmi & Federmann, 2023) in reference-free
mode: GEMBA-DA (continuous 0-100, meaning/grammar anchors) and GEMBA-SQM (same
scale, descriptive quality-band rubric). Every QE run scores both variants so the
prompt-sensitivity gap is reported per run. APE/MQM templates stay deterministic;
chain-of-thought variants belong in a separate experiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .languages import Pair


# ---------------------------------------------------------------- QE (DA 0-100)
# Canonical GEMBA prompts (Kocmi & Federmann, 2023, "Large Language Models Are
# State-of-the-Art Evaluators of Translation Quality"), reference-free / QE mode.
# GEMBA-DA = continuous 0-100 with meaning/grammar anchors; GEMBA-SQM = same scale
# with a descriptive quality-band rubric. Both share the quoted-segment user block.


@dataclass(frozen=True)
class QETemplate:
    """A QE prompt variant. `system`/`user` are format strings.

    system fields: src_name, tgt_name. user fields: src_name, tgt_name, source, mt.
    """
    key: str
    system: str
    user: str


GEMBA_DA = QETemplate(
    key="gemba-da",
    system=(
        "Score the following translation from {src_name} to {tgt_name} on a "
        'continuous scale from 0 to 100, where a score of zero means "no meaning '
        'preserved" and a score of one hundred means "perfect meaning and grammar". '
        "Reply with only the integer score and nothing else."
    ),
    user=(
        '{src_name} source: "{source}"\n'
        '{tgt_name} translation: "{mt}"\n'
        "Score (0-100):"
    ),
)

GEMBA_SQM = QETemplate(
    key="gemba-sqm",
    system=(
        "Score the following translation from {src_name} to {tgt_name} on a "
        'continuous scale from 0 to 100 that starts with "No meaning preserved", '
        'goes through "Some meaning preserved", then "Most meaning preserved and few '
        'grammar mistakes", up to "Perfect meaning and grammar". '
        "Reply with only the integer score and nothing else."
    ),
    user=(
        '{src_name} source: "{source}"\n'
        '{tgt_name} translation: "{mt}"\n'
        "Score (0-100):"
    ),
)

QE_TEMPLATES: dict[str, QETemplate] = {t.key: t for t in (GEMBA_DA, GEMBA_SQM)}
DEFAULT_QE_TEMPLATES: tuple[str, ...] = ("gemba-da", "gemba-sqm")


# ---------------------------------------------------------------- APE (post-edit)
APE_SYSTEM = (
    "You are an expert post-editor. Given a source sentence and a machine translation, "
    "produce a corrected translation that is fluent, accurate, and faithful to the source. "
    "Reply with only the corrected translation, no explanation."
)

APE_USER_TEMPLATE = (
    "Source ({src_name}): {source}\n"
    "Machine translation ({tgt_name}): {mt}\n\n"
    "Corrected translation ({tgt_name}):"
)


# ---------------------------------------------------------------- TRANSLATE (dedicated-MT retranslate)
# Used by the `ape --mode retranslate` track for dedicated MT systems (e.g. sarvamai/sarvam-translate):
# translate the source from scratch, then score the output reference-free with the trained metrics.
# Format follows sarvam-translate's card (system = "Translate the text below to {lang}.", user = source).
TRANSLATE_SYSTEM = "Translate the text below to {tgt_name}."


def render_translate(pair: Pair, source: str) -> tuple[str, str]:
    """Return (system, user) for a from-scratch translation of `source` into the target language."""
    return TRANSLATE_SYSTEM.format(tgt_name=pair.tgt_lang.name), source


# ---------------------------------------------------------------- MQM (severity-weighted)
MQM_SYSTEM = (
    "You are an expert MQM annotator. Identify error spans in the translation, classify each "
    "by severity (minor / major / critical) and category (mistranslation, omission, addition, "
    "grammar, fluency, style, terminology, locale-violation, source-error, non-translation). "
    "Reply with one error per line in the format: SEVERITY | CATEGORY | EXACT_SPAN_TEXT."
)

MQM_USER_TEMPLATE = (
    "Source ({src_name}): {source}\n"
    "Translation ({tgt_name}): {mt}\n\n"
    "Errors:"
)


# ----------------------------------------------------------------- few-shot
@dataclass
class Example:
    source: str
    mt: str
    label: str  # gold answer (DA score, post-edit, or MQM lines)


def render_qe(
    pair: Pair,
    source: str,
    mt: str,
    fewshot: list[Example] | None = None,
    template: str | QETemplate = "gemba-da",
) -> tuple[str, str]:
    """Return (system, user) message pair for the QE task under `template`.

    `template` is a key in QE_TEMPLATES ("gemba-da" / "gemba-sqm") or a QETemplate.
    """
    tpl = template if isinstance(template, QETemplate) else QE_TEMPLATES[template]
    src_name, tgt_name = pair.src_lang.name, pair.tgt_lang.name
    sys_msg = tpl.system.format(src_name=src_name, tgt_name=tgt_name)
    user_blocks: list[str] = []
    if fewshot:
        for ex in fewshot:
            user_blocks.append(
                tpl.user.format(src_name=src_name, tgt_name=tgt_name, source=ex.source, mt=ex.mt)
                + " " + ex.label
            )
    user_blocks.append(
        tpl.user.format(src_name=src_name, tgt_name=tgt_name, source=source, mt=mt)
    )
    return sys_msg, "\n\n".join(user_blocks)


def render_ape(pair: Pair, source: str, mt: str, fewshot: list[Example] | None = None) -> tuple[str, str]:
    """Return (system, user) message pair for the APE task."""
    user_blocks: list[str] = []
    if fewshot:
        for ex in fewshot:
            user_blocks.append(
                APE_USER_TEMPLATE.format(
                    src_name=pair.src_lang.name, tgt_name=pair.tgt_lang.name,
                    source=ex.source, mt=ex.mt,
                )
                + " " + ex.label
            )
    user_blocks.append(
        APE_USER_TEMPLATE.format(
            src_name=pair.src_lang.name, tgt_name=pair.tgt_lang.name,
            source=source, mt=mt,
        )
    )
    return APE_SYSTEM, "\n\n".join(user_blocks)


def render_mqm(pair: Pair, source: str, mt: str, fewshot: list[Example] | None = None) -> tuple[str, str]:
    user_blocks: list[str] = []
    if fewshot:
        for ex in fewshot:
            user_blocks.append(
                MQM_USER_TEMPLATE.format(
                    src_name=pair.src_lang.name, tgt_name=pair.tgt_lang.name,
                    source=ex.source, mt=ex.mt,
                ) + "\n" + ex.label
            )
    user_blocks.append(
        MQM_USER_TEMPLATE.format(
            src_name=pair.src_lang.name, tgt_name=pair.tgt_lang.name,
            source=source, mt=mt,
        )
    )
    return MQM_SYSTEM, "\n\n".join(user_blocks)


# ----------------------------------------------------------------- parsers
_QE_NUM = re.compile(r"-?\d+(?:\.\d+)?")
# A score cue ("Score: 96", "rating - 80", "quality 75") — its captured number beats any leading
# list bullet. This is what fixes "1. Score: 96" being read as 1 instead of 96.
_QE_SCORE_CUE = re.compile(r"(?:score|rating|quality|\bqe\b|\bda\b)\D{0,12}(-?\d+(?:\.\d+)?)", re.I)
_QE_FRAC = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)")
# A translation-only model (e.g. sarvam-translate) emits the translation, not a score.
_QE_TRANSLATION = re.compile(r"here'?s the translat|^\s*(?:the )?translation\b|translation:\s", re.I)


def parse_qe_score(response: str) -> float | None:
    """Extract a 0-100 QE score from an LLM response.

    Robust to: leading list bullets ("1. Score: 96" -> 96, not 1), explicit "Score: N",
    fractions ("85/100", "4/5"), and non-answers (a translation instead of a score -> None).
    Falls back to the first number for the clean "86" case."""
    if not response:
        return None
    s = response.strip()
    # model translated instead of scoring -> no score (unless it also gives an explicit cue)
    if _QE_TRANSLATION.search(s) and not _QE_SCORE_CUE.search(s):
        return None
    # explicit score cue wins over a leading enumeration digit. Take the LAST cue: a reasoning model
    # (e.g. sarvam-m) concludes with its score; a direct model has exactly one, so last == only.
    cues = _QE_SCORE_CUE.findall(s)
    if cues:
        try:
            return max(0.0, min(100.0, float(cues[-1])))
        except ValueError:
            pass
    # x/100, x/10 or x/5 -> normalise to 0..100
    fr = _QE_FRAC.search(s)
    if fr:
        num, den = float(fr.group(1)), float(fr.group(2))
        if den in (5.0, 10.0, 100.0) and 0 <= num <= den:
            return max(0.0, min(100.0, num / den * 100.0))
    # bare number: a short answer IS the score (first); a long reasoning trace ends with it (last).
    nums = _QE_NUM.findall(s)
    if not nums:
        return None
    pick = nums[-1] if len(s) > 40 else nums[0]
    try:
        return max(0.0, min(100.0, float(pick)))
    except ValueError:
        return None


# ------------------------------------------------------- strict QE score parsing
#
# `parse_qe_score` above is deliberately permissive: it will find a number almost anywhere. That
# turned out to record wrong scores while reporting a 100% parse rate, which is worse than a parse
# failure because a failure is visible and a scraped number is not. Three real cases from
# results/qe_rerun512:
#
#   'Score (0-100): 88'  -> 0.0   the cue gap \D{0,12} skipped " (" and took the range start
#   '<590-char essay>'   -> 1.0   the cue "scores:" was followed by the list marker "1."
#   '90\n89\n65\n0'      -> 90.0  four numbers under four shots; the first is not the test segment
#
# The strict parser accepts a reply only when it is unambiguously ONE score, and refuses otherwise.
# Refusing is the honest outcome: it shows up as a compliance failure rather than as a wrong number.

#: replies longer than this are prose, not a score. The longest legitimate reply observed is
#: "Score (0-100): 88" at 17 characters, so this leaves generous headroom.
STRICT_MAX_CHARS = 60
_STRICT_MD = re.compile(r"[*`_#\"']")
_STRICT_BARE = re.compile(r"\d{1,3}(?:\.\d+)?")
_STRICT_FRAC = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*/\s*(\d{1,3})")


def parse_qe_score_strict(response: str | None) -> float | None:
    """Extract a QE score only when the reply is unambiguously a single score.

    Accepts a bare number ("88", "88.5", "**88**", "88."), a labelled number where the label ends
    in a colon ("Score: 88", "Score (0-100): 88"), and a fraction on a recognised scale ("85/100").
    Returns None for multi-number replies, prose, and empty responses.

    Unlike `parse_qe_score`, this never guesses. A None here means the model did not comply, which
    is a reportable outcome rather than a silent substitution.
    """
    if not response:
        return None
    s = _STRICT_MD.sub("", str(response)).strip()
    if not s or len(s) > STRICT_MAX_CHARS:
        return None

    # A label may precede the score ("Score (0-100): 88"). Take what follows the FINAL colon; the
    # scale range then falls inside the discarded label rather than being read as the answer.
    body = (s.rsplit(":", 1)[1] if ":" in s else s).strip().rstrip(".").strip()
    if not body:
        return None

    if _STRICT_BARE.fullmatch(body):
        return max(0.0, min(100.0, float(body)))

    fr = _STRICT_FRAC.fullmatch(body)
    if fr:
        num, den = float(fr.group(1)), float(fr.group(2))
        if den in (5.0, 10.0, 100.0) and 0 <= num <= den:
            return max(0.0, min(100.0, num / den * 100.0))

    return None


def parse_ape_response(response: str) -> str:
    """Strip whitespace and trailing markdown / quotes from an APE response."""
    if not response:
        return ""
    text = response.strip()
    # Strip leading bullet/quote markers
    for prefix in ("```", "> ", "- ", "* "):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.strip()


def parse_mqm_response(response: str) -> list[dict]:
    """Parse `SEVERITY | CATEGORY | SPAN` lines into a list of error dicts."""
    out: list[dict] = []
    for line in (response or "").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        sev = parts[0].lower()
        if sev not in {"minor", "major", "critical"}:
            continue
        out.append({"severity": sev, "category": parts[1], "span": " | ".join(parts[2:])})
    return out
