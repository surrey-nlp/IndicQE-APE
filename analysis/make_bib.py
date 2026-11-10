#!/usr/bin/env python3
r"""make_bib.py — regenerate paper/latex/custom.bib from the ACL Anthology bib dump.

WHY THIS EXISTS
The hand-written bibliography carried four defects that reached the compiled PDF:

  freitag2024wmt      author = {Freitag, Markus and others}      -> printed "and 1 others"
  rei2022cometkiwi    author = {..., Guerreiro, Nuno M. and others} -> printed "and 1 others"
  popovic2015chrf     cited for BOTH chrF and chrF++. chrF is Popovic 2015; chrF++ is
                      Popovic 2017 (WMT), a different paper. The paper cited the wrong one.
  sindhujan2026beyond note = {Introduces the first English--Malayalam ...} -> an editorial
                      remark printed inside the reference list, which is not a bib field.

Several other entries were also missing editors, DOIs, ISBNs and exact page ranges, which the
ACL style prints and reviewers notice.

HOW IT WORKS
Every reference that has an ACL Anthology record is now taken VERBATIM from the Anthology's own
anthology.bib, keyed by the Anthology's own citation key. Nothing is retyped, so author lists,
diacritics, page ranges and DOIs cannot drift. Entries with no Anthology record (arXiv preprints,
a book, a software library, a Tradumatica article) live in MANUAL below and are the only hand-
maintained text in the bibliography.

The dump is 73 MB and is NOT committed. Fetch it once:

    curl -o refs/anthology.bib https://aclanthology.org/anthology.bib

Then:

    python3 analysis/make_bib.py            # rewrite paper/latex/custom.bib
    python3 analysis/make_bib.py --check    # fail if custom.bib has drifted (CI / run_all_checks)

--check works WITHOUT the dump if custom.bib is already in place and consistent: it verifies the
tex's citations against the bib and the absence of the four defects above, and only diffs against
a freshly extracted copy when the dump is present.

Set ANTHOLOGY_BIB to point somewhere else.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

OUT = _PAPER_TEX.parent / "custom.bib"   # sibling of the .tex, wherever that resolved to
TEX = _PAPER_TEX
DUMP = Path(os.environ.get("ANTHOLOGY_BIB", ROOT / "refs" / "anthology.bib"))

# ---------------------------------------------------------------------------------------------
# Anthology keys. These ARE the citation keys used in the tex, so there is no mapping table to
# get wrong. Grouped by what they are cited for.
# ---------------------------------------------------------------------------------------------
ANTHOLOGY = [
    # metrics and meta-evaluation
    "rei-etal-2020-comet",            # COMET
    "rei-etal-2022-cometkiwi",        # CometKiwi (wmt22-cometkiwi-da)
    "rei-etal-2023-scaling",          # CometKiwi-XL
    "freitag-etal-2024-llms",         # WMT24 metrics task
    "deutsch-etal-2023-ties",         # pairwise accuracy with tie calibration
    "snover-etal-2006-study",         # TER / HTER
    "wang-etal-2016-character",       # CharacTER
    "popovic-2015-chrf",              # chrF
    "popovic-2017-chrf",              # chrF++  (was wrongly cited to the 2015 paper)
    "sai-b-etal-2023-indicmt",        # IndicMT Eval
    "guerreiro-etal-2024-xcomet",     # XCOMET
    "juraska-etal-2024-metricx",      # MetricX-24
    "post-2018-call",                 # sacreBLEU: the signatures in the settings appendix
    "freitag-etal-2021-experts",      # why WMT moved metric meta-evaluation from DA to MQM
    # QE shared-task lineage
    # NOT specia-etal-2020-findings: that key is the WMT20 MT ROBUSTNESS findings paper.
    # The QE findings paper carries the -wmt suffix. The paper cites it for the word-level
    # tag derivation rule and for the MLQE-PE lineage, so the wrong one would be a
    # substantive miscitation, not a formatting slip.
    "specia-etal-2020-findings-wmt",  # WMT20 QE
    "specia-etal-2021-findings",      # WMT21 QE (not currently cited)
    "zerva-etal-2022-findings",       # WMT22 QE
    "blain-etal-2023-findings",       # WMT23 QE
    "zerva-etal-2024-findings",       # WMT24 QE (incl. the APE subtask)
    "fomicheva-etal-2022-mlqe",       # MLQE-PE
    # APE lineage
    "bhattacharyya-etal-2022-findings",
    "bhattacharyya-etal-2023-findings",
    "deoghare-etal-2024-together",
    # prompted QE
    "kocmi-federmann-2023-large",     # GEMBA
    # the En-Ml data: the EAMT project report documents the curation and annotation protocol
    "sindhujan-etal-2025-prompt",
    # provenance of the machine translations, and the linguistic interpretation of the per-pair
    # and per-language results
    "ramesh-etal-2022-samanantar",           # IndicTrans / Samanantar: the en-mr hypotheses
    "guzman-etal-2019-flores",               # the ne-en and si-en low-resource setting
    "kakwani-etal-2020-indicnlpsuite",       # per-language types and tokens, IndicNLP as standard
    "kunchukuttan-bhattacharyya-2016-orthographic",  # Marathi as the agglutinative Indo-Aryan
    "chung-gildea-2009-unsupervised",        # the orthographic word is too coarse for agglutinative
    "popovic-ney-2011-towards",              # inflectional error rate: wrong form of the right lemma
    # held for later use, not cited in the current draft
    "thompson-etal-2024-improving",   # soft pairwise accuracy
    "kiritchenko-mohammad-2017-best", # best-worst scaling
    "kanojia-etal-2020-challenge",
    "kanojia-etal-2020-harnessing",
    "kanojia-etal-2021-cognition",
]

# ---------------------------------------------------------------------------------------------
# No Anthology record exists for these, so they stay hand-maintained. Keep this block small.
# ---------------------------------------------------------------------------------------------
MANUAL = r"""
@misc{nllb2022,
  title         = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author        = {{NLLB Team}},
  year          = {2022},
  eprint        = {2207.04672},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2207.04672}
}

@article{gala2023indictrans2,
  title   = {{IndicTrans2}: Towards High-Quality and Accessible Machine Translation Models for all
             $22$ Scheduled {I}ndian Languages},
  author  = {Gala, Jay and Chitale, Pranjal A. and Raghavan, A. K. and Gumma, Varun and Doddapaneni,
             Sumanth and Kumar, Aswanth and Nawale, Janki and Sujatha, Anupama and Puduppully, Ratish
             and Raghavan, Vivek and Kumar, Pratyush and Khapra, Mitesh M. and Dabre, Raj and
             Kunchukuttan, Anoop},
  journal = {Transactions on Machine Learning Research},
  year    = {2023},
  note    = {arXiv:2305.16307},
  url     = {https://arxiv.org/abs/2305.16307}
}

@article{kumar2007telugu,
  title   = {Statistical analyses of {T}elugu text corpora},
  author  = {Kumar, G. Bharadwaja and Murthy, Kavi Narayana and Chaudhuri, B. B.},
  journal = {International Journal of Dravidian Linguistics},
  volume  = {36},
  number  = {2},
  pages   = {71--99},
  year    = {2007}
}

% Peer-reviewed at COLM 2025 (OpenReview qSFr5wJPGc), not an arXiv preprint. COLM has no ACL
% Anthology record, so this stays a manual entry; the arXiv id is kept as a note.
@inproceedings{sindhujan2025alope,
  title     = {{ALOPE}: Adaptive Layer Optimization for Translation Quality Estimation using Large Language Models},
  author    = {Sindhujan, Archchana and Qian, Shenbin and Chan, Chi Chun Matthew and Or{\u{a}}san, Constantin and Kanojia, Diptesh},
  booktitle = {Proceedings of the Second Conference on Language Modeling (COLM)},
  year      = {2025},
  note      = {arXiv:2508.07484},
  url       = {https://openreview.net/forum?id=qSFr5wJPGc}
}

@misc{sindhujan2026beyond,
  title         = {Beyond Scalar Scores: Reinforcement Learning for Error-Aware Quality Estimation
                   of Machine Translation},
  author        = {Sindhujan, Archchana and Koushik, Girish A. and Qian, Shenbin and
                   Kanojia, Diptesh and Or{\u{a}}san, Constantin},
  year          = {2026},
  eprint        = {2602.08600},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.08600}
}

@article{lommel2014mqm,
  title   = {Multidimensional Quality Metrics ({MQM}): A Framework for Declaring and Describing Translation Quality Metrics},
  author  = {Lommel, Arle and Uszkoreit, Hans and Burchardt, Aljoscha},
  journal = {Tradum{\`a}tica: tecnologies de la traducci{\'o}},
  volume  = {12},
  pages   = {455--463},
  year    = {2014}
}

@article{he2024qereward,
  title         = {Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model},
  author        = {He, Zhiwei and Wang, Xing and Jiao, Wenxiang and Zhang, Zhuosheng and Wang, Rui and Shi, Shuming and Tu, Zhaopeng},
  journal       = {arXiv preprint arXiv:2401.12873},
  year          = {2024},
  eprint        = {2401.12873},
  archivePrefix = {arXiv}
}

@misc{kunchukuttan2020indicnlp,
  title        = {The {IndicNLP} Library},
  author       = {Kunchukuttan, Anoop},
  year         = {2020},
  howpublished = {\url{https://github.com/anoopkunchukuttan/indic_nlp_library}}
}

@book{subbarao2012south,
  title     = {South {A}sian Languages: A Syntactic Typology},
  author    = {Subb{\={a}}r{\={a}}o, Karumuri V.},
  year      = {2012},
  publisher = {Cambridge University Press},
  isbn      = {9780521861489}
}
"""

HEADER = """% IndicQE-APE bibliography. GENERATED FILE -- do not hand-edit.
%
% Every entry below the ANTHOLOGY marker is copied verbatim from https://aclanthology.org/anthology.bib
% by analysis/make_bib.py, keyed by the Anthology's own citation key. To add a reference that the
% Anthology publishes, add its key to ANTHOLOGY in that script and re-run; do not paste it here.
% Entries under the MANUAL marker have no Anthology record and are the only hand-maintained text.
%
%   curl -o refs/anthology.bib https://aclanthology.org/anthology.bib
%   python3 analysis/make_bib.py
%
% The two string definitions below are the Anthology's own and the verbatim entries need them.
% Note for anyone editing this header: never write an at-sign in a comment here. BibTeX starts a
% new command at an at-sign even inside what looks like a comment line, so it breaks the parse.

@string{acl = {Association for Computational Linguistics}}
@string{anth = {https://aclanthology.org/}}
"""

BAD_PATTERNS = [
    (r"and others\}", "an 'and others' author list, which prints as 'and 1 others'"),
    (r"note\s*=\s*\{Introduces", "an editorial remark in a note field"),
    # BibTeX starts a command at an at-sign even inside a comment line, so an at-sign in a comment
    # is a parse error, not a comment. This cost one build.
    (r"(?m)^%.*@", "an at-sign inside a comment line, which BibTeX reads as a new command"),
]



def comment_inside_entry(bib: str) -> list[str]:
    """A '%' between an entry's braces silently truncates it.

    BibTeX has no comment syntax INSIDE an entry: the parser is looking for `field = value` and a
    '%' line makes it give up on the rest. A comment between entries is fine, so this cannot be a
    regex; it needs the brace depth. Found the hard way, by annotating lommel2014mqm in place and
    losing its year and journal to a warning that named the entry but not the cause.
    """
    out, depth = [], 0
    for n, line in enumerate(bib.splitlines(), 1):
        if depth > 0 and line.lstrip().startswith("%"):
            out.append(f"a comment inside a bib entry at line {n}, which truncates it: "
                       f"{line.strip()[:60]!r}")
        depth = max(0, depth + line.count("{") - line.count("}"))
    return out


def cited_keys(tex: str) -> set[str]:
    """Keys actually cited, ignoring commented-out lines."""
    live = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    keys: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]*)\}", live):
        keys.update(k.strip() for k in group.split(",") if k.strip())
    return keys


def extract(dump: Path) -> dict[str, str]:
    """Pull the requested entries verbatim. The Anthology writes one entry per line."""
    want = set(ANTHOLOGY)
    found: dict[str, str] = {}
    with dump.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("@"):
                continue
            m = re.match(r"@[a-zA-Z]+\{([^,]+),", line)
            if m and m.group(1) in want:
                key = m.group(1)
                if key in found:
                    raise SystemExit(f"duplicate Anthology entry for {key}")
                found[key] = line.rstrip("\n")
    missing = want - set(found)
    if missing:
        raise SystemExit(f"not in {dump}: {sorted(missing)}")
    return found


def render(found: dict[str, str]) -> str:
    body = ["", "% ==== ANTHOLOGY (verbatim; regenerate, do not edit) ===================="]
    for key in sorted(found):
        body.append(found[key])
    body += ["", "% ==== MANUAL (no Anthology record) ====================================", MANUAL.strip(), ""]
    return HEADER + "\n".join(body) + "\n"


def audit(bib: str, tex: str) -> list[str]:
    problems = []
    keys = set(re.findall(r"^@[a-zA-Z]+\{([^,]+),", bib, flags=re.M))
    for k in sorted(cited_keys(tex) - keys):
        problems.append(f"cited but not in custom.bib: {k}")
    for pat, why in BAD_PATTERNS:
        for hit in re.findall(pat, bib):
            problems.append(f"bibliography contains {why}: {hit!r}")
    problems += comment_inside_entry(bib)
    dupes = [k for k in keys if len(re.findall(r"^@[a-zA-Z]+\{" + re.escape(k) + ",", bib, flags=re.M)) > 1]
    problems += [f"duplicate key: {k}" for k in sorted(set(dupes))]
    return problems


def main() -> int:
    check = "--check" in sys.argv
    tex = TEX.read_text()

    if check:
        current = OUT.read_text()
        problems = audit(current, tex)
        if DUMP.exists():
            if render(extract(DUMP)) != current:
                problems.append(f"custom.bib differs from a fresh extraction; run: python3 {Path(__file__).name}")
        else:
            print(f"  note: {DUMP} absent, skipping the verbatim diff")
        for p in problems:
            print(f"FAIL: {p}")
        if problems:
            return 1
        print(f"custom.bib OK ({len(cited_keys(tex))} keys cited, all present, no malformed entries)")
        return 0

    if not DUMP.exists():
        raise SystemExit(f"{DUMP} not found. curl -o {DUMP} https://aclanthology.org/anthology.bib")
    found = extract(DUMP)
    out = render(found)
    problems = audit(out, tex)
    for p in problems:
        print(f"FAIL: {p}")
    if problems:
        print("\nrefusing to write")
        return 1
    OUT.write_text(out)
    # OUT is a sibling of the .tex, and the paper lives in its own repository, so it is usually
    # NOT under ROOT and relative_to would raise. Show the short form only when it applies.
    try:
        where = OUT.relative_to(ROOT)
    except ValueError:
        where = OUT
    print(f"wrote {where}: {len(found)} Anthology entries (verbatim) + manual block")
    print(f"  cited in the tex: {len(cited_keys(tex))} keys, all resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
