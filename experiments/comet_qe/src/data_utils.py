from __future__ import annotations

"""
data_utils.py

Data loading, text construction with character offsets, and feature engineering
for the BEA 2026 Lexical Difficulty Prediction Framework.

Team SurreyCTS
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
# NOTE: `Levenshtein` is imported lazily inside prepare_features (the BEA lexical
# path). The reference-free QE path does not use it, so it must not be a hard
# module-level dependency.

from model_registry import normalize_model_name


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str
    split_column: str = "subset"
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"
    item_id_column: str = "item_id"
    label_column: str = "GLMM_score"
    input_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    cefr_column: str = "CEFR_level"
    encode_cefr: bool = False
    missing_cefr_token: str = "<MISSING_CEFR>"
    prepend_lid: bool = False


DEFAULT_INPUT_COLUMNS = [
    "L1_context",
    "L1_source_word",
    "en_target_clue",
    "en_target_pos",
    "en_target_word",
]


def safe_text(value) -> str:
    """Convert a value into clean text for the model input."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_input_columns(
    input_columns: Optional[List[str]],
    exclude_columns: Optional[List[str]],
) -> List[str]:
    """Return the final ordered list of columns after applying ablations."""
    columns = list(input_columns) if input_columns else list(DEFAULT_INPUT_COLUMNS)
    exclude = set(exclude_columns or [])
    return [col for col in columns if col not in exclude]


def format_cefr_value(value, missing_token: str = "<MISSING_CEFR>") -> str:
    """Convert CEFR values into categorical tokens for the transformer."""
    if pd.isna(value):
        return missing_token
    cleaned = str(value).strip().upper()
    if not cleaned:
        return missing_token
    allowed = {"A1", "A2", "B1", "B2", "C1", "C2"}
    return f"CEFR_{cleaned}" if cleaned in allowed else missing_token


def build_input_text_with_offsets(
    row: pd.Series,
    sep_token: str,
    columns: List[str],
    encode_cefr: bool = False,
    cefr_column: str = "CEFR_level",
    missing_cefr_token: str = "<MISSING_CEFR>",
    prepend_lid: bool = False,
):
    """Build model input text from DataFrame columns with character-level offsets.

    Tracks the character positions of target_word, L1_context, and L1_source_word
    for downstream token-level mask construction.
    """
    segments: List[str] = []
    target_start, target_end = -1, -1
    context_start, context_end = -1, -1
    source_start, source_end = -1, -1
    current_char_idx = 0

    if prepend_lid:
        text = f"[L1: {str(row.get('L1', '')).strip()}]"
        segments.append(text)
        current_char_idx += len(text) + len(sep_token) + 2

    for col in columns:
        value = row.get(col, "")
        if encode_cefr and col == cefr_column:
            text = format_cefr_value(value, missing_token=missing_cefr_token)
        else:
            text = safe_text(value)

        if not text:
            continue

        # Feature-column formatting
        if col == "subword_ratio":
            try:
                text = f"SWR_{float(text):.3f}"
            except Exception:
                text = "SWR_MISSING"
        elif col == "subword_count":
            try:
                text = f"SWC_{int(float(text))}"
            except Exception:
                text = "SWC_MISSING"
        elif col == "char_len":
            try:
                text = f"LEN_{int(float(text))}"
            except Exception:
                text = "LEN_MISSING"

        # Track offsets for key spans
        if col == "en_target_word":
            target_start = current_char_idx
            target_end = current_char_idx + len(text)
        elif col == "L1_context":
            context_start = current_char_idx
            context_end = current_char_idx + len(text)
        elif col == "L1_source_word":
            source_start = current_char_idx
            source_end = current_char_idx + len(text)

        segments.append(text)
        current_char_idx += len(text) + len(sep_token) + 2

    joined_text = f" {sep_token} ".join(segments)
    return (joined_text, target_start, target_end,
            context_start, context_end, source_start, source_end)


def load_dataframe(data_path: str) -> pd.DataFrame:
    """Load a CSV file containing train/dev/test rows."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")
    return pd.read_csv(path)


def split_dataframe(df: pd.DataFrame, cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    """Split the full DataFrame into train/dev/test according to the subset column."""
    if cfg.split_column not in df.columns:
        raise ValueError(
            f"Split column '{cfg.split_column}' not found. "
            f"Available: {list(df.columns)}"
        )
    return {
        "train": df[df[cfg.split_column] == cfg.train_split].copy(),
        "dev": df[df[cfg.split_column] == cfg.dev_split].copy(),
        "test": df[df[cfg.split_column] == cfg.test_split].copy(),
    }


def add_text_column(
    df: pd.DataFrame,
    sep_token: str,
    input_columns: List[str],
    cfg: Optional[DataConfig] = None,
) -> pd.DataFrame:
    """Create the text field and character offsets consumed by the tokenizer."""
    out = df.copy()
    encode_cefr = cfg.encode_cefr if cfg else False
    cefr_column = cfg.cefr_column if cfg else "CEFR_level"
    missing_cefr_token = cfg.missing_cefr_token if cfg else "<MISSING_CEFR>"
    prepend_lid = getattr(cfg, "prepend_lid", False) if cfg else False

    res = out.apply(
        lambda row: build_input_text_with_offsets(
            row, sep_token=sep_token, columns=input_columns,
            encode_cefr=encode_cefr, cefr_column=cefr_column,
            missing_cefr_token=missing_cefr_token, prepend_lid=prepend_lid,
        ),
        axis=1,
    )
    out["text"] = [x[0] for x in res]
    out["target_start"] = [x[1] for x in res]
    out["target_end"] = [x[2] for x in res]
    out["context_start"] = [x[3] for x in res]
    out["context_end"] = [x[4] for x in res]
    out["source_start"] = [x[5] for x in res]
    out["source_end"] = [x[6] for x in res]
    return out


def prepare_features(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    use_normalized_levenshtein: bool = False,
    use_jaccard: bool = False,
    use_pinyin: bool = False,
):
    """Compute string similarity, CEFR indices, LID indices, and subword ratio.

    String distances are standard-scaled based on the train split.
    """
    dfs = [train_df, dev_df]
    if test_df is not None:
        dfs.append(test_df)

    for df in dfs:
        def calc_distance(row):
            is_cn = str(row.get("L1", "")).lower() == "cn"
            source = str(row.get("L1_source_word", ""))
            target = str(row.get("en_target_word", ""))

            # Pinyin romanization for Chinese
            if is_cn and use_pinyin:
                try:
                    from pypinyin import pinyin, Style
                    res = pinyin(source, style=Style.NORMAL)
                    source = "".join([x[0] for x in res])
                except ImportError:
                    pass

            # Distance metric
            if use_jaccard:
                def bigrams(s):
                    if len(s) < 2:
                        return set([s])
                    return set(s[i:i+2] for i in range(len(s) - 1))
                b1, b2 = bigrams(source.lower()), bigrams(target.lower())
                if not b1 and not b2:
                    dist = 0.0
                else:
                    dist = 1.0 - (len(b1 & b2) / len(b1 | b2))
            else:
                import Levenshtein
                dist = float(Levenshtein.distance(source, target))
                if use_normalized_levenshtein:
                    m = max(len(source), len(target))
                    dist = dist / m if m > 0 else 0.0

            # Chinese without pinyin romanization: Levenshtein is meaningless
            if is_cn and not use_pinyin and not use_jaccard:
                return 0.0
            return dist

        df["lev_score"] = df.apply(calc_distance, axis=1)
        df["subword_ratio"] = df.apply(
            lambda x: float(x.get("subword_ratio", 0.0)), axis=1
        )

    # Standard scale on train
    scaler = StandardScaler()
    train_df["lev_score_scaled"] = scaler.fit_transform(train_df[["lev_score"]])
    dev_df["lev_score_scaled"] = scaler.transform(dev_df[["lev_score"]])
    if test_df is not None:
        test_df["lev_score_scaled"] = scaler.transform(test_df[["lev_score"]])

    # Categorical indices
    cefr_map = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    lid_map = {"es": 0, "de": 1, "cn": 2}

    for df in dfs:
        if "CEFR_level" in df.columns:
            df["cefr_indices"] = df["CEFR_level"].apply(
                lambda v: cefr_map.get(str(v).strip().upper(), 6)
            )
        else:
            df["cefr_indices"] = 6  # Default category for unknown/missing

        df["lid_indices"] = df["L1"].apply(
            lambda v: lid_map.get(str(v).strip().lower(), 3)
        )


# ---------------------------------------------------------------------------
# QE (reference-free MT quality estimation) input + features
# ---------------------------------------------------------------------------

def build_qe_input_with_offsets(
    source,
    mt,
    sep_token: str,
    order: str = "src_first",
):
    """Build the QE input string ``source </s> mt`` (or mt-first) with offsets.

    Returns the same 7-tuple shape as :func:`build_input_text_with_offsets` so the
    QE path can reuse ``tokenize_function`` unchanged::

        (text, target_start, target_end, context_start, context_end,
         source_start, source_end)

    The **target** span is the MT hypothesis (the text being judged); the
    **source** span is the source sentence; there is no context span, so its
    offsets are ``(-1, -1)``.
    """
    src = safe_text(source)
    hyp = safe_text(mt)
    first, second = (hyp, src) if order == "mt_first" else (src, hyp)
    joiner = f" {sep_token} "
    text = f"{first}{joiner}{second}"

    first_start, first_end = 0, len(first)
    second_start = len(first) + len(joiner)
    second_end = second_start + len(second)

    if order == "mt_first":
        target_start, target_end = first_start, first_end
        source_start, source_end = second_start, second_end
    else:
        source_start, source_end = first_start, first_end
        target_start, target_end = second_start, second_end

    return (text, target_start, target_end, -1, -1, source_start, source_end)


def add_qe_text_column(
    df: pd.DataFrame,
    sep_token: str,
    source_col: str = "source",
    mt_col: str = "mt",
    order: str = "src_first",
) -> pd.DataFrame:
    """Create the text field and char offsets for the QE path (mirrors
    :func:`add_text_column`)."""
    out = df.copy()
    res = out.apply(
        lambda row: build_qe_input_with_offsets(
            row.get(source_col, ""), row.get(mt_col, ""),
            sep_token=sep_token, order=order,
        ),
        axis=1,
    )
    out["text"] = [x[0] for x in res]
    out["target_start"] = [x[1] for x in res]
    out["target_end"] = [x[2] for x in res]
    out["context_start"] = [x[3] for x in res]
    out["context_end"] = [x[4] for x in res]
    out["source_start"] = [x[5] for x in res]
    out["source_end"] = [x[6] for x in res]
    return out


def _length_ratio(source, mt, unit: str = "token") -> float:
    """|mt| / |src| in tokens (whitespace) or characters; 0.0 if source empty."""
    s = safe_text(source)
    m = safe_text(mt)
    if unit == "char":
        denom, numer = len(s), len(m)
    else:
        denom, numer = len(s.split()), len(m.split())
    return (numer / denom) if denom > 0 else 0.0


def prepare_qe_features(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    source_col: str = "source",
    mt_col: str = "mt",
    length_unit: str = "token",
):
    """Compute the one QE surface feature, ``length_ratio`` = |mt| / |src|.

    Standard-scaled on the train split into ``length_ratio_scaled`` (mirrors the
    BEA ``lev_score_scaled`` treatment). No BEA lexical/CEFR/LID features are
    produced. Mutates the frames in place and returns the fitted scaler.
    """
    dfs = [train_df, dev_df]
    if test_df is not None:
        dfs.append(test_df)

    for df in dfs:
        df["length_ratio"] = df.apply(
            lambda r: _length_ratio(r.get(source_col, ""), r.get(mt_col, ""), unit=length_unit),
            axis=1,
        )

    scaler = StandardScaler()
    train_df["length_ratio_scaled"] = scaler.fit_transform(train_df[["length_ratio"]])
    dev_df["length_ratio_scaled"] = scaler.transform(dev_df[["length_ratio"]])
    if test_df is not None:
        test_df["length_ratio_scaled"] = scaler.transform(test_df[["length_ratio"]])
    return scaler


def qe_dataframe_to_dataset(
    df: pd.DataFrame,
    label_column: str = "target",
    use_length_ratio: bool = False,
    use_lp: bool = False,
) -> Dataset:
    """Convert a QE DataFrame into a HuggingFace Dataset.

    Requires only the QE columns (text + char-offset spans + label), plus the
    optional ``length_ratio_scaled`` / ``lp_indices`` when those features are
    enabled. Deliberately does NOT require the BEA lexical/CEFR/LID columns.
    """
    required = [
        "text", "target_start", "target_end",
        "context_start", "context_end",
        "source_start", "source_end",
        label_column,
    ]
    if use_length_ratio:
        required.append("length_ratio_scaled")
    if use_lp:
        required.append("lp_indices")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required QE columns: {missing}")
    return Dataset.from_pandas(df[required], preserve_index=False).rename_column(
        label_column, "labels"
    )


def dataframe_to_dataset(
    df: pd.DataFrame,
    label_column: str,
    include_embeddings: bool = False,
) -> Dataset:
    """Convert a labelled DataFrame into a HuggingFace Dataset."""
    required = [
        "text", "target_start", "target_end",
        "context_start", "context_end",
        "source_start", "source_end",
        "lev_score_scaled", "cefr_indices", "lid_indices",
        label_column,
    ]
    if "subword_ratio" in df.columns:
        required.append("subword_ratio")
    if include_embeddings:
        required.append("dense_embedding")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return Dataset.from_pandas(df[required], preserve_index=False).rename_column(
        label_column, "labels"
    )


def dataframe_to_prediction_dataset(
    df: pd.DataFrame, item_id_column: str
) -> Dataset:
    """Convert a DataFrame into a HuggingFace Dataset for inference."""
    required = ["text", item_id_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return Dataset.from_pandas(df[[item_id_column, "text"]], preserve_index=False)
