from __future__ import annotations

"""
predict.py

Loads a trained checkpoint and writes predictions in the shared-task format:
item_id,prediction
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from data_utils import (
    DataConfig,
    add_text_column,
    dataframe_to_prediction_dataset,
    load_dataframe,
    normalize_model_name,
    resolve_input_columns,
    split_dataframe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or final model directory")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="Which split to predict")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    cfg = load_config(config_path)
    experiment_name = cfg.get("experiment_name", config_path.stem)
    model_name = normalize_model_name(cfg["model_name"])
    max_length = int(cfg.get("max_length", 256))

    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path

    data_cfg = DataConfig(
        data_path=str(data_path),
        split_column=cfg.get("split_column", "subset"),
        train_split=cfg.get("train_split", "train"),
        dev_split=cfg.get("dev_split", "dev"),
        test_split=cfg.get("test_split", "test"),
        item_id_column=cfg.get("item_id_column", "item_id"),
        label_column=cfg.get("label_column", "GLMM_score"),
        input_columns=cfg.get("input_columns"),
        exclude_columns=cfg.get("exclude_columns"),
        cefr_column=cfg.get("cefr_column", "CEFR_level"),
        encode_cefr=bool(cfg.get("encode_cefr", False)),
        missing_cefr_token=cfg.get("missing_cefr_token", "<MISSING_CEFR>"),
    )

    df = load_dataframe(data_cfg.data_path)
    splits = split_dataframe(df, data_cfg)
    pred_df = splits[args.split]
    if pred_df.empty:
        raise ValueError(f"Requested split '{args.split}' is empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    sep_token = tokenizer.sep_token or "[SEP]"
    input_columns = resolve_input_columns(data_cfg.input_columns, data_cfg.exclude_columns)
    pred_df = add_text_column(pred_df, sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)
    pred_ds = dataframe_to_prediction_dataset(pred_df, data_cfg.item_id_column)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    pred_ds = pred_ds.map(tokenize_function, batched=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
    trainer = Trainer(model=model)
    raw_preds = trainer.predict(pred_ds.remove_columns([data_cfg.item_id_column, "text"]))
    preds = np.squeeze(raw_preds.predictions)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
    else:
        out_path = REPO_ROOT / "submissions" / f"{experiment_name}_{args.split}_predictions.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "item_id": pred_df[data_cfg.item_id_column],
            "prediction": preds.astype(float),
        }
    )
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
