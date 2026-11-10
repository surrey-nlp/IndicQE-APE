"""
evaluate_test.py

Evaluate a trained checkpoint on any data split.

Usage:
  # Use config's data file, eval on test split:
  python src/evaluate_test.py --config configs/sweeps_v2/rembert-huber.yaml --checkpoint outputs/v2-rembert-huber/checkpoint-1234

  # Override data file (e.g., new test set):
  python src/evaluate_test.py --config configs/sweeps_v2/rembert-huber.yaml --checkpoint outputs/v2-rembert-huber/checkpoint-1234 --data data/new_test.csv

  # Evaluate on dev split instead:
  python src/evaluate_test.py --config configs/sweeps_v2/rembert-huber.yaml --checkpoint outputs/v2-rembert-huber/checkpoint-1234 --split dev

Team SurreyCTS — BEA 2026 Lexical Difficulty Prediction Framework
"""

import os
import argparse
import torch
import json
import yaml
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
)

from data_utils import (
    DataConfig,
    add_text_column,
    add_qe_text_column,
    dataframe_to_dataset,
    qe_dataframe_to_dataset,
    load_dataframe,
    prepare_features,
    prepare_qe_features,
    resolve_input_columns,
    split_dataframe,
)
from qe_data import fit_perpair_scaler, inverse_transform
from losses import get_loss_function
from hybrid_model import BEA2026HybridRegressor
from train import CustomLossTrainer, CustomDataCollator, compute_metrics
from model_registry import normalize_model_name

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on any data split."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file used for training")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--data", default=None, help="Override data CSV path (e.g., a new test set)")
    parser.add_argument("--split", default=None, help="Which split to evaluate on: train/dev/test (default: test, fallback: dev)")
    parser.add_argument("--output", default=None, help="Path to save results JSON (default: checkpoint dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_config(config_path)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    # Data path: CLI override or config
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path

    model_name = normalize_model_name(cfg["model_name"])
    seed = int(cfg.get("seed", 42))
    max_length = int(cfg.get("max_length", 256))
    eval_batch_size = int(cfg.get("eval_batch_size", 16))

    loss_name = cfg.get("loss_function", "mse")
    loss_kwargs = cfg.get("loss_kwargs", {})
    loss_fn, loss_kwargs = get_loss_function(loss_name, loss_kwargs)
    use_custom_loss = loss_name.lower() != "mse"

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
        prepend_lid=bool(cfg.get("prepend_lid", False)),
    )

    set_seed(seed)
    df = load_dataframe(data_cfg.data_path)
    splits = split_dataframe(df, data_cfg)

    # Determine eval split
    if args.split:
        eval_name = args.split
        if eval_name not in splits or splits[eval_name].empty:
            raise ValueError(f"Requested split '{eval_name}' is empty or doesn't exist in {data_path}")
        eval_df = splits[eval_name]
    elif not splits["test"].empty:
        eval_df = splits["test"]
        eval_name = "test"
    else:
        print("Warning: Test split is empty. Evaluating on Dev split instead.")
        eval_df = splits["dev"]
        eval_name = "dev"

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Evaluation Configuration                                  ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Checkpoint: {str(checkpoint_path)[:47]:>47} ║")
    print(f"║  Data:       {str(data_path)[:47]:>47} ║")
    print(f"║  Split:      {eval_name:>47} ║")
    print(f"║  Samples:    {len(eval_df):>47} ║")
    print(f"║  Model:      {model_name[:47]:>47} ║")
    print(f"║  Hybrid:     {str(cfg.get('use_hybrid_model', False)):>47} ║")
    print(f"║  Deep MLP:   {str(cfg.get('deep_mlp', False)):>47} ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), use_fast=True)
    sep_token = tokenizer.sep_token or "[SEP]"

    task = cfg.get("task", "bea")
    group_column = cfg.get("group_column", "language_pair" if task == "qe" else "L1")
    qe_use_length_ratio = bool(cfg.get("use_length_ratio", False))
    qe_use_lp = bool(cfg.get("use_lp", False))
    qe_scaler = {}
    lp_map = {}

    if task == "qe":
        source_col = cfg.get("source_column", "source")
        mt_col = cfg.get("mt_column", "mt")
        order = cfg.get("segment_order", "src_first")
        train_df = add_qe_text_column(splits["train"], sep_token=sep_token, source_col=source_col, mt_col=mt_col, order=order)
        dev_df = add_qe_text_column(splits["dev"], sep_token=sep_token, source_col=source_col, mt_col=mt_col, order=order)
        eval_text_df = add_qe_text_column(eval_df, sep_token=sep_token, source_col=source_col, mt_col=mt_col, order=order)
        prepare_qe_features(train_df, dev_df, eval_text_df, source_col=source_col, mt_col=mt_col,
                            length_unit=cfg.get("length_unit", "token"))
        # Rebuild the per-pair scaler from train (identical to train time) so we can
        # inverse-transform predictions back to the raw DA scale for the jsonl.
        qe_scaler = fit_perpair_scaler(splits["train"], label="da_mean", group=group_column)
        if qe_use_lp:
            lp_values = sorted(train_df[group_column].dropna().unique())
            lp_map = {p: i for i, p in enumerate(lp_values)}
            eval_text_df["lp_indices"] = eval_text_df[group_column].map(lambda p: lp_map.get(p, 0)).astype(int)
        input_columns = []
    else:
        input_columns = resolve_input_columns(data_cfg.input_columns, data_cfg.exclude_columns)
        # Need train for scaler fitting
        train_df = add_text_column(splits["train"], sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)
        dev_df = add_text_column(splits["dev"], sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)
        eval_text_df = add_text_column(eval_df, sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)

        prepare_features(
            train_df, dev_df, eval_text_df,
            use_normalized_levenshtein=bool(cfg.get("use_normalized_levenshtein", False)),
            use_jaccard=bool(cfg.get("use_jaccard", False)),
            use_pinyin=bool(cfg.get("use_pinyin", False)),
        )

    embeddings_path = cfg.get("embeddings_path", None)
    include_embeddings = False
    if embeddings_path is not None:
        emb_path = Path(embeddings_path)
        if not emb_path.is_absolute():
            emb_path = REPO_ROOT / emb_path
        emb_dict = torch.load(emb_path, map_location="cpu", weights_only=True)
        default_emb = [0.0] * next(iter(emb_dict.values())).shape[-1]
        eval_text_df["dense_embedding"] = eval_text_df[data_cfg.item_id_column].map(
            lambda x: emb_dict.get(x).tolist() if x in emb_dict else default_emb
        )
        include_embeddings = True

    if task == "qe":
        eval_ds = qe_dataframe_to_dataset(
            eval_text_df, data_cfg.label_column,
            use_length_ratio=qe_use_length_ratio, use_lp=qe_use_lp,
        )
    else:
        eval_ds = dataframe_to_dataset(eval_text_df, data_cfg.label_column, include_embeddings=include_embeddings)

    def tokenize_function(batch):
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        target_masks, context_masks, source_masks = [], [], []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            tgt_s, tgt_e = batch["target_start"][i], batch["target_end"][i]
            ctx_s, ctx_e = batch["context_start"][i], batch["context_end"][i]
            src_s, src_e = batch["source_start"][i], batch["source_end"][i]
            t_mask, c_mask, s_mask = [], [], []
            for off_s, off_e in offsets:
                is_special = (off_s == 0 and off_e == 0) or off_e == off_s
                if is_special:
                    t_mask.append(0); c_mask.append(0); s_mask.append(0)
                    continue
                def overlaps(span_s, span_e):
                    return (off_s >= span_s and off_e <= span_e) or (off_e > span_s and off_s < span_e)
                t_mask.append(1 if overlaps(tgt_s, tgt_e) else 0)
                c_mask.append(1 if overlaps(ctx_s, ctx_e) else 0)
                s_mask.append(1 if overlaps(src_s, src_e) else 0)
            if sum(t_mask) == 0: t_mask[0] = 1
            if sum(c_mask) == 0: c_mask[0] = 1
            if sum(s_mask) == 0: s_mask[0] = 1
            target_masks.append(t_mask); context_masks.append(c_mask); source_masks.append(s_mask)
        tokenized["target_mask"] = target_masks
        tokenized["context_mask"] = context_masks
        tokenized["source_mask"] = source_masks
        del tokenized["offset_mapping"]
        return tokenized

    eval_ds = eval_ds.map(tokenize_function, batched=True)

    # ── Model Loading ────────────────────────────────────────────────────
    use_hybrid_model = bool(cfg.get("use_hybrid_model", False))

    if use_hybrid_model:
        hf_config = AutoConfig.from_pretrained(model_name)
        flag_keys = [
            "use_attention", "use_target_context_entropy",
            "use_target_source_alignment", "use_target_source_entropy",
            "use_context_target_entropy", "use_levenshtein", "use_cefr",
            "deep_mlp", "use_lid", "use_dense_embeddings",
            "use_subword_ratio", "multi_layer_funneling",
            "use_multi_layer_entropy", "use_attention_variance",
            "use_global_cls_entropy",
            "use_length_ratio", "use_lp",
        ]
        flag_defaults = {
            "use_attention": True, "use_target_context_entropy": False,
            "use_target_source_alignment": False, "use_target_source_entropy": False,
            "use_context_target_entropy": False, "use_levenshtein": False,
            "use_cefr": False, "deep_mlp": False, "use_lid": False,
            "use_dense_embeddings": False, "use_subword_ratio": False,
            "multi_layer_funneling": False, "use_multi_layer_entropy": False,
            "use_attention_variance": False, "use_global_cls_entropy": False,
            "use_length_ratio": False, "use_lp": False,
        }
        for key in flag_keys:
            setattr(hf_config, key, bool(cfg.get(key, flag_defaults.get(key, False))))
        if "funneling_layers" in cfg:
            hf_config.funneling_layers = cfg["funneling_layers"]
        if "num_funneling_layers" in cfg:
            hf_config.num_funneling_layers = int(cfg["num_funneling_layers"])
        # Language-pair embedding sizing must match training exactly.
        if task == "qe" and qe_use_lp:
            hf_config.num_language_pairs = int(cfg.get("num_language_pairs", len(lp_map)))
            hf_config.lp_dim = int(cfg.get("lp_dim", 8))

        model = BEA2026HybridRegressor(model_name, config=hf_config)
        if (checkpoint_path / "model.safetensors").exists():
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path / "model.safetensors")
        elif (checkpoint_path / "pytorch_model.bin").exists():
            state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_path}")
        model.load_state_dict(state_dict)
        print(f"✓ Loaded hybrid model (MLP input dim: {model.mlp_input_dim})")
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
        print(f"✓ Loaded standard model from {checkpoint_path.name}")

    training_args = TrainingArguments(
        output_dir="/tmp/eval_output",
        per_device_eval_batch_size=eval_batch_size,
        report_to="none",
        fp16=torch.cuda.is_available() and bool(cfg.get("fp16", True)),
    )

    trainer_cls = CustomLossTrainer if use_custom_loss else Trainer
    trainer_kwargs = dict(
        model=model, args=training_args, eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    if use_custom_loss:
        trainer_kwargs["loss_fn"] = loss_fn
        trainer_kwargs["loss_kwargs"] = loss_kwargs

    trainer = trainer_cls(**trainer_kwargs)
    
    # ── Prediction & Evaluation ──────────────────────────────────────────
    print(f"Running predictions on {eval_name} set ({len(eval_df)} samples)...")
    prediction_output = trainer.predict(test_dataset=eval_ds)
    metrics = prediction_output.metrics
    preds = prediction_output.predictions.flatten()
    
    # Sync metrics labels
    renamed = {k.replace("test_", f"{eval_name}_"): v for k, v in metrics.items()}
    if "eval_loss" in renamed: # Handle possible eval_ prefix from trainer.predict
        renamed = {k.replace("eval_", f"{eval_name}_"): v for k, v in renamed.items()}

    # Pretty print results
    print(f"\n{'═'*50}")
    print(f"  Results — {eval_name} split")
    print(f"{'═'*50}")
    for key, val in sorted(renamed.items()):
        if isinstance(val, (int, float)):
            label = key.replace(f"{eval_name}_", "").upper()
            print(f"  {label:<20} {val:>12.6f}")
    print(f"{'═'*50}")

    # ── Save Results ─────────────────────────────────────────────────────
    base_out = Path(args.output) if args.output else checkpoint_path
    if base_out.is_dir():
        metrics_out = base_out / f"eval_{eval_name}_metrics.json"
        preds_out = base_out / f"eval_{eval_name}_predictions.csv"
    else:
        metrics_out = base_out
        preds_out = base_out.with_suffix(".csv")

    # 1. Save Metrics JSON
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(renamed, f, indent=2)
    
    # 2. Save Predictions
    eval_text_df["prediction"] = preds

    if task == "qe":
        # Inverse-transform z-scale predictions back to the raw DA scale per pair.
        eval_text_df["pred_da"] = eval_text_df.apply(
            lambda r: float(inverse_transform(r["prediction"], r[group_column], qe_scaler))
            if r[group_column] in qe_scaler else float(r["prediction"]),
            axis=1,
        )
        qe_cols = [data_cfg.item_id_column, group_column, "da_mean",
                   data_cfg.label_column, "prediction", "pred_da"]
        eval_text_df[[c for c in qe_cols if c in eval_text_df.columns]].to_csv(preds_out, index=False)

        # IndicQE-schema jsonl so outputs drop into the existing QE analysis tooling.
        jsonl_out = preds_out.with_suffix(".jsonl")
        gold = (eval_text_df["da_mean"].to_numpy(dtype=float)
                if "da_mean" in eval_text_df.columns else np.full(len(eval_text_df), np.nan))
        pda = eval_text_df["pred_da"].to_numpy(dtype=float)
        mask = ~np.isnan(gold)
        summary_row = {"n": int(mask.sum())}
        if mask.sum() >= 2:
            summary_row["pearson"] = float(pearsonr(gold[mask], pda[mask])[0])
            summary_row["spearman"] = float(spearmanr(gold[mask], pda[mask])[0])
            summary_row["kendall"] = float(kendalltau(gold[mask], pda[mask])[0])
        with open(jsonl_out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": {"checkpoint": str(checkpoint_path), "split": eval_name,
                                          "task": "qe", "scale": "da_mean (pred inverse-transformed)"}}) + "\n")
            for _, row in eval_text_df.iterrows():
                f.write(json.dumps({
                    "id": row.get(data_cfg.item_id_column),
                    "pair": row.get(group_column),
                    "gold_da_mean": (None if pd.isna(row.get("da_mean")) else float(row["da_mean"])),
                    "pred_score": float(row["pred_da"]),
                }) + "\n")
            f.write(json.dumps({"_summary": summary_row}) + "\n")
        print(f"✓ IndicQE jsonl saved:  {jsonl_out}  ({summary_row})")
    else:
        # Keep only relevant columns for the analysis CSV
        cols_to_save = [data_cfg.item_id_column, "target_word", data_cfg.label_column, "prediction"]
        existing_cols = [c for c in cols_to_save if c in eval_text_df.columns]
        eval_text_df[existing_cols].to_csv(preds_out, index=False)

    print(f"\n✓ Metrics saved to:     {metrics_out}")
    print(f"✓ Predictions saved to: {preds_out}")


if __name__ == "__main__":
    main()
