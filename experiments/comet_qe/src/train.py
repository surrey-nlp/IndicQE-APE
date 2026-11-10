from __future__ import annotations
import os

"""
train.py

Fine-tunes a HuggingFace transformer for the BEA 2026 Shared Task
as a regression model predicting lexical difficulty (GLMM scores).

Team SurreyCTS — Lexical Difficulty Prediction Framework

Features:
  - Any HuggingFace encoder backbone
  - Optional BEA2026HybridRegressor with deep MLP head
  - Optional LoRA adapter for parameter-efficient fine-tuning
  - RMSE as primary model-selection metric
  - Comprehensive CSV + W&B logging of all ablation flags
"""

import argparse
import gc
import torch
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
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
from losses import get_loss_function
from hybrid_model import BEA2026HybridRegressor
from model_registry import (
    normalize_model_name,
    get_model_info,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML experiment config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    """Compute regression metrics. RMSE is the primary model-selection metric."""
    preds, labels = eval_pred
    preds = np.squeeze(preds)

    rmse = root_mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)

    try:
        pearson = pearsonr(labels, preds)[0]
    except Exception:
        pearson = 0.0
    try:
        spearman = spearmanr(labels, preds)[0]
    except Exception:
        spearman = 0.0
    try:
        kendall = kendalltau(labels, preds)[0]
    except Exception:
        kendall = 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "kendall": float(kendall),
    }


# ---------------------------------------------------------------------------
# Custom Trainer with configurable loss
# ---------------------------------------------------------------------------

class CustomLossTrainer(Trainer):
    """Trainer subclass that uses a user-specified loss function."""

    def __init__(self, loss_fn=None, loss_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_loss_fn = loss_fn
        self.custom_loss_kwargs = loss_kwargs or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1)
        loss = self.custom_loss_fn(preds, labels, **self.custom_loss_kwargs)
        return (loss, outputs) if return_outputs else loss


from dataclasses import dataclass
from transformers import TrainerCallback


class EpochLogCallback(TrainerCallback):
    """Callback that logs comprehensive metrics at the end of each epoch.
    
    Prints all metrics to console and ensures they are captured in W&B.
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = state.epoch or 0
        step = state.global_step
        print(f"\n{'─'*70}")
        print(f"  Epoch {epoch:.0f} | Step {step} | Evaluation Results")
        print(f"{'─'*70}")
        print(f"  {'Metric':<20} {'Value':>12}")
        print(f"  {'─'*32}")
        # Priority order: RMSE first, then correlations, then MAE/loss
        priority_keys = ['eval_rmse', 'eval_pearson', 'eval_spearman', 'eval_kendall', 'eval_mae', 'eval_loss']
        for key in priority_keys:
            if key in metrics:
                label = key.replace('eval_', '').upper()
                print(f"  {label:<20} {metrics[key]:>12.6f}")
        # Log any remaining metrics
        for key, val in sorted(metrics.items()):
            if key not in priority_keys and isinstance(val, (int, float)):
                print(f"  {key:<20} {val:>12.6f}")
        print(f"{'─'*70}\n")

        # Explicit W&B logging with epoch tag
        try:
            import wandb
            if wandb.run is not None:
                epoch_metrics = {f"epoch/{k.replace('eval_', '')}": v
                                 for k, v in metrics.items()
                                 if isinstance(v, (int, float))}
                epoch_metrics["epoch"] = epoch
                wandb.log(epoch_metrics, step=step)
        except ImportError:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training loss at each logging step."""
        if logs and 'loss' in logs:
            step = state.global_step
            lr = logs.get('learning_rate', 0)
            loss = logs['loss']
            epoch = logs.get('epoch', 0)
            print(f"  Step {step:>5} | Epoch {epoch:.2f} | Loss: {loss:.6f} | LR: {lr:.2e}")

@dataclass
class CustomDataCollator(DataCollatorWithPadding):
    """Extends DataCollatorWithPadding to properly pad custom features."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        custom_features = {}
        custom_keys = [
            "target_mask", "context_mask", "source_mask",
            "lev_score_scaled", "cefr_indices", "lid_indices",
            "subword_ratio", "dense_embedding",
            # QE features (reference-free MT QE). Handled explicitly rather than
            # relying on tokenizer.pad passing them through — that is
            # transformers-version-dependent and drops unknown keys on older
            # versions (e.g. the cluster's `indicqe` env).
            "length_ratio_scaled", "lp_indices",
        ]

        for key in custom_keys:
            if key in features[0]:
                custom_features[key] = [f.pop(key) for f in features]

        batch = super().__call__(features)
        max_len = batch["input_ids"].shape[1]

        for key, vals in custom_features.items():
            if key in ["target_mask", "context_mask", "source_mask"]:
                padded = []
                for v in vals:
                    if len(v) > max_len:
                        v = v[:max_len]
                    padded.append(v + [0] * (max_len - len(v)))
                batch[key] = torch.tensor(padded, dtype=torch.long)
            elif key == "dense_embedding":
                batch[key] = torch.tensor(vals, dtype=torch.float)
            elif key == "subword_ratio":
                batch[key] = torch.tensor(vals, dtype=torch.float)
            else:
                dtype = torch.float if "scaled" in key else torch.long
                batch[key] = torch.tensor(vals, dtype=dtype)

        return batch


def main() -> None:
    # ── Logging Setup ──────────────────────────────────────────────────
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_propagation()

    parser = argparse.ArgumentParser(
        description="BEA 2026 Lexical Difficulty Prediction — Training Script"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_config(config_path)

    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    experiment_name = cfg.get("experiment_name", Path(args.config).stem)

    # ── Pre-flight validation ────────────────────────────────────────────
    warnings_list = validate_config(cfg)
    for w in warnings_list:
        print(f"⚠️  Config Warning: {w}")

    # ── W&B Setup ────────────────────────────────────────────────────────
    os.environ.setdefault("WANDB_PROJECT", "bea2026-framwrk")
    os.environ.setdefault("WANDB_LOG_MODEL", "false")

    model_name = normalize_model_name(cfg["model_name"])
    seed = int(cfg.get("seed", 42))
    max_length = int(cfg.get("max_length", 256))
    learning_rate = float(cfg.get("learning_rate", 2e-5))
    train_batch_size = int(cfg.get("train_batch_size", 16))
    eval_batch_size = int(cfg.get("eval_batch_size", 16))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    num_train_epochs = float(cfg.get("num_train_epochs", 5))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    loss_name = cfg.get("loss_function", "mse")
    loss_kwargs = cfg.get("loss_kwargs", {})
    loss_fn, loss_kwargs = get_loss_function(loss_name, loss_kwargs)
    use_custom_loss = loss_name.lower() != "mse"

    output_root = Path(cfg.get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    results_root = Path(cfg.get("results_root", "results"))
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    run_dir = output_root / experiment_name
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Data Config ──────────────────────────────────────────────────────
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

    if splits["train"].empty:
        raise ValueError("Train split is empty.")
    if splits["dev"].empty:
        raise ValueError("Dev split is empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    sep_token = tokenizer.sep_token or "[SEP]"

    # ── Task switch: BEA lexical difficulty vs reference-free MT QE ───────
    task = cfg.get("task", "bea")
    group_column = cfg.get("group_column", "language_pair" if task == "qe" else "L1")
    qe_use_length_ratio = bool(cfg.get("use_length_ratio", False))
    qe_use_lp = bool(cfg.get("use_lp", False))
    lp_map: Dict[str, int] = {}

    has_test = "test" in splits and not splits["test"].empty

    if task == "qe":
        source_col = cfg.get("source_column", "source")
        mt_col = cfg.get("mt_column", "mt")
        order = cfg.get("segment_order", "src_first")

        def _qe_text(split_df):
            return add_qe_text_column(
                split_df, sep_token=sep_token,
                source_col=source_col, mt_col=mt_col, order=order,
            )

        train_df = _qe_text(splits["train"])
        dev_df = _qe_text(splits["dev"])
        test_df = _qe_text(splits["test"]) if has_test else None

        prepare_qe_features(
            train_df, dev_df, test_df,
            source_col=source_col, mt_col=mt_col,
            length_unit=cfg.get("length_unit", "token"),
        )

        # Language-pair indices for the optional lp embedding (A4). The map is
        # rebuilt identically at eval time (train is always loaded there too).
        if qe_use_lp:
            lp_values = sorted(train_df[group_column].dropna().unique())
            lp_map = {p: i for i, p in enumerate(lp_values)}
            for d in (train_df, dev_df, test_df):
                if d is not None:
                    d["lp_indices"] = d[group_column].map(lambda p: lp_map.get(p, 0)).astype(int)

        input_columns = []  # unused for QE; kept only for the summary row
    else:
        input_columns = resolve_input_columns(data_cfg.input_columns, data_cfg.exclude_columns)

        train_df = add_text_column(splits["train"], sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)
        dev_df = add_text_column(splits["dev"], sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)
        test_df = (
            add_text_column(splits["test"], sep_token=sep_token, input_columns=input_columns, cfg=data_cfg)
            if has_test else None
        )

        prepare_features(
            train_df, dev_df, test_df,
            use_normalized_levenshtein=bool(cfg.get("use_normalized_levenshtein", False)),
            use_jaccard=bool(cfg.get("use_jaccard", False)),
            use_pinyin=bool(cfg.get("use_pinyin", False)),
        )

    # Move model/feature lookups before the banner
    use_hybrid_model = bool(cfg.get("use_hybrid_model", False))
    use_lora = bool(cfg.get("use_lora", False))

    # ── Pre-training Configuration Banner ─────────────────────────────
    data_file = Path(data_cfg.data_path).name
    print(f"\n{'╔' + '═'*62 + '╗'}")
    print(f"║  {'Experiment Configuration':^60} ║")
    print(f"{'╠' + '═'*62 + '╣'}")
    print(f"║  Experiment:  {experiment_name:<47} ║")
    print(f"║  Model:       {model_name[:47]:<47} ║")
    print(f"║  Data:        {data_file:<47} ║")
    print(f"║  Train:       {len(train_df):<47} ║")
    print(f"║  Dev:         {len(dev_df):<47} ║")
    test_info = f"{len(test_df)} (labels: {'yes' if has_test and not test_df[data_cfg.label_column].isna().all() else 'NO'})" if has_test else 'N/A'
    print(f"║  Test:        {test_info:<47} ║")
    print(f"║  Loss:        {loss_name:<47} ║")
    print(f"║  Hybrid:      {str(use_hybrid_model):<47} ║")
    print(f"║  Deep MLP:    {str(bool(cfg.get('deep_mlp', False))):<47} ║")
    print(f"║  Epochs:      {num_train_epochs:<47} ║")
    print(f"║  LR:          {learning_rate:<47} ║")
    print(f"║  Batch:       {train_batch_size} (×{gradient_accumulation_steps} accum){'':40} ║"[:65] + " ║")
    if cfg.get("multi_layer_funneling"):
        print(f"║  Funneling:   {str(cfg.get('funneling_layers', [])):<47} ║")
    print(f"{'╚' + '═'*62 + '╝'}\n")

    # ── Dense Embeddings (Optional) ──────────────────────────────────────
    embeddings_path = cfg.get("embeddings_path", None)
    include_embeddings = False
    if embeddings_path is not None:
        emb_path = Path(embeddings_path)
        if not emb_path.is_absolute():
            emb_path = REPO_ROOT / emb_path
        print(f"Loading dense embeddings from {emb_path}...")
        emb_dict = torch.load(emb_path, map_location="cpu", weights_only=True)
        default_emb = [0.0] * next(iter(emb_dict.values())).shape[-1]
        for df_split in [train_df, dev_df, test_df]:
            if df_split is not None:
                df_split["dense_embedding"] = df_split[data_cfg.item_id_column].map(
                    lambda x: emb_dict.get(x).tolist() if x in emb_dict else default_emb
                )
        include_embeddings = True

    # ── Datasets ─────────────────────────────────────────────────────────
    def _to_dataset(frame):
        if task == "qe":
            return qe_dataframe_to_dataset(
                frame, data_cfg.label_column,
                use_length_ratio=qe_use_length_ratio, use_lp=qe_use_lp,
            )
        return dataframe_to_dataset(frame, data_cfg.label_column, include_embeddings=include_embeddings)

    train_ds = _to_dataset(train_df)
    dev_ds = _to_dataset(dev_df)

    def tokenize_function(batch):
        tokenized = tokenizer(
            batch["text"], truncation=True, max_length=max_length,
            return_offsets_mapping=True,
        )

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
                    return (off_s >= span_s and off_e <= span_e) or \
                           (off_e > span_s and off_s < span_e)

                t_mask.append(1 if overlaps(tgt_s, tgt_e) else 0)
                c_mask.append(1 if overlaps(ctx_s, ctx_e) else 0)
                s_mask.append(1 if overlaps(src_s, src_e) else 0)

            # Fallback: if truncation removed a span entirely, mark CLS
            if sum(t_mask) == 0: t_mask[0] = 1
            if sum(c_mask) == 0: c_mask[0] = 1
            if sum(s_mask) == 0: s_mask[0] = 1

            target_masks.append(t_mask)
            context_masks.append(c_mask)
            source_masks.append(s_mask)

        tokenized["target_mask"] = target_masks
        tokenized["context_mask"] = context_masks
        tokenized["source_mask"] = source_masks
        del tokenized["offset_mapping"]
        return tokenized

    train_ds = train_ds.map(tokenize_function, batched=True)
    dev_ds = dev_ds.map(tokenize_function, batched=True)

    if has_test:
        test_ds = _to_dataset(test_df)
        test_ds = test_ds.map(tokenize_function, batched=True)

    # ── Model ────────────────────────────────────────────────────────────
    if use_hybrid_model:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)

        # Inject ALL ablation flags from YAML
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
        # Defaults: most features OFF for clean ablation
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

        # Language-pair embedding sizing (QE A4 ablation).
        if task == "qe" and qe_use_lp:
            hf_config.num_language_pairs = int(cfg.get("num_language_pairs", len(lp_map)))
            hf_config.lp_dim = int(cfg.get("lp_dim", 8))

        # Configurable layer indices
        if "funneling_layers" in cfg:
            hf_config.funneling_layers = cfg["funneling_layers"]
        if "num_funneling_layers" in cfg:
            hf_config.num_funneling_layers = int(cfg["num_funneling_layers"])
        if "entropy_layers" in cfg:
            hf_config.entropy_layers = cfg["entropy_layers"]

        model = BEA2026HybridRegressor(model_name, config=hf_config)
        print(f"✓ Hybrid model initialized. MLP input dim: {model.mlp_input_dim}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="regression",
        )

    # ── LoRA (Optional) ──────────────────────────────────────────────────
    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_r = int(cfg.get("lora_r", 16))
            lora_alpha = int(cfg.get("lora_alpha", 32))
            lora_dropout = float(cfg.get("lora_dropout", 0.1))
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                task_type=TaskType.SEQ_CLS,
                target_modules=cfg.get("lora_target_modules", ["query", "value"]),
            )
            if use_hybrid_model:
                model.transformer = get_peft_model(model.transformer, lora_config)
            else:
                model = get_peft_model(model, lora_config)
            print(f"✓ LoRA applied (r={lora_r}, α={lora_alpha})")
        except ImportError:
            raise ImportError("LoRA requires `peft`. Install: pip install peft>=0.7.0")

    # ── Training Arguments ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_steps=int(cfg.get("warmup_steps", 100)),
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        # Config-driven model selection (QE uses pearson/greater; BEA keeps rmse/lower).
        # See QE_ADAPTATION/01_correctness_audit.md C8.
        metric_for_best_model=cfg.get("metric_for_best_model", "rmse"),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
        logging_strategy="steps",
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        # Config-driven so QE / smoke / cluster runs can opt out of W&B without
        # env hacks (BEA default stays "wandb").
        report_to=cfg.get("report_to", "wandb"),
        disable_tqdm=False,
        run_name=experiment_name,
        fp16=torch.cuda.is_available() and bool(cfg.get("fp16", True)),
        dataloader_pin_memory=False,
        seed=seed,
    )

    # Log full config to W&B
    if "wandb" in training_args.report_to:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(cfg, allow_val_change=True)
        except ImportError:
            pass

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer_cls = CustomLossTrainer if use_custom_loss else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EpochLogCallback()],
    )
    if use_custom_loss:
        trainer_kwargs["loss_fn"] = loss_fn
        trainer_kwargs["loss_kwargs"] = loss_kwargs

    trainer = trainer_cls(**trainer_kwargs)

    try:
        trainer.train()

        # ── Evaluation ───────────────────────────────────────────────────
        dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
        test_metrics = {}
        # Only evaluate test if labels exist and are real (not NaN placeholders)
        has_real_test_labels = (
            has_test and test_df is not None
            and data_cfg.label_column in test_df.columns
            and not test_df[data_cfg.label_column].isna().all()
        )
        if has_real_test_labels:
            # Manually evaluate on test set
            test_metrics = trainer.evaluate(eval_dataset=test_ds)
            # Log test metrics to W&B with test/ prefix (so they don't overwrite eval_)
            if "wandb" in training_args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb_test = {f"test/{k.replace('eval_', '')}": v 
                                      for k, v in test_metrics.items() 
                                      if isinstance(v, (int, float))}
                        wandb.run.log(wandb_test)
                except ImportError:
                    pass
            test_metrics = {k.replace("eval_", "test_"): v for k, v in test_metrics.items()}
        elif has_test:
            print("ℹ Test split has no real labels (NaN). Skipping test evaluation.")

        # ── Per-instance predictions (saved for EVERY run) ───────────────
        # One row per segment: item_id (=uid), language_pair, gold, pred. This is the
        # substrate for per-pair, per-route (join uid->challenge routes) and cross-lingual
        # analysis, and for correlating against MQM. Never rely on aggregates alone.
        def _dump_predictions(ds, frame, split_name):
            if ds is None or frame is None or len(frame) == 0:
                return None
            preds = trainer.predict(ds).predictions.flatten()
            cols = {"pred": preds}
            for c in (data_cfg.item_id_column, group_column, data_cfg.label_column):
                if c in frame.columns:
                    cols[c] = frame[c].values
            pred_df = pd.DataFrame(cols)
            pred_df.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
            per_pair = {}
            if group_column in pred_df.columns and data_cfg.label_column in pred_df.columns:
                from scipy.stats import pearsonr as _pr, spearmanr as _sr
                for gp, g in pred_df.groupby(group_column):
                    m = g[data_cfg.label_column].notna() & g["pred"].notna()
                    if int(m.sum()) >= 3:
                        per_pair[str(gp)] = {
                            "n": int(m.sum()),
                            "pearson": round(float(_pr(g.loc[m, "pred"], g.loc[m, data_cfg.label_column])[0]), 4),
                            "spearman": round(float(_sr(g.loc[m, "pred"], g.loc[m, data_cfg.label_column])[0]), 4),
                        }
            return per_pair

        per_pair_test = _dump_predictions(test_ds, test_df, "test") if has_real_test_labels else None
        per_pair_dev = _dump_predictions(dev_ds, dev_df, "dev")

        data_file = Path(data_cfg.data_path).name
        track = "open" if data_file in ("all.csv", "official_all.csv") else "closed"

        # ── Comprehensive Summary ────────────────────────────────────────
        summary = {
            # Identifiers
            "experiment_name": experiment_name,
            "model_name": model_name,
            "track": track,
            "data_file": data_file,
            # Dev metrics
            "dev_rmse": dev_metrics.get("eval_rmse"),
            "dev_pearson": dev_metrics.get("eval_pearson"),
            "dev_spearman": dev_metrics.get("eval_spearman"),
            "dev_kendall": dev_metrics.get("eval_kendall"),
            "dev_mae": dev_metrics.get("eval_mae"),
            "dev_loss": dev_metrics.get("eval_loss"),
            # Test metrics
            "test_rmse": test_metrics.get("test_rmse"),
            "test_pearson": test_metrics.get("test_pearson"),
            "test_spearman": test_metrics.get("test_spearman"),
            "test_kendall": test_metrics.get("test_kendall"),
            "test_mae": test_metrics.get("test_mae"),
            "test_loss": test_metrics.get("test_loss"),
            # Per-pair breakdown (from the saved per-instance predictions)
            "per_pair_test": per_pair_test,
            "per_pair_dev": per_pair_dev,
            # Loss config
            "loss_function": loss_name,
            "loss_kwargs": json.dumps(loss_kwargs) if loss_kwargs else None,
            # Data
            "num_train": len(train_df),
            "num_dev": len(dev_df),
            "num_test": len(test_df) if has_test else None,
            # Hyperparameters
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "seed": seed,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            # Architecture flags
            "use_hybrid_model": use_hybrid_model,
            "deep_mlp": bool(cfg.get("deep_mlp", False)),
            "use_lora": use_lora,
            # Feature flags
            "use_attention": bool(cfg.get("use_attention", True)),
            "use_levenshtein": bool(cfg.get("use_levenshtein", False)),
            "use_normalized_levenshtein": bool(cfg.get("use_normalized_levenshtein", False)),
            "use_jaccard": bool(cfg.get("use_jaccard", False)),
            "use_pinyin": bool(cfg.get("use_pinyin", False)),
            "use_subword_ratio": bool(cfg.get("use_subword_ratio", False)),
            "use_cefr": bool(cfg.get("use_cefr", False)),
            "use_lid": bool(cfg.get("use_lid", False)),
            "prepend_lid": bool(cfg.get("prepend_lid", False)),
            # Entropy flags
            "use_target_context_entropy": bool(cfg.get("use_target_context_entropy", False)),
            "use_target_source_alignment": bool(cfg.get("use_target_source_alignment", False)),
            "use_target_source_entropy": bool(cfg.get("use_target_source_entropy", False)),
            "use_context_target_entropy": bool(cfg.get("use_context_target_entropy", False)),
            "use_multi_layer_entropy": bool(cfg.get("use_multi_layer_entropy", False)),
            "use_attention_variance": bool(cfg.get("use_attention_variance", False)),
            "use_global_cls_entropy": bool(cfg.get("use_global_cls_entropy", False)),
            # Layer config
            "multi_layer_funneling": bool(cfg.get("multi_layer_funneling", False)),
            "funneling_layers": str(cfg.get("funneling_layers")) if cfg.get("funneling_layers") else None,
            "num_funneling_layers": cfg.get("num_funneling_layers"),
        }

        # Save JSON summary
        with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Update W&B summary explicitly for the final report
        if "wandb" in training_args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    # Filter out non-serializable objects (if any)
                    wandb_summary = {k: v for k, v in summary.items() 
                                     if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
                    wandb.run.summary.update(wandb_summary)
            except ImportError:
                pass

        # Append to central CSV
        summary_df = pd.DataFrame([summary])
        results_csv = results_root / "experiment_results.csv"
        if results_csv.exists():
            old = pd.read_csv(results_csv)
            combined = pd.concat([old, summary_df], ignore_index=True)
        else:
            combined = summary_df
        combined.to_csv(results_csv, index=False)

        # ── Final Results Banner ──────────────────────────────────────
        print(f"\n{'╔' + '═'*62 + '╗'}")
        print(f"║  {'FINAL RESULTS':^60} ║")
        print(f"{'╠' + '═'*62 + '╣'}")
        print(f"║  Experiment: {experiment_name:<48} ║")
        print(f"║  Best Ckpt:  {str(trainer.state.best_model_checkpoint or 'N/A')[-48:]:<48} ║")
        print(f"{'╠' + '═'*62 + '╣'}")
        print(f"║  {'DEV SET':^60} ║")
        print(f"║  {'Metric':<20} {'Value':>12}{'':>29} ║")
        print(f"║  {'─'*34}{'':>27} ║")
        for k in ['dev_rmse', 'dev_pearson', 'dev_spearman', 'dev_kendall', 'dev_mae', 'dev_loss']:
            v = summary.get(k)
            if v is not None:
                label = k.replace('dev_', '').upper()
                print(f"║  {label:<20} {v:>12.6f}{'':>29} ║")
        # Per-group dev breakdown (BEA: L1; QE: language_pair)
        if group_column in dev_df.columns:
            print(f"║{'':>63}║")
            print(f"║  {('Per-' + group_column + ' Dev RMSE / Pearson'):^60} ║")
            print(f"║  {'─'*34}{'':>27} ║")
            from scipy.stats import pearsonr as _pr
            from sklearn.metrics import root_mean_squared_error as _rmse
            for grp_val in sorted(dev_df[group_column].dropna().unique()):
                grp_mask = dev_df[group_column] == grp_val
                grp_preds = trainer.predict(dev_ds.select(
                    [i for i, m in enumerate(grp_mask) if m]
                )).predictions.flatten()
                grp_labels = dev_df.loc[grp_mask, data_cfg.label_column].values
                grp_rmse = _rmse(grp_labels, grp_preds)
                grp_pearson = _pr(grp_preds, grp_labels)[0]
                print(f"║  {str(grp_val).upper():<20} RMSE: {grp_rmse:.4f}  Pearson: {grp_pearson:.4f}{'':>10} ║")
        if summary.get('test_rmse') is not None:
            print(f"{'╠' + '═'*62 + '╣'}")
            print(f"║  {'TEST SET':^60} ║")
            for k in ['test_rmse', 'test_pearson', 'test_spearman', 'test_kendall', 'test_mae']:
                v = summary.get(k)
                if v is not None:
                    label = k.replace('test_', '').upper()
                    print(f"║  {label:<20} {v:>12.6f}{'':>29} ║")
        elif has_test:
            print(f"║{'':>63}║")
            print(f"║  Test: Skipped (no labels available){'':>26} ║")
        print(f"{'╚' + '═'*62 + '╝'}")

    finally:
        del trainer, model, tokenizer, train_ds, dev_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
