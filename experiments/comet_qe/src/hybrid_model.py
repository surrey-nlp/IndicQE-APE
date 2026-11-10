"""
hybrid_model.py

BEA2026HybridRegressor — A custom hybrid regression model for the
BEA 2026 Shared Task on Word Complexity Prediction.

Team SurreyCTS

Architecture:
  1. Pretrained transformer backbone (any HuggingFace encoder)
  2. Multi-layer semantic funneling (configurable layer indices)
  3. Attention-based entropy features (TC, TSA, TSE, CTE, CLS)
  4. String similarity features (Levenshtein / Jaccard / Pinyin)
  5. Categorical embeddings (CEFR, LID)
  6. Auto-scaling deep MLP regression head
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from model_registry import (
    get_model_info,
    get_default_funneling_layers,
    get_entropy_layer_indices,
    get_recommended_mlp_dims,
    normalize_model_name,
)


class BEA2026HybridRegressor(nn.Module):
    """Hybrid regression model combining transformer representations with
    handcrafted linguistic features.

    All feature modules are controlled by boolean flags in the HuggingFace
    config, making this class fully ablatable from YAML experiment configs.
    """

    def __init__(self, model_name: str, config):
        super().__init__()

        self.config = config
        self.transformer = AutoModel.from_pretrained(model_name, config=config)

        # ── Ablation Flags ───────────────────────────────────────────────
        self.use_attention = getattr(config, "use_attention", True)
        self.use_levenshtein = getattr(config, "use_levenshtein", True)
        self.use_cefr = getattr(config, "use_cefr", False)
        self.use_lid = getattr(config, "use_lid", False)
        self.use_dense_embeddings = getattr(config, "use_dense_embeddings", False)
        self.dense_embedding_dim = getattr(config, "dense_embedding_dim", 1024)
        self.dense_projection_dim = getattr(config, "dense_projection_dim", None)
        self.use_subword_ratio = getattr(config, "use_subword_ratio", False)
        self.deep_mlp = getattr(config, "deep_mlp", False)

        # ── QE features (reference-free MT quality estimation) ────────────
        self.use_length_ratio = getattr(config, "use_length_ratio", False)
        self.use_lp = getattr(config, "use_lp", False)

        # ── Custom MLP Dims (Optional Override) ──────────────────────────
        self.custom_mlp_dims = getattr(config, "mlp_dims", None)

        # Funneling
        self.multi_layer_funneling = getattr(config, "multi_layer_funneling", False)

        # Entropy types
        self.use_target_context_entropy = getattr(config, "use_target_context_entropy", False)
        self.use_target_source_alignment = getattr(config, "use_target_source_alignment", False)
        self.use_target_source_entropy = getattr(config, "use_target_source_entropy", False)
        self.use_context_target_entropy = getattr(config, "use_context_target_entropy", False)

        # Advanced entropy
        self.use_multi_layer_entropy = getattr(config, "use_multi_layer_entropy", False)
        self.use_attention_variance = getattr(config, "use_attention_variance", False)
        self.use_global_cls_entropy = getattr(config, "use_global_cls_entropy", False)

        # ── Layer Indices (Configurable) ─────────────────────────────────
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_hidden_layers

        # Funneling layers — user can specify or use defaults
        self._funneling_layers = getattr(
            config, "funneling_layers",
            get_default_funneling_layers(num_layers, getattr(config, "num_funneling_layers", 4))
        )
        num_pooled_layers = len(self._funneling_layers) if self.multi_layer_funneling else 1

        # Entropy layers — derive from funneling layers or use defaults
        self._entropy_layers = getattr(
            config, "entropy_layers",
            get_entropy_layer_indices(self._funneling_layers, num_layers) if self.use_multi_layer_entropy else [num_layers - 1]
        )

        # Whether we need attention/hidden outputs from the transformer
        self._needs_attentions = (
            self.use_attention or self.use_global_cls_entropy
        )
        self._needs_hidden_states = self.multi_layer_funneling

        # ── Categorical Embeddings ───────────────────────────────────────
        self.cefr_dim = 16
        self.lid_dim = 8
        self.cefr_embedder = nn.Embedding(num_embeddings=7, embedding_dim=self.cefr_dim)
        self.lid_embedder = nn.Embedding(num_embeddings=4, embedding_dim=self.lid_dim)

        # Language-pair embedding (QE joint multi-pair models); mirrors LID.
        self.lp_dim = getattr(self.config, "lp_dim", 8)
        self.num_language_pairs = getattr(self.config, "num_language_pairs", 20)
        if self.use_lp:
            self.lp_embedder = nn.Embedding(
                num_embeddings=self.num_language_pairs, embedding_dim=self.lp_dim
            )

        # ── Compute MLP Input Dimension ──────────────────────────────────
        self.mlp_input_dim = hidden_size * (1 + num_pooled_layers)  # CLS + Pooled

        if self.use_attention:
            entropy_multiplier = len(self._entropy_layers) if self.use_multi_layer_entropy else 1
            if self.use_target_context_entropy:
                self.mlp_input_dim += entropy_multiplier
            if self.use_target_source_alignment:
                self.mlp_input_dim += 1  # Always single-layer
            if self.use_target_source_entropy:
                self.mlp_input_dim += entropy_multiplier
            if self.use_context_target_entropy:
                self.mlp_input_dim += entropy_multiplier

        if self.use_global_cls_entropy:
            self.mlp_input_dim += 1
        if self.use_levenshtein:
            self.mlp_input_dim += 1
        if self.use_cefr:
            self.mlp_input_dim += self.cefr_dim
        if self.use_lid:
            self.mlp_input_dim += self.lid_dim
        if self.use_subword_ratio:
            self.mlp_input_dim += 1
        if self.use_length_ratio:
            self.mlp_input_dim += 1
        if self.use_lp:
            self.mlp_input_dim += self.lp_dim

        # ── Projection Heads (Option B) ──────────────────────────────────
        if self.use_dense_embeddings:
            if self.dense_projection_dim is not None:
                # We will project the transformer features AND the dense embeddings
                # to the same dimension before joining, drastically reducing the bottleneck
                self.proj_transformer = nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.dense_projection_dim),
                    nn.GELU()
                )
                self.proj_dense = nn.Sequential(
                    nn.Linear(self.dense_embedding_dim, self.dense_projection_dim),
                    nn.GELU()
                )
                # After concatenation, the regression head will see 2 * projection_dim
                self.regression_input_dim = self.dense_projection_dim * 2
            else:
                # Classical flat concatenation
                self.regression_input_dim = self.mlp_input_dim + self.dense_embedding_dim
                self.proj_transformer = None
                self.proj_dense = None
        else:
            self.regression_input_dim = self.mlp_input_dim
            self.proj_transformer = None
            self.proj_dense = None

        # ── Regression Head ──────────────────────────────────────────────
        if self.deep_mlp:
            if self.custom_mlp_dims:
                mlp_dims = self.custom_mlp_dims
            else:
                mlp_dims = get_recommended_mlp_dims(hidden_size)
                
            # If the user provides 4 dims, we use them; otherwise default fallback structure
            if len(mlp_dims) >= 3:
                self.regression_head = nn.Sequential(
                    nn.Linear(self.regression_input_dim, mlp_dims[0]),
                    nn.LayerNorm(mlp_dims[0]),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(mlp_dims[0], mlp_dims[1]),
                    nn.LayerNorm(mlp_dims[1]),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(mlp_dims[1], mlp_dims[2]),
                    nn.LayerNorm(mlp_dims[2]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(mlp_dims[2], 1),
                )
            else:
                self.regression_head = nn.Sequential(
                    nn.Linear(self.regression_input_dim, mlp_dims[0]),
                    nn.GELU(),
                    nn.Linear(mlp_dims[0], 1)
                )
        else:
            self.regression_head = nn.Linear(self.regression_input_dim, 1)

        self.loss_fct = nn.MSELoss()

    def gradient_checkpointing_enable(self, **kwargs):
        self.transformer.gradient_checkpointing_enable(**kwargs)

    # ── Entropy Helper ───────────────────────────────────────────────────

    def _compute_directed_entropy(
        self,
        layer_attns: list,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute directed attention entropy from query tokens to key tokens.

        This is the unified implementation for TC, TSE, and CTE entropy.

        Args:
            layer_attns: List of attention tensors, each (B, H, S, S).
            query_mask: (B, S) binary mask for query tokens.
            key_mask: (B, S) binary mask for key tokens.

        Returns:
            Tensor of shape (B, num_layers) with entropy values.
        """
        eps = 1e-9
        results = []

        for layer_attn in layer_attns:
            # Expand masks for broadcasting: query(B,1,S,1) × key(B,1,1,S)
            q_mask = query_mask.unsqueeze(1).unsqueeze(-1).bool()
            k_mask = key_mask.unsqueeze(1).unsqueeze(1).bool()
            valid_mask = q_mask & k_mask

            # Mask and normalize
            masked_attn = layer_attn.masked_fill(~valid_mask, 0.0)
            attn_sums = masked_attn.sum(dim=-1, keepdim=True)
            normalized = masked_attn / (attn_sums + eps)

            # Shannon entropy
            entropy_raw = -torch.sum(normalized * torch.log(normalized + eps), dim=-1)

            # Mask out non-query positions
            valid_queries = query_mask.unsqueeze(1).bool()
            entropy_masked = entropy_raw.masked_fill(~valid_queries, 0.0)

            # Aggregate across heads
            query_count = torch.clamp(query_mask.float().sum(dim=1), min=eps)
            if self.use_attention_variance:
                agg = entropy_masked.var(dim=1)
            else:
                agg = entropy_masked.mean(dim=1)

            # Average across query tokens → (B, 1)
            scalar = (agg.sum(dim=1) / query_count).unsqueeze(-1)
            results.append(scalar)

        return torch.cat(results, dim=1)

    # ── Forward Pass ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        target_mask=None,
        context_mask=None,
        source_mask=None,
        lev_score_scaled=None,
        cefr_indices=None,
        lid_indices=None,
        dense_embedding=None,
        subword_ratio=None,
        length_ratio_scaled=None,
        lp_indices=None,
        **kwargs,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 1. Transformer forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=self._needs_attentions,
            output_hidden_states=self._needs_hidden_states,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state
        cls_embedding = last_hidden[:, 0, :]

        # 2. Target word pooling (semantic funneling)
        def pool_state(state):
            expanded = target_mask.unsqueeze(-1).expand(state.size()).float()
            return torch.sum(state * expanded, dim=1) / torch.clamp(expanded.sum(dim=1), min=1e-9)

        if target_mask is not None:
            if self.multi_layer_funneling:
                all_hs = outputs.hidden_states
                pooled_parts = []
                for layer_idx in self._funneling_layers:
                    pooled_parts.append(pool_state(all_hs[layer_idx]))
                pooled_target = torch.cat(pooled_parts, dim=1)
            else:
                pooled_target = pool_state(last_hidden)
        else:
            n_pool = len(self._funneling_layers) if self.multi_layer_funneling else 1
            pooled_target = torch.zeros(
                (batch_size, self.config.hidden_size * n_pool), device=device
            )

        # 3. Entropy features
        empty = torch.empty((batch_size, 0), device=device)

        if self.use_attention and target_mask is not None and self._needs_attentions:
            attn_indices = self._entropy_layers if self.use_multi_layer_entropy else [len(outputs.attentions) - 1]
            layer_attns = [outputs.attentions[i] for i in attn_indices]

            # 3A. Target → Context
            tc_entropy = (
                self._compute_directed_entropy(layer_attns, target_mask, context_mask)
                if self.use_target_context_entropy and context_mask is not None
                else empty
            )

            # 3B. Target → Source Entropy
            ts_entropy = (
                self._compute_directed_entropy(layer_attns, target_mask, source_mask)
                if self.use_target_source_entropy and source_mask is not None
                else empty
            )

            # 3C. Context → Target
            ct_entropy = (
                self._compute_directed_entropy(layer_attns, context_mask, target_mask)
                if self.use_context_target_entropy and context_mask is not None
                else empty
            )

            # 3D. Target → Source Alignment (always last layer only)
            if self.use_target_source_alignment and source_mask is not None:
                last_attn = outputs.attentions[-1]
                t_mask = target_mask.unsqueeze(1).unsqueeze(-1).bool()
                s_mask = source_mask.unsqueeze(1).unsqueeze(1).bool()
                masked_align = last_attn.masked_fill(~(t_mask & s_mask), 0.0)
                align_sums = masked_align.sum(dim=-1).mean(dim=1)
                ts_alignment = align_sums.sum(dim=-1).unsqueeze(-1)
            else:
                ts_alignment = empty
        else:
            tc_entropy = ts_entropy = ct_entropy = ts_alignment = empty

        # 3E. Global CLS Entropy
        if self.use_global_cls_entropy and self._needs_attentions:
            eps = 1e-9
            cls_attn = outputs.attentions[-1][:, :, 0, :]
            cls_ent = -torch.sum(cls_attn * torch.log(cls_attn + eps), dim=-1)
            global_cls_entropy = cls_ent.mean(dim=1).unsqueeze(-1)
        else:
            global_cls_entropy = empty

        # 4. Scalar features
        lev_feature = (
            lev_score_scaled.view(-1, 1).float()
            if self.use_levenshtein and lev_score_scaled is not None
            else empty
        )

        cefr_feature = (
            self.cefr_embedder(cefr_indices)
            if self.use_cefr and cefr_indices is not None
            else empty
        )

        lid_feature = (
            self.lid_embedder(lid_indices)
            if self.use_lid and lid_indices is not None
            else empty
        )

        dense_feature = (
            dense_embedding.float()
            if self.use_dense_embeddings and dense_embedding is not None
            else empty
        )

        sw_feature = (
            subword_ratio.view(-1, 1).float()
            if self.use_subword_ratio and subword_ratio is not None
            else empty
        )

        lr_feature = (
            length_ratio_scaled.view(-1, 1).float()
            if self.use_length_ratio and length_ratio_scaled is not None
            else empty
        )

        lp_feature = (
            self.lp_embedder(lp_indices)
            if self.use_lp and lp_indices is not None
            else empty
        )

        # 5. Concatenate transformer modalities
        features = [cls_embedding, pooled_target]
        if self.use_attention:
            features.extend([tc_entropy, ts_alignment, ts_entropy, ct_entropy])
        if self.use_global_cls_entropy:
            features.append(global_cls_entropy)
        if self.use_levenshtein:
            features.append(lev_feature)
        if self.use_cefr:
            features.append(cefr_feature)
        if self.use_lid:
            features.append(lid_feature)
        if self.use_subword_ratio:
            features.append(sw_feature)
        if self.use_length_ratio:
            features.append(lr_feature)
        if self.use_lp:
            features.append(lp_feature)

        transformer_combined = torch.cat(features, dim=1)

        # 6. Dense feature fusion (Option B handling)
        if self.use_dense_embeddings:
            if self.proj_transformer is not None and self.proj_dense is not None:
                # Separate projections before fusion
                t_proj = self.proj_transformer(transformer_combined)
                d_proj = self.proj_dense(dense_feature)
                final_input = torch.cat([t_proj, d_proj], dim=1)
            else:
                # Classic flat concatenation
                final_input = torch.cat([transformer_combined, dense_feature], dim=1)
        else:
            final_input = transformer_combined

        # 7. Regression
        logits = self.regression_head(final_input)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.squeeze(-1), labels.squeeze(-1))

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)
