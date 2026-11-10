#!/usr/bin/env python3
"""make_fig_head.py — regenerate fig:head from what the code actually does.

The previous figure shipped as a PDF with no source, and its caption said "the encoder is
frozen". It is not: experiments/comet_qe/src/hybrid_model.py builds the backbone with
AutoModel.from_pretrained and never calls requires_grad_(False); there is no freeze anywhere in
src/, scripts/, configs/ or tests/. The encoder is fine-tuned end to end at lr 3e-5. That is also
what makes the E1 result meaningful -- the head does not merely read a frozen representation, it
adapts the encoder, and even so the encoder it starts from dominates the outcome.

Everything drawn here is read off the reference configuration qe-E2-A3.yaml and confirmed against
the saved run summary at results/campaign/outputs/qe-E2-A3/run_summary.json:

  input          segment_order: src_first -> [CLS] source [SEP] MT [SEP], max_length 256
  encoder        models/cometkiwi-infoxlm (InfoXLM-large, 24 layers, hidden 1024), fine-tuned
  funnelling     multi_layer_funneling, num_funneling_layers 4
                 -> get_default_funneling_layers(24, 4) == [12, 16, 20, -1]
                 each tap is masked-mean-pooled over the MT tokens only (pool_state, via
                 target_mask), NOT over the whole sequence
  CLS            last_hidden[:, 0, :] enters the concatenation as its own 1024-d block, which is
                 why the A0 arm is named "CLS+funnel"
  features       use_target_source_alignment (TSA, attention MASS from MT to source, last layer
                 only), use_target_source_entropy (TSE, Shannon entropy of the same attention),
                 use_length_ratio. use_levenshtein is forced False by train.py:707, so the
                 default-True flag in hybrid_model.py never fires here.
  width          1024 (CLS) + 4x1024 (taps) + 1 + 1 + 1 = 5123
  head           deep_mlp -> 5123 -> 1024 -> 512 -> 128 -> 1, each hidden block
                 Linear + LayerNorm + GELU + Dropout (0.2 / 0.15 / 0.1)
  objective      Huber delta 1.0 against a per-pair z-scored DA target

The five ablation axes of Table~\\ref{tab:cometqe} are tagged E1-E5 at the component each varies.

    python3 analysis/make_fig_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "latex" / "figures" / "fig_head.pdf"

INK = "#1a1a1a"
GREY = "#5c5c5c"
FILL_IN = "#f2f2f2"
FILL_ENC = "#dfe7f2"
FILL_FEAT = "#eef2e6"
FILL_HEAD = "#f6ece2"
TAG = "#8a2f2f"


def box(ax, x, y, w, h, text, fc, fs=6.4, weight="normal", ec=GREY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                                linewidth=0.7, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, linespacing=1.35, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=0.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, color=GREY, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def tag(ax, x, y, label):
    ax.text(x, y, label, fontsize=6.0, color=TAG, ha="left", va="center",
            fontweight="bold", zorder=4)


def main() -> int:
    fig, ax = plt.subplots(figsize=(3.85, 3.80))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    L, W = 0.045, 0.795            # main column; tags live to the right of L+W
    CX = L + W / 2
    TAGX = L + W + 0.022

    # ---- input -------------------------------------------------------------------------------
    box(ax, L, 0.928, W, 0.056, "[CLS] source [SEP] MT [SEP]   (max 256)", FILL_IN, fs=6.2)
    arrow(ax, CX, 0.928, CX, 0.898)

    # ---- encoder -----------------------------------------------------------------------------
    box(ax, L, 0.808, W, 0.090,
        "CometKiwi encoder (InfoXLM-large, 24 layers)\n"
        r"fine-tuned end to end,  lr $3\!\times\!10^{-5}$", FILL_ENC, fs=5.9, weight="bold")
    tag(ax, TAGX, 0.853, "E1")

    # ---- three branches off the encoder ------------------------------------------------------
    bx = [(L, 0.295), (L + 0.315, 0.145), (L + 0.475, 0.320)]   # (x, width)
    cx = [x + w / 2 for x, w in bx]
    for c in cx:
        arrow(ax, CX, 0.808, c, 0.768)

    box(ax, bx[0][0], 0.632, bx[0][1], 0.136,
        "layer taps\n12 / 16 / 20 / 24\nmean-pool over\nMT tokens", FILL_FEAT, fs=5.8)
    box(ax, bx[1][0], 0.632, bx[1][1], 0.136, "[CLS]\nlast\nlayer", FILL_FEAT, fs=5.8)
    box(ax, bx[2][0], 0.632, bx[2][1], 0.136,
        "last-layer attn.\nMT$\\to$source\nTSA mass\nTSE entropy", FILL_FEAT, fs=5.8)
    tag(ax, bx[0][0], 0.786, "E3")
    ax.text(bx[2][0] + bx[2][1], 0.786, "E2", fontsize=6.0, color=TAG, ha="right",
            va="center", fontweight="bold", zorder=4)

    # the right branch's arrow runs through this row, so its width label is offset off the line
    for c, lab, dx in zip(cx, (r"$4\times1024$", r"$1024$", r"$2$"), (-0.052, -0.040, -0.058)):
        ax.text(c + dx, 0.604, lab, ha="center", va="center", fontsize=5.8, color=GREY)

    # length ratio bypasses the encoder entirely
    box(ax, bx[2][0], 0.512, bx[2][1], 0.052, "length ratio (1)", FILL_FEAT, fs=5.8)
    arrow(ax, cx[2], 0.512, cx[2], 0.474, ls=(0, (2, 1.6)))
    for c in (cx[0], cx[1]):
        arrow(ax, c, 0.632, c, 0.474)
    arrow(ax, cx[2], 0.632, cx[2], 0.564)

    # ---- concat ------------------------------------------------------------------------------
    box(ax, L, 0.414, W, 0.058, r"concatenate  $\rightarrow$  $5123$", FILL_IN, fs=6.2)
    arrow(ax, CX, 0.414, CX, 0.382)

    # ---- MLP ---------------------------------------------------------------------------------
    box(ax, L, 0.208, W, 0.174,
        "MLP head\n"
        r"$5123 \rightarrow 1024 \rightarrow 512 \rightarrow 128 \rightarrow 1$"
        "\n\nLinear + LayerNorm + GELU + Dropout\nper hidden block (0.2 / 0.15 / 0.1)",
        FILL_HEAD, fs=5.9)
    arrow(ax, CX, 0.208, CX, 0.176)

    # ---- objective ---------------------------------------------------------------------------
    box(ax, L, 0.082, W, 0.094,
        "predicted DA\n"
        r"Huber ($\delta{=}1$) against per-pair $z$ target", FILL_IN, fs=6.2)
    tag(ax, TAGX, 0.143, "E4")
    tag(ax, TAGX, 0.106, "E5")

    ax.text(0.50, 0.022,
            "E1 encoder · E2 features · E3 funnelling · E4 loss · E5 target",
            ha="center", va="center", fontsize=5.8, color=TAG, style="italic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.012)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
