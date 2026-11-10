#!/usr/bin/env python3
"""make_head_figure.py — architecture of the COMET-encoder regression head (fig:head).

The figure does double duty: it shows the pipeline, and it marks where each ablation axis of S6
intervenes, so a reader can map tab:cometqe's E1-E5 rows onto the component each one varies. The
tags sit ON the components rather than in a margin column, which keeps the figure to one column
without clipping.

Components read off experiments/comet_qe/src/hybrid_model.py and scripts/gen_qe_configs.py:
  E1 encoder        CometKiwi / COMET-DA / XLM-R / InfoXLM      the dominant factor
  E2 features       A0 (CLS only) .. A3 (full entropy set)
  E3 funnelling     layer taps, default [12, 16, 20, -1] on 24L
  E4 loss           Huber(delta) / MSE / CCC
  E5 target scale   per-pair z / min-max / raw DA

Writes paper/latex/figures/fig_head.pdf. No LaTeX macros in the labels: matplotlib renders them
literally.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "latex" / "figures" / "fig_head.pdf"

INK, MUTED, FILL, TAP, ABL = "#1a1a1a", "#666666", "#f4f4f4", "#dcdcdc", "#7a7a7a"


def box(ax, x, y, w, h, label, sub=None, fc=FILL, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                                fc=fc, ec=INK, lw=0.9, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=6.0, color=MUTED, style="italic", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)


def arrow(ax, x1, y1, x2, y2, lw=0.9, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 lw=lw, color=color, shrinkA=1, shrinkB=1, zorder=4))


def tag(ax, x, y, t):
    """Ablation marker placed on the component it varies."""
    ax.text(x, y, t, ha="center", va="center", fontsize=5.9, color="white", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.20", fc=ABL, ec="none"))


def main() -> int:
    fig, ax = plt.subplots(figsize=(3.32, 3.95))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- inputs
    box(ax, 0.10, 0.905, 0.34, 0.062, "source")
    box(ax, 0.52, 0.905, 0.34, 0.062, "MT output")
    arrow(ax, 0.27, 0.905, 0.33, 0.858)
    arrow(ax, 0.69, 0.905, 0.63, 0.858)

    # ---- encoder
    ax.add_patch(FancyBboxPatch((0.06, 0.545), 0.60, 0.312,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                fc="white", ec=INK, lw=0.9, zorder=2))
    ax.text(0.36, 0.824, "frozen COMET encoder", ha="center", va="center",
            fontsize=7.6, color=INK, zorder=3)
    tag(ax, 0.615, 0.824, "E1")
    ax.text(0.36, 0.788, "24 layers, weights not updated", ha="center", va="center",
            fontsize=6.0, color=MUTED, style="italic", zorder=3)

    taps = [("layer 24", 0.735), ("layer 20", 0.680), ("layer 16", 0.625), ("layer 12", 0.570)]
    for lab, y in taps:
        ax.add_patch(FancyBboxPatch((0.115, y - 0.019), 0.36, 0.038,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    fc=TAP, ec="none", zorder=3))
        ax.text(0.145, y, lab, ha="left", va="center", fontsize=6.1, color=INK, zorder=4)
        arrow(ax, 0.475, y, 0.700, 0.663, lw=0.7, color=MUTED)
    tag(ax, 0.505, 0.570, "E3")

    # ---- pooled
    box(ax, 0.700, 0.615, 0.245, 0.095, "pooled", sub="4 taps")

    # ---- entropy features
    box(ax, 0.055, 0.400, 0.40, 0.098, "attention-entropy", sub="TC  TSA  TSE  CTE  CLS")
    tag(ax, 0.437, 0.483, "E2")
    arrow(ax, 0.255, 0.545, 0.255, 0.498)

    # ---- concat
    box(ax, 0.545, 0.400, 0.36, 0.098, "concatenate")
    arrow(ax, 0.822, 0.615, 0.760, 0.498)
    arrow(ax, 0.455, 0.449, 0.545, 0.449)

    # ---- MLP
    box(ax, 0.29, 0.225, 0.42, 0.098, "MLP regression head", sub="auto-scaled dims", fs=7.2)
    tag(ax, 0.693, 0.308, "E4")
    arrow(ax, 0.725, 0.400, 0.60, 0.323)

    # ---- output
    box(ax, 0.29, 0.055, 0.42, 0.098, "predicted DA", sub="per-pair z target")
    tag(ax, 0.693, 0.138, "E5")
    arrow(ax, 0.50, 0.225, 0.50, 0.153)

    # ---- legend
    ax.text(0.03, 0.005, "E1 encoder   E2 features   E3 taps   E4 loss   E5 target scale",
            ha="left", va="bottom", fontsize=5.8, color=ABL)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
