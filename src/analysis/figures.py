"""Manuscript figures from frozen result tables (E21): E12 scores by model and E14 model - human differences.

Small multiples, one panel per metric with its own x-axis, the models as rows (identity by position and label, so
one ink colour serves every mark). Dots are point estimates, lines 95% bootstrap CIs. In the difference figure a
hollow dot marks a difference that is not significant after Holm correction (E16), and a dashed line marks zero.
Same style and reproducible saving as the E20 figures.
"""

from __future__ import annotations

import math
from typing import Any

from src.analysis.qualitative_figures import AXIS, SURFACE, TEXT, _style, plt

INK = "#2a78d6"


def dot_panels(
    panels: list[dict[str, Any]], systems: list[str], names: dict[str, str], title: str, zero_line: bool = False
) -> plt.Figure:
    """`panels`: [{"label": ..., "rows": {system: (value, ci_low, ci_high, filled)}}], one panel per metric."""
    columns = 5
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7.2, 1.15 * len(systems) * rows / 2 + 0.9), facecolor=SURFACE,
                             sharey=True, squeeze=False)  # fmt: skip
    positions = {system: len(systems) - 1 - i for i, system in enumerate(systems)}  # first system on top
    for ax, panel in zip(axes.flat, panels, strict=False):
        for system in systems:
            value, low, high, filled = panel["rows"][system]
            y = positions[system]
            ax.plot([low, high], [y, y], color=INK, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([value], [y], s=26, zorder=3, color=INK if filled else SURFACE, edgecolors=INK, linewidths=1.5)
        if zero_line:
            ax.axvline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
        ax.set_title(panel["label"], fontsize=8, color=TEXT, loc="left")
        ax.set_yticks(list(positions.values()), [names[s] for s in positions])
        ax.set_ylim(-0.6, len(systems) - 0.4)
        ax.locator_params(axis="x", nbins=3)
        _style(ax)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.margins(x=0.15)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=9.5)
    fig.tight_layout()
    return fig


def e12_panels(summary: list[dict[str, str]], metrics: list[str], labels: dict[str, str]) -> list[dict[str, Any]]:
    """Panels from five_reference_summary.csv rows (sample `all`): item mean and its CI per system."""
    by = {(r["metric"], r["system"]): r for r in summary if r["sample"] == "all"}
    return [
        {
            "label": labels[m],
            "rows": {
                system: (float(r["item_mean"]), float(r["ci_low"]), float(r["ci_high"]), True)
                for (metric, system), r in by.items()
                if metric == m
            },
        }
        for m in metrics
    ]


def e14_panels(tests: list[dict[str, str]], metrics: list[str], labels: dict[str, str]) -> list[dict[str, Any]]:
    """Panels from statistics_tests.csv rows of family rq2_model_vs_human, sample `all`: model - human difference,
    CI, filled when significant after Holm."""
    by = {
        (r["metric"], r["system_a"]): r for r in tests if r["family"] == "rq2_model_vs_human" and r["sample"] == "all"
    }
    return [
        {
            "label": labels[m],
            "rows": {
                system: (float(r["difference"]), float(r["ci_low"]), float(r["ci_high"]), r["significant"] == "True")
                for (metric, system), r in by.items()
                if metric == m
            },
        }
        for m in metrics
    ]
