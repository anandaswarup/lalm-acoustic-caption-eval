"""Qualitative-analysis figures (E19): E17 categories and E18 audio ratings per model, and the E17 x E18 heatmap.

Light-mode print figures. E17 categories use the first six categorical slots in fixed order; E18 ratings, which are
ordinal, a three-step blue ramp; the heatmap a sequential blue ramp. Both palettes were checked with the dataviz
validator (categorical: adjacent CVD and normal-vision separation pass, three slots are below 3:1 contrast, so every
segment carries its count; ordinal: monotone lightness and visible steps). Saved without creation dates so reruns
give identical files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

SURFACE, TEXT, TEXT_SECONDARY, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#c3c2b7"
CATEGORY_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
RATING_COLORS = ("#86b6ef", "#2a78d6", "#104281")
SEQUENTIAL = ("#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
METADATA = {"pdf": {"CreationDate": None, "Producer": None}, "png": {"Software": None}}


def _ink_on(color: str) -> str:
    """Dark or white text, whichever reads on the fill (relative luminance threshold)."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color))
    return TEXT if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else "#ffffff"


def _style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8)
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=8.5)


def stacked_bars(
    counts: dict[str, list[int]], labels: list[str], colors: tuple[str, ...], title: str, total_label: str
) -> plt.Figure:
    """Horizontal stacked bars, one per model (first model on top), segments in code order with their counts.
    `counts` maps a model's display name to its counts per code."""
    models = list(counts)
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(models) + 1.4), facecolor=SURFACE)
    for row, model in enumerate(reversed(models)):
        left = 0
        for value, color in zip(counts[model], colors, strict=True):
            if value:
                ax.barh(row, value, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
                if value >= 2:
                    ax.text(left + value / 2, row, str(value), ha="center", va="center", fontsize=7.5,
                            color=_ink_on(color))  # fmt: skip
            left += value
    total = max(sum(values) for values in counts.values())
    ax.set_xlim(0, total)
    ax.set_yticks(range(len(models)), list(reversed(models)))
    ax.set_xlabel(total_label, color=TEXT_SECONDARY, fontsize=8)
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=7.5,
              labelcolor=TEXT, handlelength=1, borderaxespad=0)  # fmt: skip
    ax.set_title(title, loc="left", color=TEXT, fontsize=9.5, pad=8 + 13 * -(-len(labels) // 3))
    fig.tight_layout()
    return fig


def heatmap(
    counts: list[list[int]],
    row_labels: list[str],
    column_labels: list[str],
    title: str | None,
    xlabel: str,
    ylabel: str | None = None,
) -> plt.Figure:
    """Annotated count heatmap (rows x columns), 2 px gaps between cells.

    Drawn at its printed size, one 3.4 in column of a two-column paper, with 9 pt text, so it is never scaled down;
    labels wrap onto two lines. With `title` None the figure has no title (the paper caption has one).
    """
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    peak = max(max(row) for row in counts)
    fig, ax = plt.subplots(figsize=(3.4, 0.36 * len(row_labels) + 0.8), facecolor=SURFACE)
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            color = cmap(value / peak if peak else 0)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2))
            hex_color = matplotlib.colors.to_hex(color)
            ax.text(j + 0.5, i + 0.5, str(value), ha="center", va="center", fontsize=9, color=_ink_on(hex_color))
    ax.set_xlim(0, len(column_labels))
    ax.set_ylim(len(row_labels), 0)
    ax.set_xticks([j + 0.5 for j in range(len(column_labels))], [textwrap.fill(c, 10) for c in column_labels])
    ax.set_yticks([i + 0.5 for i in range(len(row_labels))], [textwrap.fill(r, 14) for r in row_labels])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    _style(ax)
    ax.tick_params(labelsize=9)
    ax.spines["bottom"].set_visible(False)
    if title:
        fig.suptitle(title, x=0.02, ha="left", color=TEXT, fontsize=9.5)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, stem: Path, formats: list[str], dpi: int) -> list[Path]:
    paths = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, facecolor=SURFACE, metadata=METADATA.get(fmt))
        paths.append(path)
    plt.close(fig)
    return paths
