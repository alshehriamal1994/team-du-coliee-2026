"""Shared figure style for the manuscript.

One authored width, one type size, one palette, one panel-title idiom. Every figure is
authored at the text measure so it prints at scale 1 and its lettering matches the captions.
Panel titles are short. Quantities go inside the axes as annotations, never in the title,
because long titles are what collide.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH = 5.14          # sn-jnl text measure, inches
INK = "#333333"
ACC = "#4477aa"
MUT = "#6b6b6b"
LIGHT = "#c9c9c9"
FAINT = "#e8e8e8"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9a9a9a",
    "axes.linewidth": 0.8,
    "xtick.color": "#6b6b6b",
    "ytick.color": "#6b6b6b",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": FAINT,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def panel_title(ax, text):
    """Short, left-aligned, placed above the axes so it cannot collide with a neighbour."""
    ax.set_title(text, loc="left", pad=6)


def note(ax, x, y, text, color=MUT, ha="left", va="center", size=7.2):
    """An annotation in axes coordinates, so it is placed relative to empty space."""
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=size,
                   color=color, ha=ha, va=va)


def figure(ncols=1, height=2.3):
    """A figure at the text measure. Two panels get generous horizontal room."""
    fig, ax = plt.subplots(1, ncols, figsize=(TEXTWIDTH, height))
    if ncols > 1:
        fig.subplots_adjust(wspace=0.34)
    return fig, ax
