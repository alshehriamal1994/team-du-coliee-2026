"""Shared figure style for the manuscript: print-first, serif, vector.

Figures are drawn at their true printed width (the sn-jnl text block is
about 5.5 inches) so fonts render at genuine point sizes. STIX serif
matches the paper's typography and ships with matplotlib. Colours are the
colour-blind-safe pair used throughout the paper.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#0072B2"
VERM = "#D55E00"
GREY = "#5A5A5A"
FAMILY_COLOR = {
    "DeepSeek": "#0072B2",
    "Llama": "#D55E00",
    "Qwen": "#009E73",
    "Mistral": "#E69F00",
    "Gemma": "#CC79A7",
}

TEXTWIDTH_IN = 5.5


def apply():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, stem):
    fig.savefig(f"figures/{stem}.pdf")
    fig.savefig(f"figures/{stem}.png", dpi=350)
    print(f"written figures/{stem}.pdf and .png")
