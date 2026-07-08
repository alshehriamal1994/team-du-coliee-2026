"""System architecture figure: the deployed pipeline at a glance.

Journal-diagram styling: family-tinted chips with architecture tags, a
grouped expert container, tinted module boxes. Content is verbatim from
Table 2: nine experts, six base models, three families.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fig_style
from fig_style import BLUE, VERM, TEXTWIDTH_IN

fig_style.apply()
plt.rcParams.update({"font.size": 8})

FAM = {"DeepSeek": "#0072B2", "Llama": "#D55E00", "Qwen": "#009E73"}

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 64)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec="0.35", fs=8, lw=0.9, ls="-",
        weight="normal", tc="black", pad=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad={pad},rounding_size=1.1",
                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc)


def arrow(x1, y1, x2, y2, ls="-", color="0.4"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=8, linewidth=0.85, color=color,
                 linestyle=ls, shrinkA=1, shrinkB=1))


# ---- input -------------------------------------------------------------
box(1, 36, 15.5, 16, "Bar-exam\nstatement $S$\n+ articles $A$",
    fc="0.965", ec="0.3", fs=7.5, lw=1.0)
box(1, 7, 15.5, 15, "Task 3 only:\nbigram BM25 +\ncross-reference\nexpansion",
    fc="white", ec="0.55", ls=(0, (3, 2)), fs=6.8, tc="0.25")
arrow(8.75, 23.5, 8.75, 34.5, ls=(0, (3, 2)), color="0.55")

# ---- expert container ---------------------------------------------------
ax.add_patch(FancyBboxPatch((22.5, 6.5), 33.5, 51,
             boxstyle="round,pad=0.8,rounding_size=1.6",
             facecolor="none", edgecolor="0.75", linewidth=0.8,
             linestyle=(0, (4, 3))))
ax.text(39.2, 60.5, "Nine experts, six base models",
        ha="center", fontsize=7.5, fontstyle="italic", color="0.3")

experts = [
    ("DeepSeek-R1", "DeepSeek", "MoE"),
    ("Llama-4-Maverick", "Llama", "MoE"),
    ("Llama-4-Scout", "Llama", "MoE"),
    ("Llama-3.3-70B ($\\times$2)", "Llama", "Dense"),
    ("Qwen-2.5-72B", "Qwen", "Dense"),
    ("Qwen3-235B ($\\times$3)", "Qwen", "MoE"),
]
ys = [51.5, 43.5, 35.5, 27.5, 19.5, 11.5]
for (name, fam, arch), y in zip(experts, ys):
    c = FAM[fam]
    box(24.5, y - 2.9, 22.5, 6.1, name, fc=to_rgba(c, 0.10), ec=c,
        lw=1.2, fs=7)
    box(48.3, y - 2.9, 6.2, 6.1, arch, fc=to_rgba(c, 0.28), ec=c,
        lw=1.0, fs=5.8, tc="0.15")
    arrow(17.3, 44, 23.6, y + 0.2)

# ---- vote ---------------------------------------------------------------
box(62, 26, 22, 14, "Unweighted\nmajority vote\none expert, one vote",
    fc=to_rgba(BLUE, 0.10), ec=BLUE, lw=1.4, fs=7.5, weight="bold")
for y in ys:
    arrow(55.4, y + 0.2, 61.1, 32.5)

# ---- outputs ------------------------------------------------------------
box(89, 38, 10, 10, "$Y/N$\nanswer", fc="0.965", ec="0.25", fs=8,
    weight="bold")
box(86.5, 12, 12.5, 14, "Vote margin:\nunanimous\nor contested",
    fc=to_rgba(VERM, 0.08), ec=VERM, fs=6.8, tc=VERM, lw=1.2)
arrow(84.9, 35, 88.1, 41)
arrow(84.9, 30, 87.5, 22.5)

ax.text(50, 1.8, "Runs: DU3 is this vote. DU2 adds a deliberation stage "
        "when the margin is one. DU1 is a meta-ensemble of three "
        "sub-ensembles.", ha="center", fontsize=6.8, color="0.32")

fig_style.save(fig, "fig_architecture")
