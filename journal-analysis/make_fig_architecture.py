"""System architecture figure: the deployed pipeline at a glance.

One horizontal flow in the paper's own visual language: input, the
retrieval stage that Task 3 adds, the nine experts grouped by base model,
the unweighted vote, and the two outputs (answer and margin signal). The
run ladder (DU3/DU2/DU1) is summarised beneath the vote.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import fig_style
from fig_style import BLUE, VERM, GREY, TEXTWIDTH_IN

fig_style.apply()
plt.rcParams.update({"font.size": 8})

FAM = {"DeepSeek": "#0072B2", "Llama": "#D55E00", "Qwen": "#009E73"}

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec="0.35", fs=8, lw=0.9, ls="-",
        weight="normal", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.6,rounding_size=1.2",
                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc)


def arrow(x1, y1, x2, y2, ls="-", color="0.3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=9, linewidth=0.9, color=color,
                 linestyle=ls, shrinkA=1, shrinkB=1))


# input
box(1, 34, 16, 17, "Bar-exam\nstatement $S$\n+ articles $A$", fs=7.5)
# retrieval (Task 3 only)
box(1, 6, 16, 16, "Task 3 only:\nbigram BM25\n+ cross-reference\nexpansion", ec="0.5",
    ls=(0, (3, 2)), fs=7, tc="0.25")
arrow(9, 22.5, 9, 32.5, ls=(0, (3, 2)), color="0.5")

# experts
experts = [
    ("DeepSeek-R1", "DeepSeek"),
    ("Llama-4-Maverick", "Llama"), ("Llama-4-Scout", "Llama"),
    ("Llama-3.3-70B (2 prompts)", "Llama"),
    ("Qwen-2.5-72B", "Qwen"), ("Qwen3-235B (3 prompts)", "Qwen"),
]
ys = [52, 44, 36, 28, 20, 12]
for (name, fam), y in zip(experts, ys):
    box(25, y - 3, 28, 6.4, name, fc="white", ec=FAM[fam], lw=1.2, fs=7)
    arrow(17.8, 42, 24.2, y + 0.2)
ax.text(39, 58.5, "Nine experts, six base models", ha="center", fontsize=7.5,
        fontstyle="italic", color="0.25")

# vote
box(60, 25, 23, 14, "Unweighted\nmajority vote\none expert, one vote",
    fc="#EBF2FA", ec=BLUE, lw=1.3, fs=7.5, weight="bold")
for y in ys:
    arrow(53.8, y + 0.2, 59.2, 32)

# outputs
box(88, 37, 11, 10, "$Y/N$\nanswer", ec="0.3", fs=8, weight="bold")
box(85.5, 12, 13.5, 14, "Vote margin:\nunanimous\nor contested", ec=VERM, fs=7,
    tc=VERM)
arrow(83.8, 34, 87.2, 40)
arrow(83.8, 30, 86, 22)

# run ladder note
ax.text(50, 2.5, "Runs: DU3 is this vote. DU2 adds a deliberation stage when the margin is one. DU1 is a meta-ensemble of three sub-ensembles.",
        ha="center", fontsize=6.8, color="0.3")

fig_style.save(fig, "fig_architecture")
