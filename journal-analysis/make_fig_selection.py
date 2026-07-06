"""Figures 1 and 2: the selection scatter and the accuracy ladder.

One message per figure. The scatter carries "validation cannot rank the
pool": all experts in neutral grey, only the two story-bearing points
highlighted, and a single typical-error cross in place of thirty error bars.
The ladder carries the policy comparison with Wilson intervals.
"""

import numpy as np

import fig_style
from fig_style import BLUE, VERM, GREY, TEXTWIDTH_IN
from selection_policy_analysis import load_gold, build_pool

fig_style.apply()
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


gold = load_gold()
pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
val_acc = c_val.mean(axis=1)
test_acc = c_test.mean(axis=1)
vb = int(val_acc.argmax())
tb = int(test_acc.argmax())
assert pool[vb] == "llama-3.3-70b-instruct_standard_v1"

# ---- Figure 1: the scatter, one message ------------------------------------
fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.92, 3.2))
x, y = val_acc * 100, test_acc * 100

others = [i for i in range(len(pool)) if i not in (vb, tb)]
ax.scatter(x[others], y[others], s=26, color="0.62",
           edgecolor="white", linewidth=0.5, zorder=3)

ax.scatter(x[vb], y[vb], s=42, color=VERM, edgecolor="white",
           linewidth=0.6, zorder=4)
ax.scatter(x[vb], y[vb], s=130, facecolor="none", edgecolor=VERM,
           linewidth=1.0, zorder=4)
ax.annotate("best on validation,\n28th of 30 on test",
            xy=(x[vb] - 0.14, y[vb] - 0.6), xytext=(83.4, 79.3),
            fontsize=8.5, color=VERM, ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color=VERM, lw=0.6))

ax.scatter(x[tb], y[tb], s=52, facecolor="white", edgecolor=GREY,
           linewidth=1.0, marker="D", zorder=4)
ax.annotate("best on test\n(hindsight only)",
            xy=(x[tb] + 0.12, y[tb]), xytext=(85.4, 92.6),
            fontsize=8.5, color=GREY, ha="left", va="top",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))

# one typical error cross instead of thirty error bars
se_val = float(np.median(np.sqrt(val_acc * (1 - val_acc) / 258))) * 100
se_test = float(np.median(np.sqrt(test_acc * (1 - test_acc) / 82))) * 100
ex, ey = 78.9, 95.3
ax.errorbar([ex], [ey], xerr=[se_val], yerr=[se_test], fmt="none",
            ecolor="0.45", elinewidth=0.9, capsize=2.5, zorder=3)
ax.text(ex + 0.35, ey - 1.0, "typical $\\pm$1 s.e.", fontsize=8,
        color="0.35", ha="left", va="top")

ax.text(0.03, 0.03, "Pearson $r = 0.43$, 95% CI $[0.09, 0.69]$",
        transform=ax.transAxes, fontsize=8.5, color="0.35", va="bottom")

ax.set_xlabel("Validation accuracy (%), 258 questions")
ax.set_ylabel("Test accuracy (%), R07, 82 questions")
ax.set_xlim(77.8, 87)
ax.set_ylim(77.5, 97.5)
ax.yaxis.grid(True, color="0.93", lw=0.4)
ax.set_axisbelow(True)
fig_style.save(fig, "fig_scatter")

# ---- Figure 2: the ladder, full width ---------------------------------------
fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.92, 2.1))
rows = [
    ("Validation-selected single model", 68, VERM, "o", False),
    ("Plain nine-expert vote (DU3, reproduced)", 77, BLUE, "o", False),
    ("Official winning runs (DU1/DU2)", 79, BLUE, "s", False),
    ("Hindsight-best single (oracle, unknowable)", 78, GREY, "D", True),
]
ypos = [3, 2, 1, 0]
for (label, k, color, marker, open_m), yp in zip(rows, ypos):
    lo, hi = wilson(k, 82)
    pc = k / 82 * 100
    ax.plot([lo, hi], [yp, yp], color=color, lw=1.8, alpha=0.45,
            solid_capstyle="round", ls=(0, (3, 2)) if open_m else "-",
            zorder=2)
    ax.scatter([pc], [yp], s=42, color="white" if open_m else color,
               edgecolor=color, linewidth=1.0, marker=marker, zorder=3)
    ax.text(hi + 1.0, yp, f"{pc:.1f}", va="center", fontsize=9,
            fontweight="bold", color=color)

ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("R07 test accuracy (%)")
ax.set_xlim(64, 102)
ax.set_ylim(-0.55, 3.55)
ax.xaxis.grid(True, color="0.93", lw=0.4)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

ax.plot([68.5, 68.5], [2.15, 2.85], color="0.35", lw=0.8)
ax.text(67.6, 2.5, "McNemar $p = 0.012$", ha="right", va="center",
        fontsize=8, color="0.25")
ax.plot([68.5, 68.5], [1.15, 1.85], color="0.35", lw=0.8)
ax.text(67.6, 1.5, "$p = 0.5$ (n.s.)", ha="right", va="center",
        fontsize=8, color="0.25")
fig_style.save(fig, "fig_ladder")
