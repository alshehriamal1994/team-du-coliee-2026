"""Figures 1 and 2: the selection scatter and the accuracy ladder.

Split into two full-width figures so each has room to read at print size.
"""

import numpy as np

import fig_style
from fig_style import BLUE, VERM, GREY, FAMILY_COLOR, TEXTWIDTH_IN
from selection_policy_analysis import load_gold, build_pool

fig_style.apply()
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5})


def family(name):
    for pref, fam in (("deepseek", "DeepSeek"), ("llama", "Llama"),
                      ("qwen", "Qwen"), ("qwq", "Qwen"),
                      ("mistral", "Mistral"), ("gemma", "Gemma")):
        if name.startswith(pref):
            return fam
    raise ValueError(name)


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
assert pool[vb] == "llama-3.3-70b-instruct_standard_v1"

# ---- Figure 1: the scatter, full width -------------------------------------
fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.92, 3.3))
se_val = np.sqrt(val_acc * (1 - val_acc) / 258) * 100
se_test = np.sqrt(test_acc * (1 - test_acc) / 82) * 100
x, y = val_acc * 100, test_acc * 100

for i in range(len(pool)):
    ax.errorbar(x[i], y[i], xerr=se_val[i], yerr=se_test[i], fmt="none",
                ecolor="0.88", elinewidth=0.6, capsize=0, zorder=1)
seen = set()
for i in range(len(pool)):
    fam = family(pool[i])
    ax.scatter(x[i], y[i], s=30, color=FAMILY_COLOR[fam],
               edgecolor="white", linewidth=0.5, zorder=3,
               label=fam if fam not in seen else None)
    seen.add(fam)
ax.scatter(x[vb], y[vb], s=110, facecolor="none", edgecolor="black",
           linewidth=1.0, zorder=4)
ax.annotate("validation best, 28th of 30 on test",
            xy=(x[vb] - 0.12, y[vb] - 0.55), xytext=(82.6, 79.5),
            fontsize=8.5, color="black", ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6))

ax.set_xlabel("Validation accuracy (%), 258 questions")
ax.set_ylabel("Test accuracy (%), R07, 82 questions")
ax.set_xlim(77.8, 87)
ax.set_ylim(77.5, 97.5)
ax.legend(frameon=False, ncol=5, loc="lower left",
          bbox_to_anchor=(-0.02, 1.0), columnspacing=1.1,
          handletextpad=0.2)
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
