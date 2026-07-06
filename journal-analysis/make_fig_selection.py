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

# ---- Figure 1: the rank slopegraph ------------------------------------------
# Rank = 1 + number strictly greater (the text's rule). For display, tied
# accuracies are ordered so the highlighted expert sits at its text rank.
def display_order(accs, favoured):
    key = [(-accs[i], 0 if i == favoured else 1, i) for i in range(len(accs))]
    order = sorted(range(len(accs)), key=lambda i: key[i])
    pos = np.empty(len(accs), int)
    for r, i in enumerate(order):
        pos[i] = r + 1
    return pos


rv = display_order(val_acc, vb)
rt = display_order(test_acc, vb)
assert rv[vb] == 1 and rt[vb] == 28

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.92, 3.6))
for i in range(len(pool)):
    if i in (vb, tb):
        continue
    ax.plot([0, 1], [rv[i], rt[i]], color="0.80", lw=0.9, zorder=2)
    ax.scatter([0, 1], [rv[i], rt[i]], s=11, color="0.62", zorder=3)

ax.plot([0, 1], [rv[tb], rt[tb]], color=GREY, lw=1.6, zorder=4,
        ls=(0, (4, 2)))
ax.scatter([0], [rv[tb]], s=30, color=GREY, zorder=5)
ax.scatter([1], [rt[tb]], s=46, facecolor="white", edgecolor=GREY,
           linewidth=1.1, marker="D", zorder=5)
ax.text(1.05, rt[tb], "best on test (hindsight only)", ha="left",
        va="center", fontsize=8.5, color=GREY)

ax.plot([0, 1], [rv[vb], rt[vb]], color=VERM, lw=2.4, zorder=6)
ax.scatter([0, 1], [rv[vb], rt[vb]], s=42, color=VERM, zorder=7)
ax.text(-0.08, rv[vb], "1st on validation", ha="right", va="center",
        fontsize=9.5, fontweight="bold", color=VERM)
ax.text(1.05, rt[vb], "28th of 30 on test", ha="left", va="center",
        fontsize=9.5, fontweight="bold", color=VERM)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Rank on validation\n(258 questions)",
                    "Rank on test\n(R07, 82 questions)"], fontsize=9)
ax.set_ylim(31.2, 0)
ax.set_yticks([1, 10, 20, 30])
ax.set_ylabel("Rank among the 30 experts")
ax.set_xlim(-0.52, 1.62)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0)
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
