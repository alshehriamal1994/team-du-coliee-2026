"""Figure 2 (fig_selection_policy): random committees versus the
validation-selected single expert, as committee size grows.

Single panel. The bootstrap deployed-accuracy distributions are reported in
the policy table of the manuscript; this figure carries the one result a
table cannot show at a glance: as committee size grows, even the worst
randomly composed committee clears the validation-selected single model.

Reads selection_policy_numbers.npz written by selection_policy_analysis.py.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = np.load(HERE / "selection_policy_numbers.npz")

BLUE = "#0072B2"
VERM = "#D55E00"
GREY = "#5A5A5A"
plt.rcParams.update({
    "font.size": 11.5,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(7.6, 4.6))

ks = [1, 3, 5, 7, 9, 11, 13, 15]
qs = {p: [] for p in (5, 25, 50, 75, 95)}
mins = []
for k in ks:
    acc = d[f"size_k{k}"] * 100
    for p in qs:
        qs[p].append(np.percentile(acc, p))
    mins.append(acc.min())

ax.fill_between(ks, qs[5], qs[95], color=BLUE, alpha=0.15, lw=0,
                label="5th–95th percentile of random committees")
ax.fill_between(ks, qs[25], qs[75], color=BLUE, alpha=0.32, lw=0,
                label="25th–75th percentile")
ax.plot(ks, qs[50], color=BLUE, lw=2.2, marker="o", ms=5, label="median")
ax.plot(ks, mins, color=BLUE, lw=1.2, ls=":",
        label="worst sampled committee")

oracle = 78 / 82 * 100
single = 68 / 82 * 100
ax.axhline(single, color=VERM, lw=1.6, ls=(0, (4, 3)))
ax.text(0.75, single - 0.5, "validation-selected single expert  82.9",
        fontsize=10.5, color=VERM, va="top", ha="left")
ax.axhline(oracle, color=GREY, lw=1.3, ls=(0, (4, 3)))
ax.text(0.75, oracle + 0.4, "hindsight-best single expert  95.1",
        fontsize=10.5, color=GREY, va="bottom", ha="left")

ax.set_xlabel("Committee size $k$ (members drawn at random, no selection)")
ax.set_ylabel("R07 test accuracy (%)")
ax.set_xticks(ks)
ax.set_xlim(0.4, 15.6)
ax.set_ylim(77.5, 99.5)
ax.legend(frameon=False, fontsize=9.5, loc="lower right",
          bbox_to_anchor=(1.0, 0.0), labelspacing=0.4, handlelength=1.8)
ax.yaxis.grid(True, color="0.92", lw=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(HERE / "figures/fig_selection_policy.png", dpi=300,
            bbox_inches="tight")
print("written figures/fig_selection_policy.png")
