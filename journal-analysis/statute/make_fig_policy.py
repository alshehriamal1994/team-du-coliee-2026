"""Figure 2: random ensembles against the validation-selected single model."""

import numpy as np

import fig_style
from fig_style import BLUE, VERM, GREY, TEXTWIDTH_IN

fig_style.apply()
import matplotlib.pyplot as plt

d = np.load("selection_policy_numbers.npz")

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.82, 2.6))

ks = [1, 3, 5, 7, 9, 11, 13, 15]
qs = {p: [] for p in (5, 25, 50, 75, 95)}
mins = []
for k in ks:
    acc = d[f"size_k{k}"] * 100
    for p in qs:
        qs[p].append(np.percentile(acc, p))
    mins.append(acc.min())

ax.fill_between(ks, qs[5], qs[95], color=BLUE, alpha=0.14, lw=0,
                label="5th to 95th percentile")
ax.fill_between(ks, qs[25], qs[75], color=BLUE, alpha=0.30, lw=0,
                label="25th to 75th percentile")
ax.plot(ks, qs[50], color=BLUE, lw=1.3, marker="o", ms=3, label="median")
ax.plot(ks, mins, color=BLUE, lw=0.8, ls=":")
ax.text(15.35, mins[-1] - 0.4, "worst sampled\nensemble", fontsize=6.8,
        color=BLUE, ha="right", va="top")

oracle = 78 / 82 * 100
single = 68 / 82 * 100
ax.axhline(single, color=VERM, lw=0.9, ls=(0, (4, 3)))
ax.text(0.75, single - 0.6, "validation-selected single expert, 82.9",
        fontsize=7, color=VERM, va="top", ha="left")
ax.axhline(oracle, color=GREY, lw=0.8, ls=(0, (4, 3)))
ax.text(0.75, oracle + 0.5, "hindsight-best single expert, 95.1",
        fontsize=7, color=GREY, va="bottom", ha="left")

ax.set_xlabel("Ensemble size $k$, members drawn at random")
ax.set_ylabel("R07 test accuracy (%)")
ax.set_xticks(ks)
ax.set_xlim(0.4, 15.6)
ax.set_ylim(77, 99.5)
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
          labelspacing=0.3, handlelength=1.5, fontsize=6.8)
ax.yaxis.grid(True, color="0.93", lw=0.4)
ax.set_axisbelow(True)

fig_style.save(fig, "fig_selection_policy")
