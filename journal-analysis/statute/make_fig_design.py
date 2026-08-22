"""Figure 3 (appendix): the selection reliability surface."""

import json

import numpy as np

import fig_style
from fig_style import VERM, TEXTWIDTH_IN

fig_style.apply()
import matplotlib.pyplot as plt

d = json.load(open("design_rule_numbers.json"))["selection_reliability"]
deltas = d["deltas_pp"]
ns = d["n_grid"]
surface = np.array(d["p_select_best"])

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.82, 2.7))
colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(deltas)))
for i, dlt in enumerate(deltas):
    ax.plot(ns, surface[i] * 100, marker="o", ms=2.6, lw=1.0,
            color=colors[i], label=f"{dlt:.1f} pp")

ax.axhline(80, color="0.4", lw=0.7, ls=(0, (4, 3)))
ax.text(4900, 81.5, "80% reliability", fontsize=7, color="0.35", ha="right")
ax.axvline(258, color=VERM, lw=0.8, ls=(0, (4, 3)))
ax.text(268, 3, "this benchmark, $n=258$,\ntop-of-pool gaps under 1 pp",
        fontsize=7, color=VERM, va="bottom")

ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()
ax.set_xlabel("Validation size $n$ (questions)")
ax.set_ylabel("P(true best model selected) (%)")
ax.set_ylim(0, 100)
ax.legend(title="True gap, best against runner-up", frameon=False,
          loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=6.8,
          title_fontsize=6.8, labelspacing=0.3, handlelength=1.5)
ax.yaxis.grid(True, color="0.93", lw=0.4)
ax.set_axisbelow(True)

fig_style.save(fig, "fig_design_rule")
