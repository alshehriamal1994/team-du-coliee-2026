"""Figure: the four-way decomposition of the micro-F1 gap, both tasks.

Reads expL_decomposition_numbers.json only. Authored at the manuscript's text width
(5.14 in) with font.size 8, so it prints at scale 1 and its lettering matches the captions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
INK, ACC = "#333333", "#4477aa"
GREY_A, GREY_B, ACC_LT = "#8d8d8d", "#c9c9c9", "#8fb4d0"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#999999", "xtick.color": "#666666", "ytick.color": "#333333",
    "axes.labelcolor": "#333333", "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": "#e8e8e8", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 300, "pdf.fonttype": 42, "savefig.bbox": "tight",
})

d = json.load(open(HERE / "expL_decomposition_numbers.json"))
rows = [("Task 2", d["task2_v2"]), ("Task 1", d["task1"])]   # Task 1 drawn on top

fig, ax = plt.subplots(figsize=(5.14, 2.15))
ypos = [0, 1]

for y, (name, t) in zip(ypos, rows):
    c = t["components_pp"]
    segs = [
        (t["deployed_constant"] * 100, INK,    "achieved"),
        (c["1_better_constant"],       ACC,    "a better constant"),
        (c["2_per_query_adaptivity"],  ACC_LT, "per-query count"),
        (c["3_ranking_within_list"],   GREY_A, "ranking in the list"),
        (c["4_gold_never_retrieved"],  GREY_B, "below evaluated depth"),
    ]
    left = 0.0
    for w, col, _ in segs:
        if w <= 0:
            continue
        ax.barh(y, w, left=left, height=0.46, color=col,
                edgecolor="white", linewidth=0.8)
        if w > 5:
            ax.text(left + w / 2, y, f"{w:.1f}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        left += w
    # the zero-width Task 1 constant segment is itself a finding
    if c["1_better_constant"] == 0.0:
        ax.annotate("0.0  five was already\nthe best constant",
                    xy=(t["deployed_constant"] * 100, y + 0.23),
                    xytext=(t["deployed_constant"] * 100 - 2, y + 0.52),
                    fontsize=7, color="#555555", ha="center",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))
    elif c["1_better_constant"] <= 5:
        ax.text(t["deployed_constant"] * 100 + c["1_better_constant"] / 2, y + 0.30,
                f"{c['1_better_constant']:.1f}", ha="center", fontsize=7, color=ACC)
    # a per-query segment too narrow for an inline label is labelled above the bar
    if 0 < c["2_per_query_adaptivity"] <= 5:
        x_mid = (t["deployed_constant"] * 100 + c["1_better_constant"]
                 + c["2_per_query_adaptivity"] / 2)
        ax.text(x_mid + 1.2, y - 0.42, f"{c['2_per_query_adaptivity']:.1f}",
                ha="center", fontsize=7, color=ACC_LT)

ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 100)
ax.set_xlabel("micro-F1 (points)")
ax.set_ylim(-0.55, 1.85)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (INK, ACC, ACC_LT, GREY_A, GREY_B)]
labels = ["achieved", "a better constant", "per-query count",
          "ranking in the list", "below evaluated depth"]
ax.legend(handles, labels, frameon=False, fontsize=7, ncol=5,
          loc="lower center", bbox_to_anchor=(0.5, -0.42), handlelength=1.1,
          columnspacing=1.0, handletextpad=0.4)

fig.savefig(HERE / "fig_decomp.pdf")
t1, t2 = d["task1"]["components_pp"], d["task2_v2"]["components_pp"]
print("written: fig_decomp.pdf")
print(f"  Task 1  count {t1['1_better_constant']+t1['2_per_query_adaptivity']:.1f}"
      f"  ranking+recall {t1['3_ranking_within_list']+t1['4_gold_never_retrieved']:.1f}")
print(f"  Task 2  count {t2['1_better_constant']+t2['2_per_query_adaptivity']:.1f}"
      f"  ranking+recall {t2['3_ranking_within_list']+t2['4_gold_never_retrieved']:.1f}")
