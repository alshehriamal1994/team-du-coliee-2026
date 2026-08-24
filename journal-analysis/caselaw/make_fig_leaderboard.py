"""Figure: the audited leaderboards, and the null that governs how to read them.

Rebuilt 2026-08-11. The previous version had colliding panel titles, a rotated gold-mean
label printed over the data, and two filled-marker classes against a caption claiming one.

Design. Short panel titles. The correlation is stated inside the panel, together with the
verdict the permutation null returns for that task, so a reader cannot take the two
coefficients at equal face value. The winner has its own glyph so the figure survives
greyscale. Gold-mean label sits above the axis, horizontal, clear of every mark.

Reads expH_runs.csv, expH_numbers.json, expH2_null_numbers.json.
"""
import csv
import json
from pathlib import Path

from matplotlib.ticker import ScalarFormatter
from figstyle import TEXTWIDTH, INK, ACC, MUT, plt

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "expH_runs.csv")))
null = json.load(open(HERE / "expH2_null_numbers.json"))["tasks"]

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.45))
fig.subplots_adjust(wspace=0.30)

specs = [
    (axes[0], "T1", "Task 1", 4.375, {"du3", "du2", "du1"}, [1, 2, 5, 10, 20],
     "survives the null", (0.055, 0.93)),
    (axes[1], "T2", "Task 2", 2.94, {"task2_du3", "task2_du2", "task2_du1"},
     [0.5, 1, 2, 5, 8], "does not survive it", (0.055, 0.93)),
]

for ax, task, label, gmean, du_runs, ticks, verdict, notepos in specs:
    pts = [r for r in rows if r["task"] == task]
    rho = null[task]["observed_spearman_F1_vs_closeness"]

    ax.scatter([float(r["avg_set_size"]) for r in pts], [float(r["F1"]) for r in pts],
               s=15, facecolors="none", edgecolors="#9a9a9a", lw=0.8, zorder=2)
    for r in pts:
        x, y = float(r["avg_set_size"]), float(r["F1"])
        if r["run"] in du_runs:
            ax.scatter([x], [y], s=24, color=INK, zorder=4)
        if r["rank"] == "1":
            ax.scatter([x], [y], s=44, color=ACC, marker="D", zorder=5,
                       edgecolors="white", linewidths=0.6)

    ax.axvline(gmean, color=ACC, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("average answer-set size")
    ax.set_title(f"({'ab'[task == 'T2']})  {label}", loc="left", pad=14)

    # gold mean named above the plotting area, horizontal, clear of the data
    ax.annotate(f"gold mean {gmean:.2f}", xy=(gmean, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.0, color=ACC)

    # the coefficient and the verdict, inside the panel, in empty space


axes[0].set_ylabel("official micro-F1")
axes[0].annotate("our runs", xy=(5.0, 0.3141), xytext=(9.0, 0.365),
                 fontsize=7.2, color=INK, ha="center",
                 arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.7, shrinkB=4))
axes[1].annotate("our runs", xy=(0.97, 0.325), xytext=(1.55, 0.275),
                 fontsize=7.2, color=INK, ha="center",
                 arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.7, shrinkB=4))
axes[1].scatter([], [], s=44, color=ACC, marker="D", label="winner")
axes[1].scatter([], [], s=24, color=INK, label="our runs")

fig.savefig(HERE / "fig_leaderboard.pdf")
print(f"written: fig_leaderboard.pdf  (T1 rho={null['T1']['observed_spearman_F1_vs_closeness']}, "
      f"T2 rho={null['T2']['observed_spearman_F1_vs_closeness']})")
