"""Figure: what the answer-count decision is worth, against ranker quality.

One panel. Seven real rankers and the simulation share a single quality axis, so measured
systems occupy the range they actually reach and the simulation extends beyond them.
Reads expM_count_value_numbers.json and expN_real_rankers_numbers.json.
"""
import json
from pathlib import Path

from figstyle import TEXTWIDTH, INK, ACC, MUT, LIGHT, plt

HERE = Path(__file__).parent
m = json.load(open(HERE / "expM_count_value_numbers.json"))["curve"]
n = json.load(open(HERE / "expN_real_rankers_numbers.json"))

alphas = sorted(float(a) for a in m)
key = lambda a: f"{a:g}" if f"{a:g}" in m else str(a)
qual = [m[key(a)]["best_fixed_F1"] for a in alphas]
val = [m[key(a)]["count_value_pp"] for a in alphas]

feats = [r for r in n["task1_rankers"] if r["kind"] == "first-stage signal"]
models = [r for r in n["task1_rankers"] if r["kind"] == "trained model"]

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.6))

# the region real systems occupy, shaded behind everything
qmax = max(r["best_fixed_F1"] for r in n["task1_rankers"])
ax.axvspan(0.10, qmax, color="#f2f4f6", zorder=0)

ax.plot(qual, val, color="#9fb6cc", lw=1.0, ls=(0, (5, 3)), zorder=2,
        solid_capstyle="round")
ax.scatter([r["best_fixed_F1"] for r in feats], [r["count_value_pp"] for r in feats],
           s=44, facecolors="white", edgecolors=INK, linewidths=1.3, zorder=5)
ax.scatter([r["best_fixed_F1"] for r in models], [r["count_value_pp"] for r in models],
           s=44, color=INK, marker="s", zorder=5)

ax.set_xlim(0.08, 0.72)
ax.set_ylim(0, 29)
ax.set_xlabel("ranker quality (micro-F1 at its best fixed count)")
ax.set_ylabel("count decision worth (points)")

# labels placed in empty space, never over a mark
ax.annotate("first-stage signals", xy=(0.155, 2.1), xytext=(0.155, 7.4),
            fontsize=7.2, color=MUT, ha="center",
            arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.7,
                            shrinkA=2, shrinkB=4))
ax.text(0.60, 19.4, "simulation", fontsize=7.2, color=ACC, ha="center")
ax.text((0.10 + qmax) / 2, 26.6, "range of real systems", fontsize=7.0,
        color="#8a9099", ha="center", style="italic")

# the three trained models fall within seven thousandths of one another, so they
# overlap at this scale. An inset separates them without distorting the main axes.
ins = ax.inset_axes([0.10, 0.36, 0.30, 0.34])
ins.scatter([r["best_fixed_F1"] for r in models], [r["count_value_pp"] for r in models],
            s=30, color=INK, marker="s", zorder=4)
for r in models:
    ins.annotate(r["ranker"].split("(")[-1].rstrip(")").upper(),
                 xy=(r["best_fixed_F1"], r["count_value_pp"]),
                 xytext=(0, 7), textcoords="offset points",
                 fontsize=6, color=MUT, ha="center")
ins.set_xlim(0.3200, 0.3320)
ins.set_ylim(5.2, 8.2)
ins.set_xticks([0.322, 0.326, 0.330])
ins.set_yticks([6, 7, 8])
ins.tick_params(labelsize=5.8, length=2, pad=1.5)
for sp in ins.spines.values():
    sp.set_linewidth(0.6); sp.set_color("#b0b0b0")
ins.set_facecolor("white")
ins.grid(True, lw=0.4, color="#e8e8e8")
ins.set_axisbelow(True)
ax.indicate_inset_zoom(ins, edgecolor="#b0b0b0", linewidth=0.6, alpha=0.9)

fig.savefig(HERE / "fig_countvalue.pdf")
r = n["task1_relation"]
print(f"written: fig_countvalue.pdf  (single panel; real {r['range_count_value_pp']} pts "
      f"over quality {r['range_best_fixed_F1']}, Spearman {r['spearman_quality_vs_count_value']})")
