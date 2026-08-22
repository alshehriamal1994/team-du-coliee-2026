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

ax.plot(qual, val, color=ACC, lw=1.6, zorder=2, solid_capstyle="round")
ax.scatter([r["best_fixed_F1"] for r in feats], [r["count_value_pp"] for r in feats],
           s=26, facecolors="white", edgecolors=INK, linewidths=1.0, zorder=4)
ax.scatter([r["best_fixed_F1"] for r in models], [r["count_value_pp"] for r in models],
           s=26, color=INK, marker="s", zorder=4)

ax.set_xlim(0.08, 0.72)
ax.set_ylim(0, 29)
ax.set_xlabel("ranker quality (micro-F1 at its best fixed count)")
ax.set_ylabel("count decision worth (points)")

# labels placed in empty space, never over a mark
ax.annotate("first-stage signals", xy=(0.155, 2.1), xytext=(0.155, 7.4),
            fontsize=7.2, color=MUT, ha="center",
            arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.7,
                            shrinkA=2, shrinkB=4))
ax.annotate("trained rankers", xy=(0.327, 6.6), xytext=(0.245, 13.0),
            fontsize=7.2, color=MUT, ha="center",
            arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.7,
                            shrinkA=2, shrinkB=4))
ax.text(0.60, 19.4, "simulation", fontsize=7.2, color=ACC, ha="center")
ax.text((0.10 + qmax) / 2, 26.6, "range of real systems", fontsize=7.0,
        color="#8a9099", ha="center", style="italic")

fig.savefig(HERE / "fig_countvalue.pdf")
r = n["task1_relation"]
print(f"written: fig_countvalue.pdf  (single panel; real {r['range_count_value_pp']} pts "
      f"over quality {r['range_best_fixed_F1']}, Spearman {r['spearman_quality_vs_count_value']})")
