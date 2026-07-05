"""Figure 1: ensembling as selection-risk reduction (fig_selection_risk).

Panel (a): validation vs test accuracy for the 30-expert pool, coloured by
model family, with +/-1 binomial standard error bars on both axes and the
validation-best expert circled.
Panel (b): the accuracy ladder on R07 as a dot plot with Wilson 95% intervals
and paired McNemar p-values (deployable single vs plain vote p=0.012; plain
vote vs official run p=0.5).

All quantities recomputed from the prediction logs at draw time; the McNemar
p-values are from gate1_audit_numbers.json (verified 2026-07-04).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selection_policy_analysis import load_gold, build_pool

HERE = Path(__file__).parent

FAMILY_COLOR = {
    "DeepSeek": "#0072B2",
    "Llama": "#D55E00",
    "Qwen": "#009E73",
    "Mistral": "#E69F00",
    "Gemma": "#CC79A7",
}
BLUE = "#0072B2"
VERM = "#D55E00"
GREY = "#5A5A5A"
plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# ---- panel (a): scatter with error bars, coloured by family ----------------
se_val = np.sqrt(val_acc * (1 - val_acc) / 258) * 100
se_test = np.sqrt(test_acc * (1 - test_acc) / 82) * 100
x = val_acc * 100
y = test_acc * 100

for i in range(len(pool)):
    ax1.errorbar(x[i], y[i], xerr=se_val[i], yerr=se_test[i],
                 fmt="none", ecolor="0.82", elinewidth=0.9, capsize=0,
                 zorder=1)
seen = set()
for i in range(len(pool)):
    fam = family(pool[i])
    ax1.scatter(x[i], y[i], s=52, color=FAMILY_COLOR[fam],
                edgecolor="white", linewidth=0.7, zorder=3,
                label=fam if fam not in seen else None)
    seen.add(fam)

ax1.scatter(x[vb], y[vb], s=190, facecolor="none", edgecolor="black",
            linewidth=1.6, zorder=4)
ax1.annotate("validation-best\n(28th of 30 on test)",
             xy=(x[vb] - 0.15, y[vb] - 0.55), xytext=(83.2, 79.9),
             fontsize=9, color="black", ha="center", va="top",
             arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

ax1.set_xlabel("Validation accuracy (%)  [258 questions]")
ax1.set_ylabel("Test accuracy (%)  [R07, 82 questions]")
ax1.set_title("(a)  Validation cannot rank the pool "
              "(r = 0.43, 95% CI [0.09, 0.69])", loc="left", pad=26)
ax1.set_xlim(77.8, 87)
ax1.set_ylim(77.5, 97.5)
ax1.legend(frameon=False, fontsize=8.5, ncol=5, loc="lower left",
           bbox_to_anchor=(-0.02, 1.0), columnspacing=0.9, handletextpad=0.2)
ax1.yaxis.grid(True, color="0.92", lw=0.7)
ax1.set_axisbelow(True)

# ---- panel (b): accuracy ladder with Wilson intervals ----------------------
rows = [
    ("Validation-selected\nsingle model", 68, VERM, "o"),
    ("Plain 9-expert vote\n(DU3, reproduced)", 77, BLUE, "o"),
    ("Official winning runs\n(DU1/DU2)", 79, BLUE, "s"),
    ("Hindsight-best single\n(oracle, unknowable)", 78, GREY, "D"),
]
ypos = [3, 2, 1, 0]
for (label, k, color, marker), yp in zip(rows, ypos):
    lo, hi = wilson(k, 82)
    pc = k / 82 * 100
    open_marker = (label.startswith("Hindsight"))
    ax2.plot([lo, hi], [yp, yp], color=color, lw=2.2, alpha=0.45,
             solid_capstyle="round", zorder=2,
             ls=(0, (3, 2)) if open_marker else "-")
    ax2.scatter([pc], [yp], s=95, color="white" if open_marker else color,
                edgecolor=color, linewidth=1.6, marker=marker, zorder=3)
    ax2.text(hi + 0.7, yp, f"{pc:.1f}", va="center", fontsize=10.5,
             fontweight="bold", color=color)

ax2.set_yticks(ypos)
ax2.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax2.set_xlabel("R07 test accuracy (%), Wilson 95% interval")
ax2.set_title("(b)  The accuracy ladder, with uncertainty", loc="left",
              pad=10)
ax2.set_xlim(66, 104)
ax2.set_ylim(-0.6, 3.6)
ax2.xaxis.grid(True, color="0.92", lw=0.7)
ax2.set_axisbelow(True)
ax2.spines["left"].set_visible(False)
ax2.tick_params(axis="y", length=0)

# paired McNemar annotations (gate1_audit_numbers.json)
ax2.annotate("", xy=(70.5, 2.12), xytext=(70.5, 2.88),
             arrowprops=dict(arrowstyle="-", color="0.35", lw=1.0))
ax2.text(69.6, 2.5, "McNemar\np = 0.012", ha="right", va="center",
         fontsize=8.5, color="0.25")
ax2.annotate("", xy=(70.5, 1.12), xytext=(70.5, 1.88),
             arrowprops=dict(arrowstyle="-", color="0.35", lw=1.0))
ax2.text(69.6, 1.5, "p = 0.5\n(n.s.)", ha="right", va="center",
         fontsize=8.5, color="0.25")

fig.tight_layout(w_pad=4.0)
fig.savefig(HERE / "figures/fig_selection_risk.png", dpi=300,
            bbox_inches="tight")
print("written figures/fig_selection_risk.png")
