"""A design rule for validation-based selection, and the committee ceiling.

1. Design rule. Selection reliability depends on two quantities a
   practitioner can estimate: the true gap between the best and second-best
   candidate, and the validation size. Simulation over a grid: two leading
   candidates separated by gap delta (embedded in a pool whose remaining
   members sit below), independent Bernoulli validation outcomes of size n;
   report P(the true best is selected). Produces the reliability surface and
   the operating point of the COLIEE pool (top-of-pool gaps well under one
   point at n=258). Model-based, stated as such; no error-correlation
   assumptions are needed because only single-model selection is simulated.

2. Committee ceiling. From the observed 30-expert predictions on R07: how
   many questions could no nine-member committee answer correctly (fewer
   than five of the thirty experts correct), and what is the best achievable
   nine-committee accuracy.

Gate: reproduces pool and DU3. Writes design_rule_numbers.json and
figures/fig_design_rule.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selection_policy_analysis import (load_gold, build_pool,
                                       committee_correct, DU3_EXPERTS)

HERE = Path(__file__).parent
OUT = HERE / "design_rule_numbers.json"
SEED = 20260704


def main():
    rng = np.random.default_rng(SEED)
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    du3 = [pool.index(e) for e in DU3_EXPERTS]
    if not (len(pool) == 30 and int(
            committee_correct(c_test, np.array([du3]))[0].sum()) == 77):
        print("GATE FAILED")
        sys.exit(1)
    print("gate: PASS")

    results = {}

    # ---- 1. selection-reliability surface ----------------------------------
    # pool model: best at p0, runner-up at p0 - delta, remaining 28 spread
    # uniformly from p0 - delta down to p0 - delta - 0.08 (a tail like ours)
    p0 = 0.87
    deltas = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
    ns = np.array([100, 258, 500, 1_000, 2_500, 5_000])
    B = 4_000
    surface = np.zeros((len(deltas), len(ns)))
    for i, d in enumerate(deltas):
        p = np.concatenate([[p0, p0 - d],
                            np.linspace(p0 - d, p0 - d - 0.08, 28)])
        for j, n in enumerate(ns):
            wins = rng.binomial(int(n), p[None, :].repeat(B, axis=0))
            surface[i, j] = float((wins.argmax(1) == 0).mean())
    results["selection_reliability"] = {
        "model": "best expert at 87%; runner-up delta below; 28-member tail "
                 "spread over the next 8 points; independent Bernoulli "
                 "validation of size n; P(true best selected)",
        "deltas_pp": (deltas * 100).tolist(),
        "n_grid": ns.tolist(),
        "p_select_best": np.round(surface, 3).tolist(),
    }

    # ---- 2. committee ceiling on R07 ---------------------------------------
    counts = c_test.sum(0)  # of 30 experts correct, per question
    unwinnable = int((counts <= 4).sum())
    hard_ids = [test_ids[i] for i in np.where(counts <= 4)[0]]
    hard_counts = [int(counts[i]) for i in np.where(counts <= 4)[0]]
    majority_pool = int((counts >= 16).sum())
    results["committee_ceiling_R07"] = {
        "questions_unwinnable_by_any_9_committee": unwinnable,
        "unwinnable_ids_and_expert_counts": dict(zip(hard_ids, hard_counts)),
        "best_possible_9committee_pc": round(
            float((82 - unwinnable) / 82 * 100), 1),
        "pool_majority_correct_questions": majority_pool,
        "note": "a nine-member majority needs >=5 correct members; with <=4 "
                "of 30 experts correct no nine-committee can be right",
    }

    print(json.dumps(results, indent=2))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure: reliability surface ---------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(deltas)))
    for i, d in enumerate(deltas):
        ax.plot(ns, surface[i] * 100, marker="o", ms=4.5, lw=1.8,
                color=colors[i], label=f"{d*100:.1f} pp")
    ax.axhline(80, color="0.4", lw=1, ls=(0, (4, 3)))
    ax.text(4900, 81.5, "80% reliability", fontsize=9, color="0.35",
            ha="right")
    ax.axvline(258, color="#D55E00", lw=1.2, ls=(0, (4, 3)))
    ax.text(268, 4, "this benchmark\n(n = 258; top-of-pool\ngaps under 1 pp)",
            fontsize=9, color="#D55E00", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Validation size $n$ (questions)")
    ax.set_ylabel("P(true best model selected)  (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="True best-vs-runner-up gap", frameon=False, fontsize=9,
              title_fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 0.02))
    ax.yaxis.grid(True, color="0.92", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "figures/fig_design_rule.png", dpi=300,
                bbox_inches="tight")
    print("written figures/fig_design_rule.png")


if __name__ == "__main__":
    main()
