"""Recompute the two Section 7 numbers that were not in any ledger:

  (1) McNemar (exact, two-sided) and a bootstrap CI for the nine-expert vote
      against the best-retrieved single expert of the nine (the 71 vs 75 pair).
  (2) The individual retrieved accuracies of the Qwen3-235B members, to check
      the claimed "86.6 to 91.5%" range.

Reuses the same loaders and retrieved-prediction files as
task3_gap_analysis.py / deltaq_ci_analysis.py, so it is the same data the rest
of Section 7 rests on. Gate: oracle vote 77/82, retrieved vote 71/82.
Writes section7_verify_numbers.json.
"""

import json
from math import comb
from pathlib import Path

import numpy as np

from selection_policy_analysis import load_gold, build_pool, DU3_EXPERTS
from deltaq_ci_analysis import RETR_FILE, RETR

HERE = Path(__file__).parent
SEED = 20260713


def mcnemar_exact_two_sided(b, c):
    """Exact binomial McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def main():
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    du3 = [pool.index(e) for e in DU3_EXPERTS]

    # oracle correctness for the nine, in DU3_EXPERTS order
    oracle_correct = c_test[du3].astype(bool)          # [9, 82]
    oracle_vote = oracle_correct.sum(0) > 4.5

    # retrieved correctness for the nine, in DU3_EXPERTS order
    retr_correct = []
    for e in DU3_EXPERTS:
        pr = {}
        for line in open(RETR / RETR_FILE[e]):
            d = json.loads(line)
            pr[d["id"]] = d["label"]
        retr_correct.append([pr.get(q) == gold[q] for q in test_ids])
    retr_correct = np.array(retr_correct, dtype=bool)   # [9, 82]
    retr_vote = retr_correct.sum(0) > 4.5

    assert int(oracle_vote.sum()) == 77, int(oracle_vote.sum())
    assert int(retr_vote.sum()) == 71, int(retr_vote.sum())

    oracle_acc = oracle_correct.mean(1) * 100
    retr_acc = retr_correct.mean(1) * 100

    # ---- (1) vote vs best-retrieved single of the nine --------------------
    best_i = int(retr_acc.argmax())
    best_single_retr = retr_correct[best_i]
    n_best = int(best_single_retr.sum())
    ties = [DU3_EXPERTS[i] for i in np.where(retr_acc == retr_acc.max())[0]]

    b = int((retr_vote & ~best_single_retr).sum())   # vote right, single wrong
    c = int((~retr_vote & best_single_retr).sum())   # vote wrong, single right
    p_mcnemar = mcnemar_exact_two_sided(b, c)

    # bootstrap CI on (vote - single) accuracy difference, in points
    rng = np.random.default_rng(SEED)
    diff = retr_vote.astype(int) - best_single_retr.astype(int)  # per question
    B = 10000
    idx = rng.integers(0, 82, size=(B, 82))
    boot = diff[idx].mean(1) * 100
    ci = (round(float(np.percentile(boot, 2.5)), 1),
          round(float(np.percentile(boot, 97.5)), 1))
    point_diff = round(float(diff.mean() * 100), 1)

    # ---- (2) Qwen3-235B members' retrieved accuracies ---------------------
    qwen3 = {e: round(float(retr_acc[i]), 1)
             for i, e in enumerate(DU3_EXPERTS) if e.startswith("qwen3-235b")}

    out = {
        "seed": SEED,
        "gate": {"oracle_vote": int(oracle_vote.sum()),
                 "retrieved_vote": int(retr_vote.sum())},
        "flag1_vote_vs_best_retrieved_single": {
            "best_single_expert": DU3_EXPERTS[best_i],
            "best_single_retrieved": f"{n_best}/82 = {retr_acc[best_i]:.1f}%",
            "tie_for_best": ties,
            "vote_retrieved": f"{int(retr_vote.sum())}/82 = {retr_vote.mean()*100:.1f}%",
            "discordant_vote_right_single_wrong_b": b,
            "discordant_vote_wrong_single_right_c": c,
            "mcnemar_exact_two_sided_p": round(p_mcnemar, 4),
            "bootstrap_diff_vote_minus_single_pp": point_diff,
            "bootstrap_95ci_pp": list(ci),
        },
        "flag2_qwen3_235b_retrieved_scores": {
            "per_expert_pc": qwen3,
            "range_pc": [min(qwen3.values()), max(qwen3.values())],
        },
        "cross_check_within_nine_val_selected": {
            "expert": "qwen3-235b-a22b_sc3_v1",
            "oracle_pc": round(float(oracle_acc[DU3_EXPERTS.index("qwen3-235b-a22b_sc3_v1")]), 1),
            "retrieved_pc": round(float(retr_acc[DU3_EXPERTS.index("qwen3-235b-a22b_sc3_v1")]), 1),
        },
    }
    print(json.dumps(out, indent=2))
    with open(HERE / "section7_verify_numbers.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
