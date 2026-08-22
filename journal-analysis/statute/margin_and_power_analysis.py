"""Vote margin as a confidence signal, and the benchmark size selection needs.

1. Selective prediction. The nine-expert committee's vote margin
   (|votes_Y - votes_N| in {1,3,5,7,9}) costs nothing and is available per
   question. Accuracy by margin bucket and the coverage-accuracy trade-off,
   on the 258-question validation benchmark and the 82-question R07 test.
   A single model deployed alone has no such signal.

2. Benchmark-size projection. Model-based simulation: fix each expert's true
   ability at its pooled 340-question estimate, draw independent binomial
   validation sets of size n, select the argmax, and record the TRUE regret
   (best true ability minus selected true ability). Reports expected regret
   and P(selected within 1pp of best) as n grows. Clearly a projection under
   an independence model, not an empirical claim.

Gate: reproduces pool and DU3 before computing. Writes
margin_power_numbers.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from selection_policy_analysis import (load_gold, build_pool,
                                       committee_correct, DU3_EXPERTS)

HERE = Path(__file__).parent
OUT = HERE / "margin_power_numbers.json"
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

    # ---- 1. vote margin as confidence --------------------------------------
    for split, c_mat, n in (("val", c_val, 258), ("test_R07", c_test, 82)):
        correct_counts = c_mat[du3].sum(0)          # experts correct per q
        vote_correct = correct_counts > 4.5
        # margin = |#majority - #minority| among the nine votes;
        # since correctness is vs gold: margin = |2*#correct - 9| regardless
        # of whether the majority is right or wrong
        margin = np.abs(2 * correct_counts - 9)
        buckets = {}
        for m in (1, 3, 5, 7, 9):
            mask = margin == m
            if mask.sum():
                buckets[int(m)] = {
                    "n": int(mask.sum()),
                    "share_pc": round(float(mask.mean() * 100), 1),
                    "vote_accuracy_pc": round(
                        float(vote_correct[mask].mean() * 100), 1),
                }
        coverage = {}
        for tau in (3, 5, 7, 9):
            mask = margin >= tau
            coverage[f"margin>={tau}"] = {
                "coverage_pc": round(float(mask.mean() * 100), 1),
                "selective_accuracy_pc": round(
                    float(vote_correct[mask].mean() * 100), 1),
                "flagged_pc": round(float((~mask).mean() * 100), 1),
                "accuracy_on_flagged_pc": round(
                    float(vote_correct[~mask].mean() * 100), 1)
                if (~mask).sum() else None,
            }
        results[f"margin_{split}"] = {"buckets": buckets,
                                      "coverage": coverage}

    # ---- 2. benchmark size needed for reliable selection -------------------
    pooled = np.concatenate([c_val, c_test], axis=1)  # (30, 340)
    p_true = pooled.mean(1)
    best = p_true.max()
    grid = [258, 500, 1_000, 2_500, 5_000, 10_000]
    B = 4_000
    proj = {}
    for n in grid:
        wins = rng.binomial(n, p_true[None, :].repeat(B, axis=0)) / n
        sel = wins.argmax(1)
        regret = best - p_true[sel]
        proj[n] = {
            "expected_true_regret_pp": round(float(regret.mean() * 100), 2),
            "p_within_1pp_of_best": round(
                float((regret <= 0.01).mean()), 3),
        }
    # winner's curse on the test side: expected OBSERVED maximum among the
    # 30 experts on an 82-question test, under the same ability model
    obs = rng.binomial(82, p_true[None, :].repeat(20_000, axis=0)) / 82
    obs_max = obs.max(1)
    results["benchmark_size_projection"] = {
        "model": "true ability = pooled 340-q estimate per expert; "
                 "independent Bernoulli questions; select argmax on a "
                 "simulated validation set of size n",
        "true_best": {"expert": pool[int(p_true.argmax())],
                      "ability_pc": round(float(best * 100), 1)},
        "true_top5_spread_pp": round(float(
            (best - np.sort(p_true)[-5]) * 100), 1),
        "true_full_spread_pp": round(float(
            (best - p_true.min()) * 100), 1),
        "by_n": proj,
        "winners_curse_at_n82": {
            "expected_observed_max_pc": round(
                float(obs_max.mean() * 100), 1),
            "p_observed_max_ge_95.1": round(
                float((obs_max >= 78 / 82).mean()), 3),
            "note": "no true ability exceeds 86.8, yet the best observed "
                    "score on 82 questions is expected near the observed "
                    "oracle 95.1",
        },
    }

    print(json.dumps(results, indent=2))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
