"""Direct measurement of error complementarity, without the keyword taxonomy.

The reasoning-trace taxonomy classifies errors by deterministic keyword matching, which
licenses only the claim that a feature was mentioned. This script measures the property
the taxonomy is used to support, namely whether the experts fail on different questions,
straight from the label predictions, so it involves no proxy at all.

Three quantities per split.

  1. Pairwise error overlap. For every pair of experts, the fraction of questions both
     answer incorrectly, against the product of their individual error rates, which is
     what statistical independence would give. The ratio is reported without adjustment.
     A ratio above one means errors are positively correlated, so the experts fail
     together more than chance, which is evidence AGAINST full complementarity and is
     reported as such.

  2. Pool coverage. The fraction of questions answered correctly by at least one expert,
     and the number defeating every expert. This is an oracle quantity. It bounds what
     any aggregation over this pool could reach and is not attained by majority voting;
     the achieved vote accuracy is reported beside it so the two are not confused.

  3. Majority vote accuracy over the whole pool, for that comparison.

Verification gate: the pool is rebuilt through selection_policy_analysis, and the expert
count, question counts and pool mean test accuracy must match selection_policy_numbers.json
before anything is written.

Writes error_overlap_numbers.json.
"""
import itertools
import json
from pathlib import Path

import numpy as np

import selection_policy_analysis as base

HERE = Path(__file__).parent
LEDGER = HERE / "selection_policy_numbers.json"
OUT = HERE / "error_overlap_numbers.json"


def split_stats(correct, label):
    """correct: (n_experts, n_questions) boolean, True where the expert is right."""
    err = ~correct
    n, q = err.shape
    rates = err.mean(1)

    obs, ind = [], []
    for i, j in itertools.combinations(range(n), 2):
        obs.append((err[i] & err[j]).mean())
        ind.append(rates[i] * rates[j])
    obs, ind = np.array(obs), np.array(ind)

    any_right = (~err).any(0)
    all_wrong = int(err.all(0).sum())
    vote = (correct.sum(0) > n / 2)

    return {
        "split": label,
        "n_experts": int(n),
        "n_questions": int(q),
        "mean_expert_error_rate_pc": round(float(rates.mean() * 100), 2),
        "mean_pairwise_both_wrong_pc": round(float(obs.mean() * 100), 2),
        "independence_expectation_pc": round(float(ind.mean() * 100), 2),
        "overlap_ratio_observed_over_independent": round(float(obs.mean() / ind.mean()), 3),
        "n_pairs": int(len(obs)),
        "pairs_more_correlated_than_independent": int((obs > ind).sum()),
        "coverage_at_least_one_correct_pc": round(float(any_right.mean() * 100), 2),
        "questions_all_experts_fail": all_wrong,
        "pool_majority_vote_acc_pc": round(float(vote.mean() * 100), 2),
        "coverage_minus_vote_pp": round(float((any_right.mean() - vote.mean()) * 100), 2),
    }


def main():
    gold = base.load_gold()
    names, val, test, vids, tids = base.build_pool(gold)

    ledger = json.loads(LEDGER.read_text())
    pool_mean_test = float(test.mean() * 100)
    gates = {
        "n_experts_is_30": len(names) == 30,
        "n_val_questions_is_258": val.shape[1] == 258,
        "n_test_questions_is_82": test.shape[1] == 82,
        "pool_mean_test_acc_matches_ledger": abs(pool_mean_test - ledger["pool_mean_test_acc_pc"]) < 0.06,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates} "
                         f"(pool mean test {pool_mean_test:.2f} vs "
                         f"ledger {ledger['pool_mean_test_acc_pc']})")

    res = {
        "purpose": "complementarity measured from predictions alone, no keyword proxy",
        "gates": gates,
        "n_distinct_base_models": len({e["model"] for e in json.loads((HERE / "appendix_pool_numbers.json").read_text())["experts"]}),
        "validation": split_stats(val, "validation (H30, R01, R02)"),
        "test": split_stats(test, "official test (R07)"),
        "reading": {
            "errors_are_not_independent": (
                "Both splits show pairwise error overlap well above the independence "
                "expectation, so the experts fail together far more than chance. The "
                "complementarity that voting exploits is partial, not a profile of "
                "unrelated failures."),
            "coverage_is_an_oracle_bound": (
                "Coverage counts questions some expert answers correctly. It is an upper "
                "bound on any aggregation over this pool and is not reached by majority "
                "voting; the gap between the two is reported as coverage_minus_vote_pp."),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    for s in (res["validation"], res["test"]):
        print(f"  {s['split']:28s} overlap {s['mean_pairwise_both_wrong_pc']:5.2f}% vs "
              f"independent {s['independence_expectation_pc']:5.2f}%  "
              f"ratio {s['overlap_ratio_observed_over_independent']:.2f}x | "
              f"coverage {s['coverage_at_least_one_correct_pc']:.1f}% "
              f"(all fail: {s['questions_all_experts_fail']}) | "
              f"vote {s['pool_majority_vote_acc_pc']:.1f}%")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
