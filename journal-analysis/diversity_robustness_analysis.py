"""Robustness and diversity analyses for Manuscript 1 (reviewer-response pass).

Computes, from the same prediction logs as selection_policy_analysis.py:
  1. Fisher confidence interval for the val-test Pearson r over the 30 experts,
     and the binomial standard error of an expert's test accuracy at n=82.
  2. Tie-break sensitivity of the single-expert policy bootstrap (alphabetical
     vs random tie-breaking).
  3. Section 5.4 (diversity vs size) redone with uncertainty: reproduces the
     best low-diversity vs best high-diversity subset gaps at k=3,5,7 among
     the nine deployed experts (verification gate against the earlier numbers),
     then bootstraps the gap over the 258 validation questions and evaluates
     the same fixed subsets on the R07 test set.
  4. A selection-free, fixed-size diversity analysis over the full 30-expert
     pool: for random nine-member committees, the committee's lift over its
     mean member accuracy is regressed against its diversity (distinct
     families, cross-family pair fraction, mean pairwise disagreement).
     No subset selection is involved, so the analysis is immune to the
     validation-cannot-rank objection.

Writes diversity_robustness_numbers.json. Aborts if the verification gates
fail.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from selection_policy_analysis import (load_gold, build_pool,
                                       committee_correct, DU3_EXPERTS,
                                       bootstrap_argmax_test_acc)

HERE = Path(__file__).parent
OUT = HERE / "diversity_robustness_numbers.json"
SEED = 20260704
B = 10_000

FAMILY_PREFIX = [
    ("deepseek", "DeepSeek"), ("llama", "Llama"), ("qwen", "Qwen"),
    ("qwq", "Qwen"), ("mistral", "Mistral"), ("gemma", "Gemma"),
]


def family(name):
    for pref, fam in FAMILY_PREFIX:
        if name.startswith(pref):
            return fam
    raise ValueError(name)


def fisher_ci(r, n):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def vote_acc(c, idx):
    return (c[list(idx)].sum(axis=0) > len(idx) / 2).mean()


def main():
    rng = np.random.default_rng(SEED)
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    val_acc = c_val.mean(1)
    test_acc = c_test.mean(1)
    n_val = len(val_ids)

    results = {}

    # ---- 1. correlation CI and test-side noise -----------------------------
    r = float(np.corrcoef(val_acc, test_acc)[0, 1])
    lo, hi = fisher_ci(r, 30)
    p_mean = float(test_acc.mean())
    se_binom = float(np.sqrt(p_mean * (1 - p_mean) / 82))
    results["corr_ci"] = {
        "pearson_r": round(r, 2),
        "fisher_95ci": [round(float(lo), 2), round(float(hi), 2)],
        "binomial_se_test_acc_pp": round(se_binom * 100, 1),
    }

    # ---- 2. tie-break sensitivity of the policy bootstrap ------------------
    weights = rng.multinomial(n_val, np.full(n_val, 1 / n_val), size=B).T
    dep_alpha, _ = bootstrap_argmax_test_acc(c_val, test_acc, weights)
    # random tie-breaking: add a tiny random jitter per (expert, replicate)
    cf = c_val.astype(np.float32)
    dep_rand = np.empty(B)
    chunk = 1_000
    for s in range(0, B, chunk):
        w = weights[:, s:s + chunk].astype(np.float32)
        scores = cf @ w + rng.random((30, w.shape[1])).astype(np.float32) * 0.5
        dep_rand[s:s + chunk] = test_acc[scores.argmax(axis=0)]
    results["tiebreak_sensitivity"] = {
        "alphabetical": {"mean_pc": round(float(dep_alpha.mean() * 100), 1),
                         "p5_pc": round(float(np.percentile(dep_alpha, 5) * 100), 1),
                         "p95_pc": round(float(np.percentile(dep_alpha, 95) * 100), 1)},
        "random": {"mean_pc": round(float(dep_rand.mean() * 100), 1),
                   "p5_pc": round(float(np.percentile(dep_rand, 5) * 100), 1),
                   "p95_pc": round(float(np.percentile(dep_rand, 95) * 100), 1)},
    }

    # ---- 3. diversity vs size among the nine deployed experts --------------
    # Diversity at MODEL granularity (distinct base models). Reproduces the
    # earlier draft numbers at k=3 and k=5 exactly; the earlier k=7 low value
    # (89.9) is unreproducible under any tested definition and is corrected
    # here to 90.7 (best 4-model subset). Family-level strata are also
    # reported: at family granularity the frontier gap vanishes.
    du3_idx = [pool.index(e) for e in DU3_EXPERTS]
    base_model = {
        "deepseek-r1-zs": "DSR1",
        "llama-4-maverick_standard_v1": "L4-Mav",
        "llama-4-scout_standard_v1": "L4-Scout",
        "llama-3.3-70b-instruct_concise_v1": "L3.3",
        "llama-3.3-70b-instruct_cot_strict_v1": "L3.3",
        "qwen-2.5-72b-instruct_cot_strict_v1": "Q2.5",
        "qwen3-235b-a22b_standard_v1": "Q3-235",
        "qwen3-235b-a22b_irac_v1": "Q3-235",
        "qwen3-235b-a22b_sc3_v1": "Q3-235",
    }
    mods9 = [base_model[pool[i]] for i in du3_idx]
    fams9 = [family(pool[i]) for i in du3_idx]

    def frontier(k, labels):
        stats = []
        for sub in combinations(range(9), k):
            nd = len({labels[j] for j in sub})
            members = tuple(du3_idx[j] for j in sub)
            stats.append((nd, members, vote_acc(c_val, members)))
        lo = min(s[0] for s in stats)
        hi = max(s[0] for s in stats)
        best_low = max((s for s in stats if s[0] == lo), key=lambda s: s[2])
        best_high = max((s for s in stats if s[0] == hi), key=lambda s: s[2])
        low_c = c_val[list(best_low[1])].sum(0) > k / 2
        high_c = c_val[list(best_high[1])].sum(0) > k / 2
        idx = rng.integers(0, n_val, size=(B, n_val))
        gaps = high_c[idx].mean(1) - low_c[idx].mean(1)
        return {
            "n_low": best_low[0], "n_high": best_high[0],
            "val_low_pc": round(float(best_low[2] * 100), 1),
            "val_high_pc": round(float(best_high[2] * 100), 1),
            "val_gap_pp": round(float((best_high[2] - best_low[2]) * 100), 1),
            "val_gap_95ci_pp": [round(float(np.percentile(gaps, 2.5) * 100), 1),
                                round(float(np.percentile(gaps, 97.5) * 100), 1)],
            "test_low_pc": round(float(vote_acc(c_test, best_low[1]) * 100), 1),
            "test_high_pc": round(float(vote_acc(c_test, best_high[1]) * 100), 1),
        }

    results["diversity_size_9experts_by_model"] = {
        k: frontier(k, mods9) for k in (3, 5, 7)}
    results["diversity_size_9experts_by_family"] = {
        k: frontier(k, fams9) for k in (3, 5, 7)}

    pm = results["diversity_size_9experts_by_model"]
    gate = (pm[3]["val_low_pc"], pm[3]["val_high_pc"],
            pm[5]["val_low_pc"], pm[5]["val_high_pc"],
            pm[7]["val_high_pc"])
    expected = (85.7, 89.5, 88.8, 90.3, 91.1)
    if gate != expected:
        print(f"GATE FAILED: got {gate}, expected {expected}")
        sys.exit(1)
    print("gate diversity-size (model-level, 5 reproducible numbers): PASS")

    # ---- 4. selection-free lift vs diversity over the 30-pool --------------
    fams30 = np.array([family(p) for p in pool])
    M = 5_000
    seen, committees = set(), []
    while len(committees) < M:
        c = tuple(sorted(rng.choice(30, size=9, replace=False)))
        if c not in seen:
            seen.add(c)
            committees.append(c)
    committees = np.array(committees)

    lift = {}
    for split, c_mat in (("val", c_val), ("test", c_test)):
        comm_acc = committee_correct(c_mat, committees).mean(1)
        member_acc = c_mat.mean(1)[committees].mean(1)
        lift[split] = comm_acc - member_acc

    n_fam = np.array([len(set(fams30[c])) for c in committees])
    cross_frac = np.array([
        np.mean([fams30[a] != fams30[b] for a, b in combinations(c, 2)])
        for c in committees])
    # mean pairwise disagreement on validation predictions
    disagree = np.zeros(M)
    for i, c in enumerate(committees):
        rows = c_val[list(c)]
        d = 0.0
        for a, b in combinations(range(9), 2):
            d += (rows[a] != rows[b]).mean()
        disagree[i] = d / 36

    def corr(x, y):
        return round(float(np.corrcoef(x, y)[0, 1]), 2)

    results["pool_lift_vs_diversity"] = {
        "n_committees": M,
        "mean_lift_val_pp": round(float(lift["val"].mean() * 100), 1),
        "mean_lift_test_pp": round(float(lift["test"].mean() * 100), 1),
        "corr_lift_val__n_families": corr(lift["val"], n_fam),
        "corr_lift_val__cross_frac": corr(lift["val"], cross_frac),
        "corr_lift_val__disagreement": corr(lift["val"], disagree),
        "corr_lift_test__n_families": corr(lift["test"], n_fam),
        "corr_lift_test__cross_frac": corr(lift["test"], cross_frac),
        "corr_lift_test__disagreement": corr(lift["test"], disagree),
        "corr_commacc_test__cross_frac": corr(
            committee_correct(c_test, committees).mean(1), cross_frac),
        "lift_val_by_nfam": {
            int(f): [round(float(lift["val"][n_fam == f].mean() * 100), 1),
                     int((n_fam == f).sum())]
            for f in np.unique(n_fam)},
        "lift_test_by_nfam": {
            int(f): [round(float(lift["test"][n_fam == f].mean() * 100), 1),
                     int((n_fam == f).sum())]
            for f in np.unique(n_fam)},
    }

    print(json.dumps(results, indent=2))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
