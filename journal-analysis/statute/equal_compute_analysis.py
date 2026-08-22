"""Equal-compute committees and selection stability across validation years.

Two analyses for the journal manuscript, both from the existing prediction
logs, both selection-free unless stated.

1. Equal-compute comparison. At a fixed inference budget of k calls, compare
   committees built from k prompt variants of ONE model (self-consistency
   style, no model diversity) against committees spanning several models and
   families. Same-model committees are enumerated exhaustively for every
   model with at least k prompt variants; multi-model comparators are random
   k-subsets of the 30-expert pool that span at least two base models.
   Also: all-Qwen nine-member committees (the largest single family, 15
   experts) against cross-family nine-member committees.

2. Selection stability across validation years. Re-run single-expert
   validation selection using each exam year alone, and each pair of years,
   and record which expert wins and its R07 test accuracy. Shows whether the
   deployed champion depends on an arbitrary choice of validation years.

Gate: reproduces the ledger pool and DU3 before computing. Writes
equal_compute_numbers.json.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from selection_policy_analysis import (load_gold, build_pool,
                                       committee_correct, DU3_EXPERTS)

HERE = Path(__file__).parent
OUT = HERE / "equal_compute_numbers.json"
SEED = 20260704
M = 2_000

BASE_MODEL_PREFIX = [
    ("deepseek-r1", "DeepSeek-R1"), ("deepseek-v3.1", "DeepSeek-V3.1"),
    ("gemma-3-27b", "Gemma-3-27B"), ("llama-3.3", "Llama-3.3-70B"),
    ("llama-4-maverick", "Llama-4-Maverick"),
    ("llama-4-scout", "Llama-4-Scout"), ("mistral-large", "Mistral-Large"),
    ("qwen-2.5-72b", "Qwen-2.5-72B"), ("qwen3-235b", "Qwen3-235B"),
    ("qwen3-32b", "Qwen3-32B"), ("qwq-32b", "QwQ-32B"),
]
FAMILY_PREFIX = [
    ("deepseek", "DeepSeek"), ("llama", "Llama"), ("qwen", "Qwen"),
    ("qwq", "Qwen"), ("mistral", "Mistral"), ("gemma", "Gemma"),
]


def base_model(name):
    for pref, m in BASE_MODEL_PREFIX:
        if name.startswith(pref):
            return m
    raise ValueError(name)


def family(name):
    for pref, f in FAMILY_PREFIX:
        if name.startswith(pref):
            return f
    raise ValueError(name)


def dist(accs):
    accs = np.asarray(accs)
    return {
        "n": int(len(accs)),
        "mean_pc": round(float(accs.mean() * 100), 1),
        "median_pc": round(float(np.median(accs) * 100), 1),
        "p5_pc": round(float(np.percentile(accs, 5) * 100), 1),
        "p95_pc": round(float(np.percentile(accs, 95) * 100), 1),
        "min_pc": round(float(accs.min() * 100), 1),
        "max_pc": round(float(accs.max() * 100), 1),
    }


def main():
    rng = np.random.default_rng(SEED)
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    val_acc = c_val.mean(1)
    test_acc = c_test.mean(1)

    # gate
    du3 = np.array([[pool.index(e) for e in DU3_EXPERTS]])
    ok = (len(pool) == 30
          and int(committee_correct(c_test, du3)[0].sum()) == 77
          and pool[int(val_acc.argmax())]
          == "llama-3.3-70b-instruct_standard_v1")
    if not ok:
        print("GATE FAILED")
        sys.exit(1)
    print("gate: PASS")

    models = np.array([base_model(p) for p in pool])
    fams = np.array([family(p) for p in pool])
    results = {"pool_models": {m: int((models == m).sum())
                               for m in sorted(set(models))}}

    # ---- 1a. same-model vs multi-model committees at equal budget ----------
    for k in (3, 5):
        same = []
        by_model = {}
        for m in sorted(set(models)):
            idx = np.where(models == m)[0]
            if len(idx) < k:
                continue
            combos = np.array(list(combinations(idx, k)))
            accs = committee_correct(c_test, combos).mean(1)
            same.extend(accs.tolist())
            by_model[m] = dist(accs)
        multi = []
        seen = set()
        while len(multi) < M:
            c = tuple(sorted(rng.choice(30, size=k, replace=False)))
            if c in seen or len({models[j] for j in c}) < 2:
                continue
            seen.add(c)
            multi.append(c)
        multi_acc = committee_correct(c_test, np.array(multi)).mean(1)
        results[f"equal_compute_k{k}"] = {
            "same_model_committees_test": dist(same),
            "same_model_by_model": by_model,
            "multi_model_committees_test": dist(multi_acc),
            "gap_of_means_pp": round(
                float((multi_acc.mean() - np.mean(same)) * 100), 1),
        }

    # ---- 1b. all-Qwen vs cross-family committees at k=9 --------------------
    qwen_idx = np.where(fams == "Qwen")[0]
    qwen_combos = np.array(list(combinations(qwen_idx, 9)))
    qwen_acc = committee_correct(c_test, qwen_combos).mean(1)
    cross = []
    seen = set()
    while len(cross) < M:
        c = tuple(sorted(rng.choice(30, size=9, replace=False)))
        if c in seen or len({fams[j] for j in c}) < 3:
            continue
        seen.add(c)
        cross.append(c)
    cross_acc = committee_correct(c_test, np.array(cross)).mean(1)
    results["k9_single_family_qwen_vs_cross_family"] = {
        "qwen_only_test": dist(qwen_acc),
        "cross_family_3plus_test": dist(cross_acc),
        "gap_of_means_pp": round(
            float((cross_acc.mean() - qwen_acc.mean()) * 100), 1),
        "note": "Qwen is the only family with >=9 configurations (15).",
    }

    # ---- 1c. which diversity unit predicts lift: models or families? -------
    seen, comm = set(), []
    while len(comm) < 5_000:
        c = tuple(sorted(rng.choice(30, size=9, replace=False)))
        if c not in seen:
            seen.add(c)
            comm.append(c)
    comm = np.array(comm)
    t_lift = committee_correct(c_test, comm).mean(1) - test_acc[comm].mean(1)
    v_lift = committee_correct(c_val, comm).mean(1) - val_acc[comm].mean(1)
    n_mod = np.array([len(set(models[c])) for c in comm])
    n_fam = np.array([len(set(fams[c])) for c in comm])
    X = np.column_stack([np.ones(len(comm)), n_mod])
    beta_f = np.linalg.lstsq(X, n_fam, rcond=None)[0]
    beta_l = np.linalg.lstsq(X, t_lift, rcond=None)[0]
    partial = np.corrcoef(n_fam - X @ beta_f, t_lift - X @ beta_l)[0, 1]
    results["lift_predictor_k9"] = {
        "corr_test_lift__n_models": round(float(
            np.corrcoef(n_mod, t_lift)[0, 1]), 2),
        "corr_test_lift__n_families": round(float(
            np.corrcoef(n_fam, t_lift)[0, 1]), 2),
        "corr_val_lift__n_models": round(float(
            np.corrcoef(n_mod, v_lift)[0, 1]), 2),
        "corr_val_lift__n_families": round(float(
            np.corrcoef(n_fam, v_lift)[0, 1]), 2),
        "partial_corr_families_given_models_test": round(float(partial), 2),
        "note": "family count adds nothing to test lift once base-model "
                "count is controlled",
    }

    # ---- 2. selection stability across validation years --------------------
    groups = {}
    for j, q in enumerate(val_ids):
        y = "R01" if q.split("-")[0] in ("R01", "R1") else q.split("-")[0]
        groups.setdefault(y, []).append(j)
    stability = {}
    year_sets = [("H30",), ("R01",), ("R02",), ("H30", "R01"),
                 ("H30", "R02"), ("R01", "R02"), ("H30", "R01", "R02")]
    for ys in year_sets:
        idx = sorted(j for y in ys for j in groups[y])
        accs = c_val[:, idx].mean(1)
        w = int(accs.argmax())
        stability["+".join(ys)] = {
            "selected": pool[w],
            "val_acc_pc": round(float(accs[w] * 100), 1),
            "test_acc_pc": round(float(test_acc[w] * 100), 1),
            "test_rank_of_30": int((test_acc > test_acc[w]).sum()) + 1,
        }
    results["selection_by_year"] = stability

    print(json.dumps(results, indent=2))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
