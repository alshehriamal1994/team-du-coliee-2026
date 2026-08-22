"""Ensemble lift against diversity over random ensembles (journal article, Section 6).

For 50,000 random nine-expert ensembles drawn uniformly from the pool, the lift is the
ensemble's majority-vote accuracy minus the mean accuracy of its own members, computed
per split. Three diversity measures are defined explicitly: the number of distinct base
models spanned, the number of distinct vendor families, and the fraction of member pairs
drawn from different families. No subset selection is involved, so the analysis is immune
to the objection that validation cannot rank near-tied candidates. The script reproduces
the module's ledger gate before computing and reports the Monte Carlo standard error.

Writes lift_diversity_numbers.json.
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import selection_policy_analysis as base

OUT = Path(__file__).resolve().parent / "lift_diversity_numbers.json"

M_ENSEMBLES = 50_000
K = 9
SEED = 20260704

BASE_MODELS = [
    "deepseek-r1", "deepseek-v3.1", "llama-3.3-70b-instruct", "llama-4-maverick",
    "llama-4-scout", "qwen-2.5-72b-instruct", "qwen3-235b-a22b", "qwen3-32b",
    "qwq-32b", "mistral-large-2411", "gemma-3-27b-it",
]
FAMILY = {
    "deepseek-r1": "DeepSeek", "deepseek-v3.1": "DeepSeek",
    "llama-3.3-70b-instruct": "Llama", "llama-4-maverick": "Llama",
    "llama-4-scout": "Llama", "qwen-2.5-72b-instruct": "Qwen",
    "qwen3-235b-a22b": "Qwen", "qwen3-32b": "Qwen", "qwq-32b": "Qwen",
    "mistral-large-2411": "Mistral", "gemma-3-27b-it": "Gemma",
}


def base_model_of(name):
    if name == "deepseek-r1-zs":
        return "deepseek-r1"
    stem = re.sub(r"_v[123]$", "", name)
    for key in sorted(BASE_MODELS, key=len, reverse=True):
        if stem.startswith(key + "_"):
            return key
    raise ValueError(f"unparsed expert id: {name}")


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def partial(x, y, z):
    """Correlation of x with y, controlling for z."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def run_gate(pool, c_val, c_test):
    val_acc, test_acc = c_val.mean(axis=1), c_test.mean(axis=1)
    vb = int(val_acc.argmax())
    du3 = np.array([[pool.index(e) for e in base.DU3_EXPERTS]])
    checks = {
        "pool_is_30": len(pool) == 30,
        "corr_0.43": round(pearson(val_acc, test_acc), 2) == 0.43,
        "val_best_is_llama33_standard":
            pool[vb] == "llama-3.3-70b-instruct_standard_v1",
        "val_best_test_68_of_82": int(c_test[vb].sum()) == 68,
        "du3_val_237": int(base.committee_correct(c_val, du3)[0].sum()) == 237,
        "du3_test_77": int(base.committee_correct(c_test, du3)[0].sum()) == 77,
    }
    for k, ok in checks.items():
        print(f"  gate {k}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


def main():
    gold = base.load_gold()
    pool, c_val, c_test, _, _ = base.build_pool(gold)
    n = len(pool)
    if not run_gate(pool, c_val, c_test):
        print("LEDGER GATE FAILED - no numbers computed.")
        sys.exit(1)

    bm = np.array([base_model_of(p) for p in pool])
    fam = np.array([FAMILY[b] for b in bm])
    print(f"  pool: {n} experts, {len(set(bm))} base models, {len(set(fam))} families")

    rng = np.random.default_rng(SEED)
    idx = np.array([rng.choice(n, K, replace=False) for _ in range(M_ENSEMBLES)])

    acc_val, acc_test = c_val.mean(axis=1) * 100, c_test.mean(axis=1) * 100
    vote_val = base.committee_correct(c_val, idx).mean(axis=1) * 100
    vote_test = base.committee_correct(c_test, idx).mean(axis=1) * 100
    mem_val, mem_test = acc_val[idx].mean(axis=1), acc_test[idx].mean(axis=1)
    lift_val, lift_test = vote_val - mem_val, vote_test - mem_test

    n_models = np.array([len(set(bm[r])) for r in idx])
    n_fams = np.array([len(set(fam[r])) for r in idx])
    pairs = list(combinations(range(K), 2))
    cross_frac = np.array([
        np.mean([fam[r[i]] != fam[r[j]] for i, j in pairs]) for r in idx])

    # mean pairwise disagreement among members, measured on validation
    dis = np.zeros(M_ENSEMBLES)
    cvb = c_val.astype(np.int8)
    for t, r in enumerate(idx):
        sub = cvb[r]
        d = 0.0
        for i, j in pairs:
            d += float((sub[i] != sub[j]).mean())
        dis[t] = d / len(pairs)

    res = {
        "protocol": {
            "n_ensembles": M_ENSEMBLES, "ensemble_size": K, "seed": SEED,
            "selection": "uniform random from the 30-expert pool; no subset "
                         "selection of any kind",
            "lift": "ensemble majority-vote accuracy minus the mean accuracy "
                    "of its own members, per split",
            "diversity_measures": {
                "n_models": "distinct base models spanned (prompt variants collapsed)",
                "n_families": "distinct vendor families spanned",
                "cross_frac": "fraction of member pairs from different families",
            },
            "monte_carlo_se_on_r": round(float(1 / np.sqrt(M_ENSEMBLES)), 4),
        },
        "mean_lift_val_pp": round(float(lift_val.mean()), 2),
        "mean_lift_test_pp": round(float(lift_test.mean()), 2),
        "correlations_test_lift": {
            "n_models": round(pearson(lift_test, n_models), 3),
            "n_families": round(pearson(lift_test, n_fams), 3),
            "cross_frac": round(pearson(lift_test, cross_frac), 3),
            "mean_member_accuracy": round(pearson(lift_test, mem_test), 3),
        },
        "correlations_val_lift": {
            "n_models": round(pearson(lift_val, n_models), 3),
            "n_families": round(pearson(lift_val, n_fams), 3),
            "member_disagreement": round(pearson(lift_val, dis), 3),
        },
        "correlation_test_lift_val_disagreement":
            round(pearson(lift_test, dis), 3),
        "partial_correlations_test_lift": {
            "n_families_given_n_models":
                round(partial(lift_test, n_fams, n_models), 3),
            "cross_frac_given_n_models":
                round(partial(lift_test, cross_frac, n_models), 3),
            "n_models_given_member_accuracy":
                round(partial(lift_test, n_models, mem_test), 3),
        },
    }

    c, p = res["correlations_test_lift"], res["partial_correlations_test_lift"]
    print(f"  mean lift  val {res['mean_lift_val_pp']:+.2f}  "
          f"test {res['mean_lift_test_pp']:+.2f}")
    print(f"  test lift ~ n_models {c['n_models']:.3f}   "
          f"n_families {c['n_families']:.3f}   "
          f"cross_frac {c['cross_frac']:.3f}   "
          f"member_acc {c['mean_member_accuracy']:.3f}")
    print(f"  partial: families|models {p['n_families_given_n_models']:.3f}   "
          f"cross_frac|models {p['cross_frac_given_n_models']:.3f}   "
          f"models|member_acc {p['n_models_given_member_accuracy']:.3f}")
    print(f"  disagreement: val lift {res['correlations_val_lift']['member_disagreement']:.3f}   "
          f"test lift {res['correlation_test_lift_val_disagreement']:.3f}")
    print(f"  Monte Carlo SE on a correlation: "
          f"{res['protocol']['monte_carlo_se_on_r']:.4f}")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
