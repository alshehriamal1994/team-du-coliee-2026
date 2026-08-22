"""Null B with the exact perfect-ranker envelope.

expH2 prices the mechanical component of the leaderboard association against a
null that places each run's yield on the envelope min(k, g), with g the gold mean.
The exact perfect-ranker yield at a constant k is E_q[min(k, n_q)] over the gold
count distribution, which is smaller than min(k, E[n_q]) by Jensen, most sharply
around k = g. The substitution therefore changes how demanding the null is, and
the article states the direction, so it needs a ledger.

Gated on reproducing expH2's observed coefficients and its min(k, g) nulls before
the exact form is computed.

Writes expH3_numbers.json.
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RUNS = HERE / "expH_runs.csv"
PRIOR = HERE / "expH2_null_numbers.json"
GOLD_T1 = HERE / "expA_numbers.json"
TREND = HERE / "expU_numbers.json"
OUT = HERE / "expH3_numbers.json"

B_PERM = 10_000
SEED = 20260810


def load_runs():
    rows = []
    with open(RUNS) as f:
        for r in csv.DictReader(f):
            rows.append({
                "task": r["task"], "F1": float(r["F1"]),
                "k": float(r["avg_set_size"]),
                "closeness": -float(r["log_dist_from_gold"]),
            })
    return rows


def gold_counts():
    """Per-query gold count distributions for both tasks, as Counters.

    Task 1 comes from expA's histogram directly. Task 2's is recovered from
    expU's perfect-ranker ceilings, which are a strictly increasing function of
    the cumulative count distribution, and the recovery is checked against the
    ledgered gold total and mean.
    """
    t1 = Counter({int(k): int(v) for k, v in
                  json.load(open(GOLD_T1))["gold_count_distribution"].items()})

    u = json.load(open(TREND))["years"]["2026"]
    ceil = {int(k): float(v) for k, v in u["ceiling_by_k"].items()}
    n_cases, gold_total = int(u["n_cases"]), int(u["gold_total"])
    # ceiling(k) = 2 * S(k) / (n*k + G) with S(k) = sum_q min(k, n_q)
    S = {k: c * (n_cases * k + gold_total) / 2.0 for k, c in ceil.items()}
    ks = sorted(S)
    t2 = Counter()
    prev, prev_k = 0.0, 0
    tail = n_cases
    for k in ks:
        gained = S[k] - prev            # queries with n_q >= k, times (k - prev_k)
        at_or_above = gained / (k - prev_k)
        t2[prev_k] = 0
        t2[k] = 0
        if prev_k > 0:
            t2[prev_k] = int(round(tail - at_or_above))
        tail = at_or_above
        prev, prev_k = S[k], k
    t2[ks[-1]] = int(round(tail))
    t2 = Counter({k: v for k, v in t2.items() if k > 0 and v > 0})

    if sum(t2.values()) != n_cases or sum(k * v for k, v in t2.items()) != gold_total:
        sys.exit(f"GATE FAILED: recovered Task 2 distribution has "
                 f"{sum(t2.values())} cases and {sum(k*v for k,v in t2.items())} "
                 f"gold, expected {n_cases} and {gold_total}")
    return {"T1": t1, "T2": t2}


def exact_envelope(k, counts):
    """E_q[min(k, n_q)] for each answer-set size in k."""
    ns = np.array(sorted(counts), dtype=float)
    w = np.array([counts[int(n)] for n in ns], dtype=float)
    w = w / w.sum()
    return np.array([float((np.minimum(kk, ns) * w).sum()) for kk in k])


def run_null(f1, k, closeness, envelope, g, rng):
    t = f1 * (k + g) / 2.0
    q = t / envelope
    rho = stats.spearmanr(f1, closeness).statistic
    null = np.empty(B_PERM)
    for b in range(B_PERM):
        null[b] = stats.spearmanr(
            2.0 * rng.permutation(q) * envelope / (k + g), closeness).statistic
    return rho, {
        "mean": round(float(null.mean()), 3),
        "sd": round(float(null.std()), 3),
        "observed_percentile": round(float((null < rho).mean() * 100.0), 1),
    }


def main():
    prior = json.load(open(PRIOR))
    rows = load_runs()
    counts = gold_counts()
    rng = np.random.default_rng(SEED)
    out = {"question": "does the exact perfect-ranker envelope change the null",
           "permutations": B_PERM, "seed": SEED, "gates": {}, "tasks": {}}

    for task in ("T1", "T2"):
        sub = [r for r in rows if r["task"] == task]
        g = prior["tasks"][task]["gold_mean"]
        f1 = np.array([r["F1"] for r in sub])
        k = np.array([r["k"] for r in sub])
        cl = np.array([r["closeness"] for r in sub])

        rho_expected = prior["tasks"][task]["observed_spearman_F1_vs_closeness"]
        rho_here = round(float(stats.spearmanr(f1, cl).statistic), 3)
        if rho_here != rho_expected:
            sys.exit(f"GATE FAILED {task}: rho {rho_here} != {rho_expected}")

        _, approx = run_null(f1, k, cl, np.minimum(k, g), g,
                             np.random.default_rng(SEED))
        expected_mean = prior["tasks"][task]["null_B_perfect_ranker_envelope"]["mean"]
        if abs(approx["mean"] - expected_mean) > 0.02:
            sys.exit(f"GATE FAILED {task}: min(k,g) null mean {approx['mean']} "
                     f"!= ledgered {expected_mean}")
        out["gates"][task] = {"rho_reproduced": rho_here,
                              "min_k_g_null_mean_reproduced": approx["mean"]}

        rho, exact = run_null(f1, k, cl, exact_envelope(k, counts[task]), g, rng)
        out["tasks"][task] = {
            "observed_spearman": rho_here,
            "null_min_k_g": approx,
            "null_exact_envelope": exact,
        }
        print(f"[{task}] rho={rho_here}  min(k,g): mean {approx['mean']} "
              f"pct {approx['observed_percentile']}  |  exact: mean "
              f"{exact['mean']} pct {exact['observed_percentile']}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
