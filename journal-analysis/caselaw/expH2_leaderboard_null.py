"""Permutation null for the leaderboard association (journal article, Section 6).

Micro-averaged F1 rewards returning roughly as many answers as the gold set contains
whatever the ranking is worth, so part of any association between answer-set calibration
and final score is a property of the metric. Under micro-averaging a run's mean true
positives per query is recoverable from the leaderboard as t = F1 (k + g) / 2, with k its
mean answers returned and g the gold mean, so true positives can be separated from
answer-set size and reassigned across runs. Two nulls bracket the assumption of whether
returning more finds more: one gives a run no extra gold for returning more, the other
places true positives on the perfect-ranker envelope min(k, g) and permutes only the
quality fraction. A best-run-per-team variant addresses the non-independence of multiple
runs per team.

The reconstruction is gated by reproducing every run's published precision and recall.
Writes expH2_null_numbers.json.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RUNS = HERE / "expH_runs.csv"
PUBLISHED = HERE / "expH_numbers.json"
OUT = HERE / "expH2_null_numbers.json"

B_PERM = 10_000
SEED = 20260810


def load():
    rows = []
    with open(RUNS) as f:
        for r in csv.DictReader(f):
            rows.append({
                "task": r["task"], "team": r["team"], "run": r["run"],
                "P": float(r["P"]), "R": float(r["R"]), "F1": float(r["F1"]),
                "k": float(r["avg_set_size"]),
                # ledger correlates F1 against CLOSENESS, i.e. minus the distance
                "closeness": -float(r["log_dist_from_gold"]),
            })
    return rows


def main():
    pub = json.load(open(PUBLISHED))
    rows = load()
    rng = np.random.default_rng(SEED)
    out = {
        "protocol": {
            "question": "how much of the leaderboard size-quality association is a "
                        "property of the micro-F1 surface rather than of the field",
            "reconstruction": "t = F1 (k + g) / 2, gated against published P and R",
            "null": "permute ranking yield t across runs, hold answer-set size k "
                    "fixed, recompute F1 = 2t/(k+g), re-measure Spearman",
            "permutations": B_PERM, "seed": SEED,
        },
        "tasks": {},
    }

    for task, key in (("T1", "task1"), ("T2", "task2")):
        sub = [r for r in rows if r["task"] == task]
        g = pub[key]["gold_mean"]
        k = np.array([r["k"] for r in sub])
        f1 = np.array([r["F1"] for r in sub])
        closeness = np.array([r["closeness"] for r in sub])
        t = f1 * (k + g) / 2.0

        # ---- gate 1: reconstruction reproduces published P and R ------------
        P_hat, R_hat = t / k, t / g
        P_obs = np.array([r["P"] for r in sub])
        R_obs = np.array([r["R"] for r in sub])
        dP, dR = np.abs(P_hat - P_obs).max(), np.abs(R_hat - R_obs).max()
        ok_recon = dP < 5e-3 and dR < 5e-3
        print(f"  [{task}] gate reconstruction: max |dP|={dP:.5f} "
              f"max |dR|={dR:.5f}  {'PASS' if ok_recon else 'FAIL'}")

        # ---- gate 2: reproduce the published Spearman values ----------------
        rho_obs = stats.spearmanr(f1, closeness).statistic
        rho_size = stats.spearmanr(f1, k).statistic
        ok_rho = (round(float(rho_obs), 3) ==
                  round(pub[key]["spearman_F1_vs_logdist_from_gold"], 3))
        ok_size = (round(float(rho_size), 3) ==
                   round(pub[key]["spearman_F1_vs_size"], 3))
        n_ok = len(sub) == pub[key]["n_runs_scored"]
        print(f"  [{task}] gate ledger: n={len(sub)} rho_dist={rho_obs:.3f} "
              f"rho_size={rho_size:.3f}  "
              f"{'PASS' if (ok_rho and ok_size and n_ok) else 'FAIL'}")
        if not (ok_recon and ok_rho and ok_size and n_ok):
            print("GATE FAILED - no new numbers computed.")
            sys.exit(1)

        # ---- null A: yield independent of size -------------------------------
        nullA = np.empty(B_PERM)
        for b in range(B_PERM):
            nullA[b] = stats.spearmanr(
                2.0 * rng.permutation(t) / (k + g), closeness).statistic

        # ---- null B: yield on the perfect-ranker envelope --------------------
        envelope = np.minimum(k, g)
        q = t / envelope                       # observed quality fraction
        nullB = np.empty(B_PERM)
        for b in range(B_PERM):
            nullB[b] = stats.spearmanr(
                2.0 * rng.permutation(q) * envelope / (k + g), closeness).statistic

        def summarise(nl):
            lo, hi = np.percentile(nl, [2.5, 97.5])
            return {
                "mean": round(float(nl.mean()), 3),
                "sd": round(float(nl.std()), 3),
                "ci95": [round(float(lo), 3), round(float(hi), 3)],
                "observed_percentile": round(float((nl < rho_obs).mean() * 100.0), 1),
                "p_observed_exceeds_null": round(float((nl >= rho_obs).mean()), 4),
            }
        print(f"  [{task}] quality fraction q: min {q.min():.3f} "
              f"median {np.median(q):.3f} max {q.max():.3f}")

        # ---- best run per team ----------------------------------------------
        best = {}
        for r in sub:
            if r["team"] not in best or r["F1"] > best[r["team"]]["F1"]:
                best[r["team"]] = r
        bl = list(best.values())
        bf = np.array([r["F1"] for r in bl])
        bd = np.array([r["closeness"] for r in bl])
        rb = stats.spearmanr(bf, bd)

        out["tasks"][task] = {
            "n_runs": len(sub), "gold_mean": g,
            "observed_spearman_F1_vs_closeness": round(float(rho_obs), 3),
            "observed_spearman_F1_vs_size": round(float(rho_size), 3),
            "null_A_yield_independent_of_size": summarise(nullA),
            "null_B_perfect_ranker_envelope": summarise(nullB),
            "best_run_per_team": {
                "n_teams": len(bl),
                "spearman": round(float(rb.statistic), 3),
                "p_value": round(float(rb.pvalue), 4),
            },
        }
        for lbl, nl in (("A", nullA), ("B", nullB)):
            s_ = summarise(nl)
            print(f"  [{task}] observed {rho_obs:.3f} vs null {lbl}: "
                  f"mean {s_['mean']:+.3f} 95% [{s_['ci95'][0]:+.3f},"
                  f"{s_['ci95'][1]:+.3f}] -> observed at "
                  f"{s_['observed_percentile']:.1f}th pct, "
                  f"p={s_['p_observed_exceeds_null']:.4f}")
        print(f"  [{task}] best run per team: n={len(bl)} "
              f"rho={rb.statistic:.3f} p={rb.pvalue:.4f}\n")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
