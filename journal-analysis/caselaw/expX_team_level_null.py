"""Permutation nulls for the best-run-per-team leaderboard check (journal article, Section 6).

The full-field association between answer-set calibration and final score is priced
against two permutation nulls in expH2_leaderboard_null.py, because micro-averaged F1
rewards a right-sized answer set whatever the ranking is worth. The best-run-per-team
variant reported alongside it addressed the non-independence of multiple runs per team,
but was tested only against zero correlation, which is not the null the section uses.
This script applies the same two nulls to the team-restricted subsets, so that the
non-independence check is read on the same scale as the full-field result.

Gated on reproducing the full-field and team-level Spearman coefficients recorded in
expH2_null_numbers.json before any new quantity is computed.

Writes expX_numbers.json.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RUNS = HERE / "expH_runs.csv"
PRIOR = HERE / "expH2_null_numbers.json"
OUT = HERE / "expX_numbers.json"

B_PERM = 10_000
SEED = 20260810


def load():
    rows = []
    with open(RUNS) as f:
        for r in csv.DictReader(f):
            rows.append({
                "task": r["task"], "team": r["team"], "run": r["run"],
                "F1": float(r["F1"]), "k": float(r["avg_set_size"]),
                "closeness": -float(r["log_dist_from_gold"]),
            })
    return rows


def nulls(f1, k, closeness, g, rng):
    """Return the two null distributions for one set of runs."""
    t = f1 * (k + g) / 2.0
    rho = stats.spearmanr(f1, closeness).statistic

    null_a = np.empty(B_PERM)
    for b in range(B_PERM):
        null_a[b] = stats.spearmanr(
            2.0 * rng.permutation(t) / (k + g), closeness).statistic

    envelope = np.minimum(k, g)
    q = t / envelope
    null_b = np.empty(B_PERM)
    for b in range(B_PERM):
        null_b[b] = stats.spearmanr(
            2.0 * rng.permutation(q) * envelope / (k + g), closeness).statistic

    def summarise(nl):
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return {
            "mean": round(float(nl.mean()), 3),
            "sd": round(float(nl.std()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "observed_percentile": round(float((nl < rho).mean() * 100.0), 1),
            "p_observed_exceeds_null": round(float((nl >= rho).mean()), 4),
        }

    return rho, summarise(null_a), summarise(null_b)


def main():
    prior = json.load(open(PRIOR))
    rows = load()
    rng = np.random.default_rng(SEED)

    out = {
        "protocol": {
            "question": "does the best-run-per-team association exceed what the "
                        "micro-F1 surface alone produces, as the full-field "
                        "association does in Task 1",
            "null": "same two nulls as expH2, applied to the team-restricted subset",
            "permutations": B_PERM, "seed": SEED,
        },
        "gates": {}, "tasks": {},
    }

    for task, gold_mean in (("T1", prior["tasks"]["T1"]["gold_mean"]),
                            ("T2", prior["tasks"]["T2"]["gold_mean"])):
        sub = [r for r in rows if r["task"] == task]

        # ---- gate: reproduce the published full-field and team-level coefficients
        rho_full = stats.spearmanr(
            np.array([r["F1"] for r in sub]),
            np.array([r["closeness"] for r in sub])).statistic
        expected_full = prior["tasks"][task]["observed_spearman_F1_vs_closeness"]
        if round(float(rho_full), 3) != expected_full:
            sys.exit(f"GATE FAILED {task}: full-field rho {rho_full:.3f} "
                     f"!= ledger {expected_full}")

        best = {}
        for r in sub:
            if r["team"] not in best or r["F1"] > best[r["team"]]["F1"]:
                best[r["team"]] = r
        bl = list(best.values())
        expected_team = prior["tasks"][task]["best_run_per_team"]["spearman"]
        rho_team_check = stats.spearmanr(
            np.array([r["F1"] for r in bl]),
            np.array([r["closeness"] for r in bl])).statistic
        if round(float(rho_team_check), 3) != expected_team:
            sys.exit(f"GATE FAILED {task}: team-level rho {rho_team_check:.3f} "
                     f"!= ledger {expected_team}")
        if len(bl) != prior["tasks"][task]["best_run_per_team"]["n_teams"]:
            sys.exit(f"GATE FAILED {task}: team count mismatch")
        out["gates"][task] = {
            "full_field_spearman_reproduced": expected_full,
            "team_level_spearman_reproduced": expected_team,
        }

        rho, na, nb = nulls(
            np.array([r["F1"] for r in bl]),
            np.array([r["k"] for r in bl]),
            np.array([r["closeness"] for r in bl]),
            gold_mean, rng)

        out["tasks"][task] = {
            "n_teams": len(bl),
            "observed_spearman": round(float(rho), 3),
            "null_A_yield_independent_of_size": na,
            "null_B_perfect_ranker_envelope": nb,
        }
        print(f"[{task}] n_teams={len(bl)} rho={rho:.3f} "
              f"nullA mean={na['mean']} pct={na['observed_percentile']} | "
              f"nullB mean={nb['mean']} pct={nb['observed_percentile']}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
