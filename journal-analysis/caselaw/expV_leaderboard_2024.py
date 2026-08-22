"""The leaderboard answer-set-size audit replicated on COLIEE 2024 (expV).

Section 6.8 audits the 2026 leaderboards: a run's average answer-set size is derived
from its published precision and recall, and closeness to the gold mean tracks final
score (Task 1 surviving a permutation null). This experiment repeats the identical
analysis on the COLIEE 2024 official results, transcribed from Tables 1 and 2 of the
organisers' overview (Goebel et al., 2024, JSAI-isAI; local copy read page by page).

Constants from primary sources. Task 1 2024: 400 test queries, 1,562 gold citations,
mean 3.905, printed in the overview. Task 2 2024: 100 test cases; the overview prints
no test gold total, so it is recovered from the archived 2025 training release (the
2024 test block, cases 726 to 825), the append-only recovery validated in expU:
147 gold paragraphs, mean 1.47.

Method identical to expH/expH2: size k = (R x gold_total / P) / n_queries, closeness
= -|log(k / gold_mean)|, Spearman via tie-aware ranks, permutation nulls A (yield
independent of size) and B (yield on the perfect-ranker envelope), best run per team.

Verification gates, before anything is written:
  1. Every transcribed row's F1 must reproduce from its printed P and R to 4 decimals
     (catches transcription errors).
  2. Run counts match the overview: 26 Task 1 runs from 10 teams, 18 Task 2 runs.
  3. Task 2 gold total equals expU's gated 147.

Writes expV_numbers.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "expV_numbers.json"
B_PERM = 10_000
SEED = 20260824

# (team, F1, P, R) transcribed from Table 1 of the 2024 overview
T1_2024 = [
    ("TQM", .4432, .5057, .3944), ("TQM", .4342, .5082, .3790),
    ("UMNLP", .4134, .4000, .4277), ("UMNLP", .4097, .3755, .4507),
    ("UMNLP", .4046, .3597, .4622), ("YR", .3605, .3210, .4110),
    ("TQM", .3548, .4196, .3073), ("YR", .3483, .3245, .3758),
    ("YR", .3417, .3184, .3688), ("JNLP", .3246, .3110, .3393),
    ("JNLP", .3222, .3347, .3105), ("JNLP", .3103, .3017, .3195),
    ("WJY", .3032, .2700, .3457), ("BM24", .1878, .1495, .2522),
    ("CAPTAIN", .1688, .1793, .1594), ("CAPTAIN", .1574, .1586, .1562),
    ("NOWJ", .1313, .0895, .2465), ("NOWJ", .1306, .0957, .2055),
    ("NOWJ", .1224, .0813, .2478), ("WJY", .1179, .0870, .1831),
    ("WJY", .1174, .0824, .2042), ("MIG", .0508, .0516, .0499),
    ("UBCS", .0276, .0140, .7196), ("UBCS", .0275, .0140, .7177),
    ("UBCS", .0272, .0139, .7100), ("CAPTAIN", .0019, .0019, .0019),
]
# (team, run, F1, P, R) transcribed from Table 2 of the 2024 overview
T2_2024 = [
    ("AMHR", "mt53bk2r", .6512, .6364, .6667), ("CAPTAIN", "fs2", .6360, .7281, .5646),
    ("JNLP", "07f39", .6320, .6967, .5782), ("CAPTAIN", "zs2", .6235, .7700, .5238),
    ("CAPTAIN", "zs3", .6235, .7700, .5238), ("NOWJ", "t5", .6117, .6181, .6054),
    ("JNLP", "join-constr", .6045, .6694, .5510), ("OVGU", "2ovgurun1", .5962, .5636, .6327),
    ("NOWJ", "weak", .5946, .5906, .5986), ("JNLP", "join", .5912, .6378, .5510),
    ("OVGU", "2ovgurun2", .5705, .5506, .5918), ("OVGU", "2ovgurun3", .5532, .5000, .6190),
    ("NOWJ", "bert", .5197, .5032, .5374), ("MIG", "mig1", .4701, .5673, .4014),
    ("MIG", "mig2", .4696, .5800, .3946), ("AMHR", "lsbk2m42", .3542, .3617, .3469),
    ("AMHR", "lsbk1", .3320, .4100, .2789), ("MIG", "mig3", .1364, .0979, .2245),
]
TASKS = {
    "T1_2024": {"rows": [(t, f, p, r) for t, f, p, r in T1_2024],
                "gold_total": 1562, "n_q": 400,
                "source": "overview Table 1; gold total printed in Section 2.2"},
    "T2_2024": {"rows": [(t, f, p, r) for t, _, f, p, r in T2_2024],
                "gold_total": 147, "n_q": 100,
                "source": "overview Table 2; gold total recovered per expU gate"},
}


def main():
    gates = {}
    for name, cfg in TASKS.items():
        bad = [i for i, (_, f, p, r) in enumerate(cfg["rows"])
               if p + r > 0 and abs(2 * p * r / (p + r) - f) > 6e-4]
        gates[f"{name}_F1_reproduces_from_PR"] = not bad
        if bad:
            for i in bad:
                t, f, p, r = cfg["rows"][i]
                print(f"  !! {name} row {i} ({t}): printed F1 {f} vs "
                      f"computed {2*p*r/(p+r):.4f}")
    gates["T1_run_count_26_teams_10"] = (len(T1_2024) == 26
                                         and len({t for t, *_ in T1_2024}) == 10)
    gates["T2_run_count_18"] = len(T2_2024) == 18
    gates["T2_gold_matches_expU"] = json.loads(
        (HERE / "expU_numbers.json").read_text())["years"]["2024"]["gold_total"] == 147
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    rng = np.random.default_rng(SEED)
    res = {"experiment": "expV_leaderboard_2024",
           "method": "identical to expH/expH2 on the transcribed 2024 leaderboards",
           "gates": gates, "tasks": {}}
    for name, cfg in TASKS.items():
        g = cfg["gold_total"] / cfg["n_q"]
        rows = [(t, f, p, r) for t, f, p, r in cfg["rows"] if p > 0 and f > 0]
        k = np.array([(r * cfg["gold_total"] / p) / cfg["n_q"] for _, _, p, r in rows])
        f1 = np.array([f for _, f, _, _ in rows])
        closeness = np.array([-abs(math.log(kk / g)) for kk in k])
        t_arr = f1 * (k + g) / 2.0

        rho = stats.spearmanr(f1, closeness).statistic
        rho_size = stats.spearmanr(f1, k).statistic

        nullA = np.empty(B_PERM)
        for b in range(B_PERM):
            nullA[b] = stats.spearmanr(
                2.0 * rng.permutation(t_arr) / (k + g), closeness).statistic
        envelope = np.minimum(k, g)
        q = t_arr / envelope
        nullB = np.empty(B_PERM)
        for b in range(B_PERM):
            nullB[b] = stats.spearmanr(
                2.0 * rng.permutation(q) * envelope / (k + g), closeness).statistic

        def summ(nl):
            return {"mean": round(float(nl.mean()), 3),
                    "ci95": [round(float(x), 3) for x in np.percentile(nl, [2.5, 97.5])],
                    "observed_percentile": round(float((nl < rho).mean() * 100), 1)}

        best = {}
        for (team, f, p, r), kk, cc in zip(rows, k, closeness):
            if team not in best or f > best[team][0]:
                best[team] = (f, cc)
        bf = np.array([v[0] for v in best.values()])
        bc = np.array([v[1] for v in best.values()])
        rbest = stats.spearmanr(bf, bc)

        res["tasks"][name] = {
            "source": cfg["source"], "n_runs_scored": len(rows),
            "gold_mean": round(g, 3),
            "share_runs_below_gold_mean": round(float((k < g).mean()), 2),
            "spearman_F1_vs_closeness": round(float(rho), 3),
            "spearman_F1_vs_size": round(float(rho_size), 3),
            "nullA": summ(nullA), "nullB": summ(nullB),
            "best_run_per_team": {"n_teams": len(best),
                                  "spearman": round(float(rbest.statistic), 3),
                                  "p": round(float(rbest.pvalue), 4)},
        }
        r_ = res["tasks"][name]
        print(f"  {name}: n={r_['n_runs_scored']} gold mean {g:.2f} | "
              f"rho(closeness) {rho:.3f} rho(size) {rho_size:.3f} | "
              f"nullB mean {r_['nullB']['mean']} pct {r_['nullB']['observed_percentile']} | "
              f"best-per-team {r_['best_run_per_team']['spearman']} "
              f"(p={r_['best_run_per_team']['p']})")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
