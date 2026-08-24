"""Split-conformal answer sets for Task 2 (expAB).

Section 8.1 records conformal prediction as the one principled route not tested,
and predicts that a conformal set calibrated before the test would inherit the
answer-count shift as the learned predictor of Section 8.2 does. This experiment
runs the test. The prediction's reading is registered here before the run:
inheritance would appear as achieved coverage on the test collection falling
below the nominal level calibrated for, or as set sizes that track the
calibration collections rather than the test's. The outcome is genuinely open,
because the conformal threshold is set on scores rather than on counts, so its
sets grow with however many candidates clear the threshold, and the result is
reported whichever way it lands.

Method: standard split conformal. For each calibration case the nonconformity
score is one minus the lowest fused score of any gold paragraph among its
candidates, so the calibrated threshold is the score level needed to capture
every retrievable gold. For a nominal coverage level the threshold is the usual
finite-sample quantile of the calibration nonconformity scores, and the test-set
prediction for a case is every candidate whose fused score clears it. Coverage
is the fraction of test cases whose retrievable gold is entirely contained in
the returned set.

Calibration: the 725-case training split plus the 100-case stress split, the
same 825 pre-test cases as expT, with per-candidate fused ensemble scores from
the archived LTR feature tables (per-case min-max normalised components, 0.8/0.2
weights, the deployed fusion). The 100 primary development cases are unused. The
test side reproduces the deployed fusion from the pipeline's own cache.

Because calibration coverage matching the nominal level is guaranteed by
construction, the non-tautological check is the held-out primary development
split, 100 cases at a gold mean of 1.22 that entered neither the calibration nor
any tuning: if exchangeability is what fails, coverage should roughly hold there
and collapse on the shifted test collection, the same dose-response structure as
the 2025 replication of the prompt result.

Verification gates, before anything is written: the test-side fused ranking must
reproduce the ledgered fixed-count and oracle scores (top-1 = 0.350,
top-3 = 0.465, oracle = 0.493), the calibration split must hold 825 cases with
no overlap with the test cases or the held-out split, and calibration coverage
at each nominal level must reach that level, which split conformal guarantees by
construction.

Writes expAB_numbers.json.
"""
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = os.environ.get("COLIEE_ROOT", "data")
HERE = Path(__file__).parent
LTR = Path(ROOT) / "ltr_data"
CACHE = Path(ROOT) / "test_cache_monot5v2.pkl"
LABELS = Path(ROOT) / "task2_test_labels_2026(1).json"
OUT = HERE / "expAB_numbers.json"

NOMINAL = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def load_split(name):
    """Per-case candidate score lists and gold flags from an LTR table."""
    scores = defaultdict(list)
    flags = defaultdict(list)
    with open(LTR / name) as fh:
        for r in csv.DictReader(fh):
            scores[r["qid"]].append(float(r["ens_0802"]))
            flags[r["qid"]].append(int(r["label"]))
    return scores, flags


def main():
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    raw_gold = json.load(open(LABELS))
    gold_test = {cid: {x.strip().replace(".txt", "").zfill(3)
                       for x in val.split(",") if x.strip()}
                 for cid, val in raw_gold.items()}
    total_gold = sum(len(g) for g in gold_test.values())

    ranking, fused = {}, {}
    for row in cache["rows"]:
        cid = row["cid"]
        m5, q3 = np.array(row["m5"]), np.array(row["q3"])
        pids = [p.zfill(3) for p in row["cand_ids"]]
        r1, r2 = m5.max() - m5.min(), q3.max() - q3.min()
        n1 = np.ones_like(m5) if r1 < 1e-9 else (m5 - m5.min()) / r1
        n2 = np.ones_like(q3) if r2 < 1e-9 else (q3 - q3.min()) / r2
        comb = 0.8 * n1 + 0.2 * n2
        order = np.argsort(-comb)
        ranking[cid] = [pids[i] for i in order]
        fused[cid] = [float(comb[i]) for i in order]
    test_ids = sorted(gold_test)

    def micro(preds):
        tp = sum(len(set(preds[c]) & gold_test[c]) for c in test_ids)
        n = sum(len(preds[c]) for c in test_ids)
        P, R = (tp / n if n else 0.0), tp / total_gold
        return round(2 * P * R / (P + R), 4) if P + R else 0.0

    fixed = {k: micro({c: ranking[c][:k] for c in test_ids}) for k in (1, 3)}
    oracle_f1 = micro({c: ranking[c][:len(gold_test[c])] for c in test_ids})

    tr_s, tr_f = load_split("train_main_ltr_features.csv")
    st_s, st_f = load_split("devstress_ltr_features.csv")
    cal_scores = {**tr_s, **st_s}
    cal_flags = {**tr_f, **st_f}
    ho_scores, ho_flags = load_split("dev_ltr_features.csv")

    # nonconformity: the score level needed to capture every retrievable gold
    alphas, dropped = [], 0
    for qid, ss in cal_scores.items():
        gold_scores = [s for s, l in zip(ss, cal_flags[qid]) if l == 1]
        if not gold_scores:
            dropped += 1
            continue
        alphas.append(1.0 - min(gold_scores))
    alphas = np.array(sorted(alphas))
    n_cal = len(alphas)

    gates = {
        "test_top1_is_0.350": abs(fixed[1] - 0.350) < 6e-4,
        "test_top3_is_0.465": abs(fixed[3] - 0.465) < 6e-4,
        "test_oracle_is_0.493": abs(oracle_f1 - 0.4932) < 6e-4,
        "no_train_test_overlap": not (set(cal_scores) & set(test_ids)),
        "heldout_disjoint_from_calibration": not (set(ho_scores) & set(cal_scores)),
        "heldout_case_count": len(ho_scores),
        "calibration_case_count": len(cal_scores),
        "calibration_cases_without_retrievable_gold": dropped,
    }
    if not all(v for k, v in gates.items()
               if k not in ("calibration_case_count", "heldout_case_count",
                            "calibration_cases_without_retrievable_gold")):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")
    if len(cal_scores) != 825:
        raise SystemExit(f"GATE FAILED: {len(cal_scores)} calibration cases, expected 825")

    # retrievable gold on the test side, for the coverage target
    retrievable = {c: gold_test[c] & set(ranking[c]) for c in test_ids}
    n_fully_retrievable = sum(1 for c in test_ids if retrievable[c] == gold_test[c])

    def coverage_cal(tau):
        return float((alphas <= tau).mean())

    rows = {}
    for level in NOMINAL:
        k = math.ceil((n_cal + 1) * level)
        if k > n_cal:
            tau = float("inf")
        else:
            tau = float(alphas[k - 1])
        preds = {c: [p for p, s in zip(ranking[c], fused[c]) if s >= 1.0 - tau]
                 for c in test_ids}
        sizes = np.array([len(preds[c]) for c in test_ids], dtype=float)
        cov_retr = float(np.mean([retrievable[c] <= set(preds[c]) for c in test_ids]))
        cov_all = float(np.mean([gold_test[c] <= set(preds[c]) for c in test_ids]))
        gold_counts = np.array([len(gold_test[c]) for c in test_ids], dtype=float)
        ho_cov, ho_sizes = [], []
        for qid, ss in ho_scores.items():
            sel = [1 for sc in ss if sc >= 1.0 - tau]
            golds = [sc for sc, l in zip(ss, ho_flags[qid]) if l == 1]
            ho_cov.append(all(sc >= 1.0 - tau for sc in golds))
            ho_sizes.append(sum(sel))
        rows[f"{level:.2f}"] = {
            "coverage_heldout_dev": round(float(np.mean(ho_cov)), 4),
            "mean_set_size_heldout": round(float(np.mean(ho_sizes)), 2),
            "threshold_tau": round(tau, 4) if math.isfinite(tau) else "inf",
            "coverage_calibration": round(coverage_cal(tau), 4),
            "coverage_test_retrievable_gold": round(cov_retr, 4),
            "coverage_test_all_gold": round(cov_all, 4),
            "mean_set_size": round(float(sizes.mean()), 2),
            "median_set_size": float(np.median(sizes)),
            "empty_sets": int((sizes == 0).sum()),
            "micro_F1": micro(preds),
            "count_MAE": round(float(np.abs(sizes - gold_counts).mean()), 2),
        }

    cal_gate = all(rows[f"{lv:.2f}"]["coverage_calibration"] >= lv - 1e-9
                   for lv in NOMINAL if rows[f"{lv:.2f}"]["threshold_tau"] != "inf")
    gates["calibration_coverage_reaches_nominal"] = cal_gate
    if not cal_gate:
        raise SystemExit(f"GATE FAILED: calibration coverage below nominal: {rows}")

    best_lv = max(rows, key=lambda k: rows[k]["micro_F1"])
    out = {
        "experiment": "expAB_conformal",
        "registered_reading": "inheritance of the shift = test coverage below "
                              "nominal, or set sizes tracking the calibration "
                              "collections rather than the test's",
        "heldout_dev": {
            "n_cases": len(ho_scores),
            "gold_mean": round(float(np.mean(
                [sum(f) for f in ho_flags.values()])), 3),
        },
        "calibration": {
            "n_cases": n_cal,
            "gold_mean": round(float(np.mean(
                [sum(f) for f in cal_flags.values()])), 3),
            "dropped_no_retrievable_gold": dropped,
        },
        "test": {
            "n_cases": len(test_ids),
            "gold_mean": round(total_gold / len(test_ids), 2),
            "cases_fully_retrievable": n_fully_retrievable,
        },
        "gates": gates,
        "reference": {"fixed_top1": fixed[1], "fixed_top3": fixed[3],
                      "oracle": oracle_f1},
        "by_nominal_level": rows,
        "headline_nominal_090": rows["0.90"],
        "best_F1_over_sweep": {"nominal_level": best_lv, **rows[best_lv]},
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"calibration n={n_cal} gold mean {out['calibration']['gold_mean']}, "
          f"test gold mean {out['test']['gold_mean']}")
    print(f"{'nominal':>8} {'tau':>7} {'cov cal':>8} {'cov dev':>8} {'cov test':>9} "
          f"{'sz dev':>6} {'sz te':>6} {'F1':>7} {'MAE':>6}")
    for lv in NOMINAL:
        r = rows[f"{lv:.2f}"]
        print(f"{lv:8.2f} {str(r['threshold_tau']):>7} "
              f"{r['coverage_calibration']:8.3f} "
              f"{r['coverage_heldout_dev']:8.3f} "
              f"{r['coverage_test_retrievable_gold']:9.3f} "
              f"{r['mean_set_size_heldout']:6.2f} "
              f"{r['mean_set_size']:6.2f} {r['micro_F1']:7.4f} "
              f"{r['count_MAE']:6.2f}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
