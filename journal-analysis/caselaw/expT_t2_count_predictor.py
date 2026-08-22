"""A learned answer-count predictor for Task 2 (expT).

Section 6.7 trains count predictors for Task 1 and the abstract summarises the outcome
as "a learned predictor recovers a fifth", but no learned predictor was ever tried on
Task 2, whose remedies table covers thresholds, elicited confidence and the classical
method only. This experiment closes that asymmetry with the same design as expG2:
predict each test case's gold paragraph count from the reranker's score shape, truncate
the fused ranking at the predicted count, and report what share of the gap between the
best fixed constant and the oracle count the policy recovers.

Training data: the 725-case main training split and the 100-case stress split, with
per-candidate ensemble scores from the archived LTR feature tables. The 100 primary
development cases are not used for training. The 100 test cases are scored from the
deployed pipeline's own cache. Per-case features are score-shape summaries of the fused
ranking (top scores, gaps, spread, counts above levels and the candidate count), the
same signal family that quadrupled recovery on Task 1.

Verification gates, before anything is written: the test-side fused ranking must
reproduce the ledgered fixed-count and oracle scores (top-1 = 0.350, top-3 = 0.465,
oracle = 0.493), and no training case id may appear among the test cases.

The distribution shift is the phenomenon under study: training cases average far fewer
gold paragraphs than the 2.94 of the test collection, so the predictor is expected to
under-predict exactly as the Task 1 regressor does, and the result is reported
whichever way it lands.

Writes expT_numbers.json.
"""
import csv
import json
import pickle
from collections import defaultdict
import os
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).parent
LTR = Path(ROOT) / "ltr_data"
CACHE = Path(ROOT) / "test_cache_monot5v2.pkl"
LABELS = Path(ROOT) / "task2_test_labels_2026(1).json"
OUT = HERE / "expT_numbers.json"
SEED = 20260823
B = 10_000


def shape_features(scores):
    """Per-case features from a descending list of fused scores."""
    s = np.array(sorted(scores, reverse=True), dtype=float)
    n = len(s)
    top = np.pad(s[:10], (0, max(0, 10 - n)), constant_values=s[-1] if n else 0.0)
    feats = {
        "n_cand": n,
        "top1": top[0], "top2": top[1], "top3": top[2], "top5": top[4],
        "gap12": top[0] - top[1], "gap23": top[1] - top[2], "gap35": top[2] - top[4],
        "mean_top5": top[:5].mean(), "std_top5": top[:5].std(),
        "mean_top10": top[:10].mean(), "std_top10": top[:10].std(),
        "range10": top[0] - top[9],
    }
    for lvl in (0.9, 0.8, 0.7, 0.6, 0.5):
        feats[f"n_above_{lvl}"] = float((s >= lvl * top[0]).sum()) if top[0] > 0 else 0.0
    return feats


def load_train_split(name):
    """Per-case fused-score lists and gold counts from an LTR feature table."""
    rows = defaultdict(list)
    gold = defaultdict(int)
    with open(LTR / name) as fh:
        for r in csv.DictReader(fh):
            rows[r["qid"]].append(float(r["ens_0802"]))
            gold[r["qid"]] += int(r["label"])
    return rows, gold


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

    fixed = {k: micro({c: ranking[c][:k] for c in test_ids}) for k in (1, 2, 3, 4)}
    oracle_f1 = micro({c: ranking[c][:len(gold_test[c])] for c in test_ids})

    tr_rows, tr_gold = load_train_split("train_main_ltr_features.csv")
    st_rows, st_gold = load_train_split("devstress_ltr_features.csv")
    train_cases = {**tr_rows, **st_rows}
    train_gold = {**tr_gold, **st_gold}

    gates = {
        "test_top1_is_0.350": abs(fixed[1] - 0.350) < 6e-4,
        "test_top3_is_0.465": abs(fixed[3] - 0.465) < 6e-4,
        "test_oracle_is_0.493": abs(oracle_f1 - 0.4932) < 6e-4,
        "no_train_test_overlap": not (set(train_cases) & set(test_ids)),
        "train_case_count": len(train_cases),
    }
    if not all(v for k, v in gates.items() if k != "train_case_count"):
        raise SystemExit(f"GATE FAILED, nothing written: {gates} "
                         f"(fixed {fixed}, oracle {oracle_f1})")

    feat_names = sorted(shape_features([1.0, 0.5]).keys())

    def matrix(case_scores):
        X = np.array([[shape_features(v)[f] for f in feat_names]
                      for v in case_scores.values()])
        return X, list(case_scores.keys())

    Xtr, tr_ids = matrix(train_cases)
    ytr = np.array([train_gold[c] for c in tr_ids], dtype=float)
    Xte, te_ids = matrix({c: fused[c] for c in test_ids})

    model = lgb.LGBMRegressor(n_estimators=400, num_leaves=15, learning_rate=0.05,
                              min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    counts = np.clip(np.rint(pred), 1, 20).astype(int)

    policy = {c: ranking[c][:k] for c, k in zip(te_ids, counts)}
    policy_f1 = micro(policy)
    gold_counts = np.array([len(gold_test[c]) for c in te_ids], dtype=float)
    corr = float(np.corrcoef(pred, gold_counts)[0, 1])
    mae = float(np.abs(counts - gold_counts).mean())
    best_fixed = max(fixed.values())
    recovery = (policy_f1 - best_fixed) / (oracle_f1 - best_fixed)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(te_ids), size=(B, len(te_ids)))
    tp_p = np.array([len(set(policy[c]) & gold_test[c]) for c in te_ids])
    np_p = np.array([len(policy[c]) for c in te_ids])
    k3 = {c: ranking[c][:3] for c in te_ids}
    tp_3 = np.array([len(set(k3[c]) & gold_test[c]) for c in te_ids])
    np_3 = np.array([3] * len(te_ids))
    ng = np.array([len(gold_test[c]) for c in te_ids])

    def f1b(tp, npred):
        TP, NP, NG = tp[idx].sum(1), npred[idx].sum(1), ng[idx].sum(1)
        P, R = TP / np.maximum(NP, 1), TP / np.maximum(NG, 1)
        return 2 * P * R / np.maximum(P + R, 1e-12)

    dif = f1b(tp_p, np_p) - f1b(tp_3, np_3)
    lo, hi = np.percentile(dif, [2.5, 97.5])

    res = {
        "experiment": "expT_t2_count_predictor",
        "design": ("LightGBM count regressor on fused-score shape features, trained on "
                   "the 725-case training split plus the 100-case stress split, applied "
                   "to the 100 test cases; the primary development split is unused"),
        "gates": gates,
        "training": {"n_cases": len(tr_ids), "gold_mean": round(float(ytr.mean()), 2)},
        "test_prediction": {"mean_pred": round(float(pred.mean()), 2),
                            "gold_mean": round(float(gold_counts.mean()), 2),
                            "count_MAE": round(mae, 2),
                            "count_corr": round(corr, 2)},
        "policies": {"fixed_k": fixed, "best_fixed": best_fixed,
                     "learned_policy_F1": policy_f1, "oracle_F1": oracle_f1},
        "gap_recovery": {
            "share_of_bestfixed_to_oracle_gap": round(float(recovery), 3),
            "learned_minus_bestfixed_pp": round(100 * (policy_f1 - best_fixed), 2),
            "paired_ci95_vs_fixed3_pp": [round(100 * float(lo), 2),
                                         round(100 * float(hi), 2)],
        },
        "reading": ("The training distribution is the shift itself: whatever the "
                    "predictor learns, it learns from collections whose counts sit far "
                    "below the test mean."),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  gates ok | train {len(tr_ids)} cases, gold mean {ytr.mean():.2f}")
    t = res["test_prediction"]
    print(f"  test: mean pred {t['mean_pred']} vs gold {t['gold_mean']} | "
          f"MAE {t['count_MAE']} corr {t['count_corr']}")
    p = res["policies"]; g = res["gap_recovery"]
    print(f"  learned policy F1 {p['learned_policy_F1']} vs best fixed "
          f"{p['best_fixed']} vs oracle {p['oracle_F1']}")
    print(f"  recovery of best-fixed-to-oracle gap: {g['share_of_bestfixed_to_oracle_gap']} "
          f"| vs fixed-3 paired CI {g['paired_ci95_vs_fixed3_pp']}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
