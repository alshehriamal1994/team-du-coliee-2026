"""Query-level bootstrap intervals for the Task 1 claims (expR).

Section 8.3 reports sampling uncertainty for every Task 2 headline and nothing for
Task 1. Three Task 1 claims deserve the same treatment: that five is the best fixed
count for our ranker (0.346 at five against 0.339 at four and 0.340 at six, Table B2),
that upgrading the ranker at a fixed five is worth 3.2 points (0.314 to 0.346), and that
supplying the true count to the upgraded ranker is worth a further 6.9 (0.346 to 0.414).

Protocol. The DU9 ranking is rebuilt exactly as in expA2 (saved booster, cached test
features, the original step8 postprocess at depth 30), the submitted DU3 predictions are
read from the official run file, and all policies are reduced to per-query true-positive,
predicted and gold counts. Resampling the 400 test queries with replacement, B = 10,000,
gives percentile intervals for each policy's micro-F1, for the paired deltas, and for the
probability that five is the argmax of the fixed-count family.

Verification gates, before anything is written: raw top-5 F1 = 0.3211, step8 top-5 =
0.3456, oracle count = 0.4143, submitted DU3 = 0.3141, and the full fixed-k sweep must
match expA2_numbers.json cell for cell.

Writes expR_numbers.json.
"""
import json
import subprocess
import sys
import os
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).parent
T1 = Path(ROOT) / "TASK1"
ARCHIVE = T1 / "ARCHIVE"
MODEL = T1 / "runs/du7_tuning/du9/model.txt"
FINAL = Path(ROOT) / "FINAL_SUBMISSION"
STEP8 = ARCHIVE / "code_AUTHORITY_v2/step8_postprocess_filters_v2.py"
CORPUS = ARCHIVE / "task_one_ready_to_use/data/task1_test_files_2026/task1_test_files_2026"
CACHE = ARCHIVE / "cache_2026/test_cache_dotxt.pkl"
EXPA2 = HERE / "expA2_numbers.json"
OUT = HERE / "expR_numbers.json"
DEPTH = 30
B = 10_000
SEED = 20260821


def norm(s):
    return s.replace(".txt", "")


def main():
    gold = {norm(k): {norm(v) for v in vs}
            for k, vs in json.load(open(FINAL / "task1_test_labels_2026.json")).items()}
    qlist = sorted(gold)
    total = sum(len(g) for g in gold.values())

    def ev(preds):
        tp = sum(len(set(p) & gold[q]) for q, p in preds.items() if q in gold)
        n = sum(len(p) for p in preds.values())
        P, R = (tp / n if n else 0.0), tp / total
        return round(2 * P * R / (P + R), 4) if P + R else 0.0

    booster = lgb.Booster(model_file=str(MODEL))
    d = np.load(ARCHIVE / "cache_features/feature_matrix_test2026.npz", allow_pickle=True)
    X, qids, cids = d["X"].astype(np.float32), d["qids"].tolist(), d["cids"].tolist()
    scores = booster.predict(X)
    per_query = {}
    for s, q, c in zip(scores, qids, cids):
        per_query.setdefault(q, []).append((c, float(s)))
    ranked = {q: [norm(c) for c, _ in sorted(p, key=lambda x: x[1], reverse=True)]
              for q, p in per_query.items()}

    gates = {"raw_top5_is_0.3211": abs(ev({q: r[:5] for q, r in ranked.items()}) - 0.3211) < 2e-4}

    base = {q + ".txt": [c + ".txt" for c in r[:100]] for q, r in ranked.items()}
    bp = HERE / "expR_base_preds.json"
    bp.write_text(json.dumps(base))
    s8_out = HERE / "expR_step8_deep.json"
    subprocess.run([sys.executable, str(STEP8), "--corpus", str(CORPUS), "--cache", str(CACHE),
                    "--base_preds", str(bp), "--out", str(s8_out), "--out_k", str(DEPTH),
                    "--rrf_k", "5", "--remove_query_cases", "--filter_future"],
                   check=True, capture_output=True, text=True)
    deep = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(s8_out)).items()}

    du3 = {}
    for line in open(FINAL / "DU3.txt"):
        q, c = line.split()[0], line.split()[1]
        du3.setdefault(norm(q), set()).add(norm(c))

    gates["step8_top5_is_0.3456"] = abs(ev({q: r[:5] for q, r in deep.items()}) - 0.3456) < 2e-4
    gates["oracle_is_0.4143"] = abs(ev({q: deep[q][:len(gold[q])] for q in deep if q in gold}) - 0.4143) < 2e-4
    gates["du3_is_0.3141"] = abs(ev(du3) - 0.3141) < 2e-4
    expa2 = json.loads(EXPA2.read_text())["fixed_k"]
    gates["fixed_k_sweep_matches_expA2"] = all(
        abs(ev({q: r[:k] for q, r in deep.items()}) - expa2[f"k{k}"]["F1"]) < 2e-4
        for k in range(1, 11))
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    # per-query (tp, npred, ngold) arrays for every policy
    policies = {"du3_submitted": {q: du3.get(q, set()) for q in qlist}}
    for k in range(1, 11):
        policies[f"du9_k{k}"] = {q: set(deep.get(q, [])[:k]) for q in qlist}
    policies["du9_oracle"] = {q: set(deep.get(q, [])[:len(gold[q])]) for q in qlist}

    arr = {}
    for name, preds in policies.items():
        tp = np.array([len(preds[q] & gold[q]) for q in qlist])
        npred = np.array([len(preds[q]) for q in qlist])
        ngold = np.array([len(gold[q]) for q in qlist])
        arr[name] = (tp, npred, ngold)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(qlist), size=(B, len(qlist)))

    def f1_boot(name):
        tp, npred, ngold = arr[name]
        TP, NP, NG = tp[idx].sum(1), npred[idx].sum(1), ngold[idx].sum(1)
        P, R = TP / np.maximum(NP, 1), TP / np.maximum(NG, 1)
        return 2 * P * R / np.maximum(P + R, 1e-12)

    boots = {name: f1_boot(name) for name in policies}

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    fixed_stack = np.stack([boots[f"du9_k{k}"] for k in range(1, 11)])
    argmax_k = fixed_stack.argmax(0) + 1
    res = {
        "experiment": "expR_t1_bootstrap",
        "protocol": {"B": B, "seed": SEED, "resampling_unit": "test query", "n_queries": len(qlist)},
        "gates": gates,
        "policy_ci95": {name: {"F1": ev({q: p for q, p in policies[name].items()}),
                               "ci95": ci(boots[name])} for name in
                        ["du3_submitted", "du9_k4", "du9_k5", "du9_k6", "du9_oracle"]},
        "peak_at_five": {
            "prob_k5_best_of_1_to_10": round(float((argmax_k == 5).mean()), 4),
            "argmax_distribution": {int(k): round(float((argmax_k == k).mean()), 4)
                                    for k in np.unique(argmax_k)},
            "delta_k5_minus_k4_ci95": ci(boots["du9_k5"] - boots["du9_k4"]),
            "delta_k5_minus_k6_ci95": ci(boots["du9_k5"] - boots["du9_k6"]),
        },
        "decomposition_deltas_pp": {
            "ranker_upgrade_at_5": {"point": 3.15,
                                    "ci95": [round(100 * x, 2) for x in
                                             ci(boots["du9_k5"] - boots["du3_submitted"])]},
            "true_count_on_upgraded": {"point": 6.87,
                                       "ci95": [round(100 * x, 2) for x in
                                                ci(boots["du9_oracle"] - boots["du9_k5"])]},
            "prob_stopping_exceeds_ranker_upgrade": round(float(
                ((boots["du9_oracle"] - boots["du9_k5"]) >
                 (boots["du9_k5"] - boots["du3_submitted"])).mean()), 4),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    for name, v in res["policy_ci95"].items():
        print(f"  {name:16s} F1 {v['F1']:.4f}  CI {v['ci95']}")
    p = res["peak_at_five"]
    print(f"  P(k=5 best of 1..10) = {p['prob_k5_best_of_1_to_10']:.3f}  "
          f"argmax dist {p['argmax_distribution']}")
    dd = res["decomposition_deltas_pp"]
    print(f"  ranker upgrade  +3.15pp CI {dd['ranker_upgrade_at_5']['ci95']}")
    print(f"  true count      +6.87pp CI {dd['true_count_on_upgraded']['ci95']}")
    print(f"  P(stopping > ranker upgrade) = {dd['prob_stopping_exceeds_ranker_upgrade']}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
