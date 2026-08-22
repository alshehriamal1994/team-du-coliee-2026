"""The count decision's value across real rankers (journal article, Section 6).

Seven rankers over the identical Task 1 candidate pool, four first-stage signals, one
fusion and three trained LambdaRank models of different capacity, with queries, gold and
metric held fixed. For each, the best fixed count over k = 1..15 is compared with the
true per-query count. The strongest ranker must reproduce the simulation's anchor in
expM_count_value_numbers.json before anything is reported. The Task 2 reranker upgrade,
monoT5 v1 to v2, provides the one real upgrade of the deployed pipeline and is reported
alongside. CPU inference only.

Writes expN_real_rankers_numbers.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import os

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))

HERE = Path(__file__).resolve().parent
FEATS = ROOT / "TASK1/ARCHIVE/cache_features/feature_matrix_test2026.npz"
GOLD = ROOT / "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"
RUNS = ROOT / "TASK1/runs"
OUT = HERE / "expN_real_rankers_numbers.json"

K_GRID = list(range(1, 16))

FEATURE_RANKERS = [
    ("paragraph Jaccard", "para_max_jaccard"),
    ("dense retrieval", "de"),
    ("BM25", "bm_norm"),
    ("reciprocal rank fusion", "rrf10"),
]
MODEL_RANKERS = [
    ("LambdaRank (du4)", RUNS / "du4_bigger/du4/model.txt"),
    ("LambdaRank (du7)", RUNS / "du7_tuning/du7/model.txt"),
    ("LambdaRank (du9)", RUNS / "du7_tuning/du9/model.txt"),
]


def norm(s):
    return s.replace(".txt", "")


def micro_f1(tp, npred, ngold):
    if npred == 0 or ngold == 0:
        return 0.0
    p, r = tp / npred, tp / ngold
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate(order_by_query, gold, total_gold):
    """order_by_query: query -> candidate ids ranked best first."""
    best_k, best_f1 = None, -1.0
    for k in K_GRID:
        tp = sum(len(set(v[:k]) & gold[q]) for q, v in order_by_query.items())
        f = micro_f1(tp, k * len(order_by_query), total_gold)
        if f > best_f1:
            best_k, best_f1 = k, f
    tp_o = sum(len(set(v[:len(gold[q])]) & gold[q]) for q, v in order_by_query.items())
    n_o = sum(len(gold[q]) for q in order_by_query)
    f_o = micro_f1(tp_o, n_o, total_gold)
    return {"best_k": best_k, "best_fixed_F1": round(best_f1, 4),
            "oracle_F1": round(f_o, 4),
            "count_value_pp": round((f_o - best_f1) * 100, 1)}


def main():
    gold = {norm(k): set(norm(v) for v in vs) for k, vs in json.load(open(GOLD)).items()}
    total_gold = sum(len(v) for v in gold.values())

    d = np.load(FEATS, allow_pickle=True)
    X = d["X"].astype(np.float32)
    names = d["feature_names"].tolist()
    qids = [norm(q) for q in d["qids"].tolist()]
    cids = [norm(c) for c in d["cids"].tolist()]
    idx_by_q = {}
    for i, q in enumerate(qids):
        idx_by_q.setdefault(q, []).append(i)
    print(f"  {len(idx_by_q)} queries, {total_gold} gold citations")

    def order_from(scores):
        out = {}
        for q, ii in idx_by_q.items():
            if q not in gold:
                continue
            ranked = sorted(ii, key=lambda i: -scores[i])
            out[q] = [cids[i] for i in ranked]
        return out

    results = []
    for label, feat in FEATURE_RANKERS:
        s = X[:, names.index(feat)].astype(float)
        r = evaluate(order_from(s), gold, total_gold)
        r["ranker"] = label
        r["kind"] = "first-stage signal"
        results.append(r)
        print(f"  {label:26s} best fixed {r['best_fixed_F1']:.4f} "
              f"oracle {r['oracle_F1']:.4f}  count worth {r['count_value_pp']:+5.1f}")

    import lightgbm as lgb
    for label, path in MODEL_RANKERS:
        if not path.exists():
            print(f"  [skip] {label}: model not found")
            continue
        s = lgb.Booster(model_file=str(path)).predict(X)
        r = evaluate(order_from(s), gold, total_gold)
        r["ranker"] = label
        r["kind"] = "trained model"
        results.append(r)
        print(f"  {label:26s} best fixed {r['best_fixed_F1']:.4f} "
              f"oracle {r['oracle_F1']:.4f}  count worth {r['count_value_pp']:+5.1f}")

    # ---- gate on DU9 against the simulation's anchor ----
    m = json.load(open(HERE / "expM_count_value_numbers.json"))
    anchor = m["headline"]["count_value_at_our_ranker_pp"]
    du9 = next((r for r in results if "du9" in r["ranker"]), None)
    ok = du9 is not None and abs(du9["count_value_pp"] - anchor) < 0.25
    print(f"\n  gate du9 vs expM alpha=0: {du9['count_value_pp'] if du9 else 'n/a'} "
          f"vs {anchor}  {'PASS' if ok else 'FAIL'}")

    # ---- Task 2, the one real reranker upgrade ----
    f = json.load(open(HERE / "expF_numbers.json"))
    t2 = {}
    for v in ("v1", "v2"):
        ks = {k: f[v][f"fixed_k{k}"]["F1"] for k in (1, 2, 3)}
        bk = max(ks, key=ks.get)
        t2[v] = {"best_k": bk, "best_fixed_F1": ks[bk],
                 "oracle_F1": f[v]["oracle_count"]["F1"],
                 "recall_at_20": f[v]["recall@20"],
                 "count_value_pp": round((f[v]["oracle_count"]["F1"] - ks[bk]) * 100, 1)}
    ok2 = abs(t2["v2"]["oracle_F1"] - 0.4932) < 5e-4 and abs(t2["v1"]["oracle_F1"] - 0.3980) < 5e-4
    print(f"  gate Task 2 vs expF: {'PASS' if ok2 else 'FAIL'}")
    if not (ok and ok2):
        print("GATE FAILED - nothing written.")
        sys.exit(1)

    order = sorted(results, key=lambda r: r["best_fixed_F1"])
    q = np.array([r["best_fixed_F1"] for r in order])
    v = np.array([r["count_value_pp"] for r in order])
    share = v / (np.array([r["oracle_F1"] for r in order]) * 100)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    res = {
        "protocol": {
            "purpose": "replace the simulated middle of the count-value curve with real "
                       "rankers over the identical candidate pool",
            "held_fixed": "queries, gold, candidate pool, micro-averaged F1",
            "k_grid": [K_GRID[0], K_GRID[-1]],
            "gate": f"du9 count value must match expM alpha=0 ({anchor} points)",
        },
        "task1_rankers": order,
        "task1_relation": {
            "spearman_quality_vs_count_value": round(spearman(q, v), 3),
            "pearson_quality_vs_count_value": round(float(np.corrcoef(q, v)[0, 1]), 3),
            "range_best_fixed_F1": [round(float(q.min()), 4), round(float(q.max()), 4)],
            "range_count_value_pp": [round(float(v.min()), 1), round(float(v.max()), 1)],
            "spearman_quality_vs_count_share": round(spearman(q, share), 3),
            "caveat": "as a share of oracle F1 the trend is much weaker, reported so that "
                      "the absolute relation is not mistaken for a scale-free one",
        },
        "task2_reranker_upgrade": t2,
    }
    r1 = res["task1_relation"]
    print(f"\n  {len(order)} real Task 1 rankers, best-fixed F1 "
          f"{r1['range_best_fixed_F1'][0]} to {r1['range_best_fixed_F1'][1]}")
    print(f"  count value {r1['range_count_value_pp'][0]} to "
          f"{r1['range_count_value_pp'][1]} points, Spearman {r1['spearman_quality_vs_count_value']}")
    print(f"  as a share of oracle F1, Spearman {r1['spearman_quality_vs_count_share']} (the caveat)")
    print(f"  Task 2 upgrade: count worth {t2['v1']['count_value_pp']} -> "
          f"{t2['v2']['count_value_pp']} as recall@20 rises "
          f"{t2['v1']['recall_at_20']:.3f} -> {t2['v2']['recall_at_20']:.3f}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
