"""What the answer-count decision is worth as ranking improves (journal article, Section 6).

Under a fixed count k, a perfect ranker recovers sum_q min(k, n_q) of the gold, so its
ceiling follows from the gold count distribution alone, while the per-query oracle is
capped only by what the candidate pool contains. The direction of the relation is
therefore arithmetic. This script measures the magnitude. Ranker quality is a single
parameter alpha: each gold citation at rank r in the full ranking moves to
floor(r(1 - alpha)), so alpha = 0 reproduces the system's ranking and alpha = 1 places
every retrievable gold citation at the top, with queries, gold and metric unchanged. At
each level the best fixed count is compared with the true per-query count.

Anchored at alpha = 0 to the raw top-five figure recorded in expA2_numbers.json and at
alpha = 1 to the candidate pool's recall. Writes expM_count_value_numbers.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import os

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))

HERE = Path(__file__).resolve().parent
MODEL = ROOT / "TASK1/runs/du7_tuning/du9/model.txt"
FEATS = ROOT / "TASK1/ARCHIVE/cache_features/feature_matrix_test2026.npz"
GOLD = ROOT / "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"
OUT = HERE / "expM_count_value_numbers.json"

ALPHAS = [round(a, 2) for a in np.arange(0.0, 1.0001, 0.05)]
K_GRID = list(range(1, 16))


def norm(s):
    return s.replace(".txt", "")


def shifted_ranks(ranks, alpha):
    """Move each gold rank towards the top by alpha, keeping ranks distinct and ordered."""
    out, taken = [], -1
    for r in sorted(ranks):
        new = int(np.floor(r * (1.0 - alpha)))
        if new <= taken:
            new = taken + 1
        out.append(new)
        taken = new
    return out


def micro_f1(tp, npred, ngold):
    if npred == 0 or ngold == 0:
        return 0.0
    p, r = tp / npred, tp / ngold
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    gold = {norm(k): set(norm(v) for v in vs) for k, vs in json.load(open(GOLD)).items()}
    queries = sorted(gold)
    total_gold = sum(len(gold[q]) for q in queries)

    import lightgbm as lgb
    d = np.load(FEATS, allow_pickle=True)
    X = d["X"].astype(np.float32)
    qids = [norm(q) for q in d["qids"].tolist()]
    cids = [norm(c) for c in d["cids"].tolist()]
    print("  scoring the candidate pool ...", flush=True)
    scores = lgb.Booster(model_file=str(MODEL)).predict(X)

    pool = {}
    for s, q, c in zip(scores, qids, cids):
        pool.setdefault(q, []).append((float(s), c))

    # rank of every gold citation in our full ranking, per query
    gold_ranks, pool_size = {}, {}
    for q in queries:
        order = sorted(pool.get(q, []), key=lambda x: -x[0])
        pool_size[q] = len(order)
        pos = {c: i for i, (_, c) in enumerate(order)}
        gold_ranks[q] = sorted(pos[c] for c in gold[q] if c in pos)
    found = sum(len(v) for v in gold_ranks.values())
    print(f"  {len(queries)} queries, {total_gold} gold, {found} located in the pool "
          f"({100 * found / total_gold:.1f}%)")

    def evaluate(alpha):
        shifted = {q: shifted_ranks(gold_ranks[q], alpha) for q in queries}
        best_k, best_f1 = None, -1.0
        for k in K_GRID:
            tp = sum(sum(1 for r in shifted[q] if r < k) for q in queries)
            f = micro_f1(tp, k * len(queries), total_gold)
            if f > best_f1:
                best_k, best_f1 = k, f
        tp5 = sum(sum(1 for r in shifted[q] if r < 5) for q in queries)
        f5 = micro_f1(tp5, 5 * len(queries), total_gold)
        tp_o = sum(sum(1 for r in shifted[q] if r < len(gold[q])) for q in queries)
        npred_o = sum(len(gold[q]) for q in queries)
        f_o = micro_f1(tp_o, npred_o, total_gold)
        return {"best_k": best_k, "best_fixed_F1": round(best_f1, 4),
                "fixed5_F1": round(f5, 4), "oracle_F1": round(f_o, 4),
                "count_value_pp": round((f_o - best_f1) * 100, 1),
                "count_value_vs_fixed5_pp": round((f_o - f5) * 100, 1)}

    lo, hi = evaluate(0.0), evaluate(1.0)
    a2 = json.load(open(HERE / "expA2_numbers.json"))
    raw5 = a2["meta"]["sanity_raw_top5"]["F1"]
    pool_recall = found / total_gold
    checks = {
        "alpha0 = expA2 raw top-5": abs(lo["fixed5_F1"] - raw5) < 2e-3,
        "alpha1 oracle = pool recall": abs(hi["oracle_F1"] - pool_recall) < 2e-3,
        "count value rises with quality": hi["count_value_pp"] > lo["count_value_pp"],
    }
    for k, v in checks.items():
        print(f"  gate {k:26s} {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        print(f"  observed: alpha0 {lo}\n            alpha1 {hi}")
        print("GATE FAILED - nothing reported.")
        sys.exit(1)

    curve = {}
    for a in ALPHAS:
        curve[a] = evaluate(a)
        c = curve[a]
        print(f"  alpha {a:4.2f}  best fixed k={c['best_k']:2d} F1 {c['best_fixed_F1']:.4f}"
              f"   oracle {c['oracle_F1']:.4f}   count worth {c['count_value_pp']:+5.1f}")

    vals = [curve[a]["count_value_pp"] for a in ALPHAS]
    res = {
        "protocol": {
            "alpha": "fraction by which each gold citation's rank is compressed towards "
                     "the top; 0 is our ranker, 1 is a perfect ranker",
            "held_fixed": "queries, gold, micro-averaged F1, candidate pool",
            "count_value": "score under the true per-query count minus score under the "
                           "best fixed count at that ranker quality",
            "n_queries": len(queries), "total_gold": total_gold,
        },
        "curve": curve,
        "headline": {
            "count_value_at_our_ranker_pp": vals[0],
            "count_value_at_perfect_ranker_pp": vals[-1],
            "ratio": round(vals[-1] / vals[0], 1) if vals[0] else None,
            "monotone_increasing": bool(all(b >= a - 0.2 for a, b in zip(vals, vals[1:]))),
        },
    }
    print(f"\n  count knowledge is worth {vals[0]:.1f} points to our ranker and "
          f"{vals[-1]:.1f} to a perfect one, a factor of {res['headline']['ratio']}")
    print(f"  monotone increasing in ranker quality: {res['headline']['monotone_increasing']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
