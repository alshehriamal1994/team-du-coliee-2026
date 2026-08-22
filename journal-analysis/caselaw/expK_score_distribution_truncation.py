"""Score-distribution truncation on the Task 1 ranker (journal article, Section 6).

An implementation of the classical approach of Arampatzis, Kamps and Robertson (2009):
each query's score distribution over the full candidate pool is modelled as a
two-component mixture, relevant scores Normal and non-relevant scores Exponential, fitted
by expectation maximisation, and the cutoff maximises expected micro-F1 under the fitted
relevance probabilities. The ranking itself is held fixed so that only the stopping rule
changes.

The script also records why the method behaves as it does on this collection, measuring
the separation between gold and non-gold scores and the distribution of gold ranks.
Gated on reproducing the published fixed-five and oracle anchors before reporting.
Writes expK_numbers.json.
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
DEEP = HERE / "expA2_step8_deep.json"
OUT = HERE / "expK_numbers.json"

FIXED5 = 0.3456      # published ladder anchor
ORACLE = 0.4143      # published ladder anchor
EM_ITERS = 200
EPS = 1e-12


def norm(s):
    return s.replace(".txt", "")


def fit_normal_exponential(scores, iters=EM_ITERS):
    """EM for a Normal (relevant) plus Exponential (non-relevant) score mixture.

    Scores are shifted so the minimum is zero, which the exponential component requires.
    Returns per-score P(relevant | score). Falls back to a flat prior if EM degenerates,
    which happens when a query's scores carry no separable structure.
    """
    s = np.asarray(scores, dtype=float)
    s = s - s.min()
    n = len(s)
    if n < 20 or s.max() <= 0:
        return np.full(n, 0.5)

    # initialise: top decile as the relevant component
    thr = np.quantile(s, 0.9)
    pi = 0.1
    mu, sigma = s[s >= thr].mean(), max(s[s >= thr].std(), 1e-3)
    lam = 1.0 / max(s[s < thr].mean(), 1e-3)

    for _ in range(iters):
        pdf_r = np.exp(-0.5 * ((s - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        pdf_n = lam * np.exp(-lam * s)
        num = pi * pdf_r
        den = num + (1 - pi) * pdf_n + EPS
        g = num / den                                   # responsibility, P(rel | s)
        nr = g.sum()
        if not np.isfinite(nr) or nr < 1e-6 or nr > n - 1e-6:
            return np.full(n, 0.5)
        pi_new = nr / n
        mu_new = float((g * s).sum() / nr)
        sigma_new = float(np.sqrt(max((g * (s - mu_new) ** 2).sum() / nr, 1e-6)))
        lam_new = float((n - nr) / max(((1 - g) * s).sum(), 1e-6))
        if (abs(pi_new - pi) < 1e-8 and abs(mu_new - mu) < 1e-8
                and abs(sigma_new - sigma) < 1e-8):
            pi, mu, sigma, lam = pi_new, mu_new, sigma_new, lam_new
            break
        pi, mu, sigma, lam = pi_new, mu_new, sigma_new, lam_new

    pdf_r = np.exp(-0.5 * ((s - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    pdf_n = lam * np.exp(-lam * s)
    return np.clip(pi * pdf_r / (pi * pdf_r + (1 - pi) * pdf_n + EPS), 0.0, 1.0)


def micro_f1(sel_per_q, gold, total_gold):
    tp = sum(len(set(v) & set(gold[q])) for q, v in sel_per_q.items())
    n = sum(len(v) for v in sel_per_q.values())
    if n == 0:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "n_pred": 0}
    P, R = tp / n, tp / total_gold
    f = 0.0 if P + R == 0 else 2 * P * R / (P + R)
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(f, 4), "n_pred": n}


def main():
    gold = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(GOLD)).items()}
    deep = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(DEEP)).items()}
    queries = sorted(gold)
    total_gold = sum(len(v) for v in gold.values())
    print(f"  {len(queries)} test queries, {total_gold} gold citations")

    # ---- gate: reproduce the two published anchors ------------------------
    fixed5 = micro_f1({q: deep.get(q, [])[:5] for q in queries}, gold, total_gold)
    oracle = micro_f1({q: deep.get(q, [])[:len(gold[q])] for q in queries},
                      gold, total_gold)
    ok = (abs(fixed5["F1"] - FIXED5) < 5e-4) and (abs(oracle["F1"] - ORACLE) < 5e-4)
    print(f"  gate fixed-five  {fixed5['F1']:.4f} vs published {FIXED5}")
    print(f"  gate oracle      {oracle['F1']:.4f} vs published {ORACLE}")
    print(f"  gate: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("GATE FAILED - no baseline computed.")
        sys.exit(1)

    # ---- ranker scores over the full candidate pool -----------------------
    import lightgbm as lgb
    d = np.load(FEATS, allow_pickle=True)
    X = d["X"].astype(np.float32)
    qids = [norm(q) for q in d["qids"].tolist()]
    cids = [norm(c) for c in d["cids"].tolist()]
    print("  scoring the candidate pool with the saved DU9 ranker ...", flush=True)
    scores = lgb.Booster(model_file=str(MODEL)).predict(X)

    pool = {}
    for s, q, c in zip(scores, qids, cids):
        pool.setdefault(q, []).append((float(s), c))
    print(f"  scored {len(pool)} query pools")

    # ---- fit the mixture per query, then pick the cutoff ------------------
    chosen, probs_at, degenerate = {}, {}, 0
    for q in queries:
        entries = pool.get(q, [])
        if not entries:
            chosen[q] = 5
            continue
        ss = np.array([e[0] for e in entries])
        p = fit_normal_exponential(ss)
        if np.allclose(p, 0.5):
            degenerate += 1
        pmap = {c: float(pi) for (_, c), pi in zip(entries, p)}
        probs_at[q] = pmap
        rhat = float(p.sum())

        lst = deep.get(q, [])
        if not lst:
            chosen[q] = 5
            continue
        pv = np.array([pmap.get(c, 0.0) for c in lst])
        cum = np.cumsum(pv)
        ks = np.arange(1, len(lst) + 1)
        exp_f1 = 2.0 * cum / (ks + rhat)
        chosen[q] = int(ks[int(np.argmax(exp_f1))])

    # ---- diagnostic: does the score distribution have the structure the ----
    # ---- method assumes, namely a separable high-scoring relevant mass?  ----
    sep, gold_ranks, beyond30 = [], [], 0
    n_gold_total = 0
    for q in queries:
        entries = sorted(pool.get(q, []), key=lambda x: -x[0])
        if not entries:
            continue
        sv = np.array([e[0] for e in entries])
        rel = np.array([e[1] in set(gold[q]) for e in entries])
        n_gold_total += int(rel.sum())
        if rel.sum() == 0:
            continue
        ranks = np.where(rel)[0]
        gold_ranks.extend(ranks.tolist())
        beyond30 += int((ranks >= 30).sum())
        spread = sv.max() - sv.min()
        if spread > 0:
            sep.append(float((sv[rel].min() - np.percentile(sv[~rel], 99)) / spread))
    sep = np.array(sep)
    gold_ranks = np.array(gold_ranks)

    counts = np.array([chosen[q] for q in queries], dtype=float)
    gold_counts = np.array([len(gold[q]) for q in queries], dtype=float)
    sd = micro_f1({q: deep.get(q, [])[:chosen[q]] for q in queries}, gold, total_gold)

    gap = ORACLE - FIXED5
    res = {
        "meta": {
            "experiment": "classical score-distribution truncation on the Task 1 ranker",
            "method": "per-query Normal plus Exponential score mixture by EM, cutoff "
                      "maximising expected micro-F1",
            "reference": "arampatzis2009stop",
            "ranking_held_fixed": "expA2 step8-filtered deep ranking, identical to every "
                                  "other rung of the ladder",
            "n_queries": len(queries), "total_gold": total_gold,
            "em_iters": EM_ITERS, "degenerate_fits": degenerate,
        },
        "ladder": {"fixed5": FIXED5, "oracle": ORACLE},
        "gate": {"fixed5_reproduced": fixed5["F1"], "oracle_reproduced": oracle["F1"]},
        "score_distribution_truncation": sd,
        "counts": {
            "mean_predicted": round(float(counts.mean()), 2),
            "median_predicted": float(np.median(counts)),
            "min": float(counts.min()), "max": float(counts.max()),
            "gold_mean": round(float(gold_counts.mean()), 2),
            "MAE": round(float(np.abs(counts - gold_counts).mean()), 3),
            "corr_with_gold": round(float(np.corrcoef(counts, gold_counts)[0, 1]), 3),
        },
        "gap_closed_pct": round(100.0 * (sd["F1"] - FIXED5) / gap, 1),
        "why_it_fails": {
            "note": "the method presumes the relevant documents form a separable "
                    "high-scoring component of the score distribution",
            "separation_median": round(float(np.median(sep)), 3),
            "separation_note": "minimum gold score minus the 99th percentile of "
                               "non-gold scores, divided by the score range. Positive "
                               "would mean gold sits clear of the non-gold bulk",
            "pct_separation_positive": round(float((sep > 0).mean() * 100), 1),
            "gold_rank_median": float(np.median(gold_ranks)),
            "gold_rank_p90": float(np.percentile(gold_ranks, 90)),
            "pct_gold_beyond_rank_30": round(100.0 * beyond30 / max(n_gold_total, 1), 1),
        },
    }
    print(f"\n  score-distribution truncation: F1 {sd['F1']:.4f}  "
          f"(fixed five {FIXED5}, oracle {ORACLE})")
    print(f"  gap closed: {res['gap_closed_pct']}%")
    print(f"  counts: mean {res['counts']['mean_predicted']}, "
          f"gold mean {res['counts']['gold_mean']}, "
          f"MAE {res['counts']['MAE']}, corr {res['counts']['corr_with_gold']}")
    print(f"  degenerate mixture fits: {degenerate} of {len(queries)}")
    w = res["why_it_fails"]
    print(f"\n  WHY: separation median {w['separation_median']}, "
          f"positive for only {w['pct_separation_positive']}% of queries")
    print(f"       gold citations sit at median rank {w['gold_rank_median']}, "
          f"90th percentile {w['gold_rank_p90']}, "
          f"and {w['pct_gold_beyond_rank_30']}% fall beyond rank 30")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
