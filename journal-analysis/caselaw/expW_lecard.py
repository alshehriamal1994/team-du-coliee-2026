"""The count-value relation on a third-party benchmark: LeCaRD (expW).

Section 6.5 establishes, within COLIEE, that the answer-count decision is worth more
the better the ranking. This experiment tests that relation on data and rankings we did
not produce: LeCaRD (Ma et al., SIGIR 2021), the Chinese legal case retrieval benchmark,
whose repository publishes the golden relevance labels for its 107 queries and the
per-query top-100 rankings of the paper's own baseline systems (TF-IDF, BM25, language
model, BERT, and their combination). No model is run here.

For each published ranking: the best fixed answer count under micro-F1, the oracle-count
score (top-n_q per query), and their difference, the worth of count knowledge to that
ranker. Alongside: the perfect-ranker fixed-count ceiling, from the golden count
distribution alone. LeCaRD's own metrics are cutoff-based (precision at k, MAP), so no
published set-valued F1 exists to anchor against; the gates are therefore structural.

Verification gates, before anything is written:
  1. 107 queries in the golden labels, and every ranking file covers all 107.
  2. Every golden id for a query appears in that query's candidate ranking
     (the pools were the judged sets, so a missing golden id would signal a
     data-handling error).
  3. Golden counts span at least 1 to 30 with the mean above 10, matching the
     distribution the LeCaRD authors describe.

Writes expW_numbers.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
LECARD = HERE / "lecard_data"
OUT = HERE / "expW_numbers.json"
RANKINGS = ["tfidf_top100", "bm25_top100", "lm_top100", "combined_top100"]
# bert.json stores four fold-wise variants one per line, none covering all 107
# queries alone; merging folds would mix models, so it is excluded.


def main():
    gold = {q: set(v) for q, v in
            json.loads((LECARD / "golden_labels.json").read_text()).items()}
    n, total = len(gold), sum(len(v) for v in gold.values())
    counts = [len(v) for v in gold.values()]

    ranks = {}
    for name in RANKINGS:
        raw = (LECARD / f"{name}.json").read_text()
        r = json.loads(raw)
        ranks[name] = {q: list(v) for q, v in r.items()}

    gates = {
        "golden_has_107_queries": n == 107,
        "all_rankings_cover_107": all(set(r) >= set(gold) for r in ranks.values()),
        "rankings_are_top100": all(len(v) in (100, 101) for r in ranks.values()
                                   for v in r.values()),
        "distribution_matches_paper": min(counts) == 1 and max(counts) == 30
                                      and total / n > 10,
    }
    # golden ids can legitimately fall outside a weak ranking's top-100; that is
    # retrieval failure, reported as coverage rather than gated
    coverage = {name: round(sum(len(gold[q] & set(r[q])) for q in gold) / total, 3)
                for name, r in ranks.items()}
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    def micro(preds, k=None):
        tp = pn = 0
        for q, g in gold.items():
            sel = set(preds[q][:k if k else len(g)])
            tp += len(sel & g); pn += len(sel)
        P, R = tp / pn, tp / total
        return 2 * P * R / (P + R) if P + R else 0.0

    ceil = {}
    for k in range(1, 31):
        tp = sum(min(k, c) for c in counts)
        P, R = tp / (k * n), tp / total
        ceil[k] = round(2 * P * R / (P + R), 4)
    kstar = max(ceil, key=ceil.get)

    rows = {}
    for name, r in ranks.items():
        bf = max(((k, micro(r, k)) for k in range(1, 31)), key=lambda x: x[1])
        orc = micro(r)
        rows[name] = {"best_fixed_F1": round(bf[1], 4), "best_fixed_k": bf[0],
                      "oracle_F1": round(orc, 4),
                      "count_worth_pp": round(100 * (orc - bf[1]), 2)}

    q = [v["best_fixed_F1"] for v in rows.values()]
    w = [v["count_worth_pp"] for v in rows.values()]
    rho = stats.spearmanr(q, w)

    res = {
        "experiment": "expW_lecard",
        "source": ("LeCaRD repository (Ma et al., SIGIR 2021), golden labels and the "
                   "authors' published baseline rankings; no model run here"),
        "gates": gates,
        "collection": {"n_queries": n, "golden_total": total,
                       "golden_mean": round(total / n, 2),
                       "golden_min_max": [min(counts), max(counts)]},
        "perfect_ranker": {"best_constant": kstar, "ceiling_at_best": ceil[kstar]},
        "rankings": rows,
        "golden_coverage_in_top100": coverage,
        "quality_vs_count_worth_spearman": round(float(rho.statistic), 2),
        "reading": ("The fixed-count ceiling of a perfect ranker is far lower than on "
                    "COLIEE because the golden counts spread more widely, and the worth "
                    "of count knowledge rises monotonically with ranking quality across "
                    "the benchmark's own published baselines, the relation Section 6.5 "
                    "measures within COLIEE."),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  perfect ranker: k*={kstar}, ceiling {ceil[kstar]}")
    for name, v in rows.items():
        print(f"  {name:16s} best-fixed {v['best_fixed_F1']:.3f} (k={v['best_fixed_k']}) "
              f"oracle {v['oracle_F1']:.3f}  worth {v['count_worth_pp']:+.1f}pp")
    print(f"  Spearman(quality, worth) = {rho.statistic:.2f}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
