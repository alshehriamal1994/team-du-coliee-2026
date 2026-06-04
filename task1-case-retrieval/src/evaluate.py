#!/usr/bin/env python3
"""
Comprehensive evaluation script for COLIEE 2026 Task 1 submissions.

Evaluates DU1, DU2, DU3 against gold labels and produces:
  - Micro-averaged F1, Precision, Recall
  - Per-query F1 distribution (mean, median, std)
  - Error analysis (zero-F1 queries, difficulty breakdown)
  - Pairwise comparison between runs
  - Ablation-ready evaluation for arbitrary prediction files

Usage:
    python evaluate.py                          # Evaluate DU1/DU2/DU3
    python evaluate.py --pred /path/to/pred.txt # Evaluate a custom prediction file
    python evaluate.py --error-analysis         # Full error analysis on DU3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
GOLD_PATH = BASE_DIR / "FINAL_SUBMISSION" / "task1_test_labels_2026.json"
SUBMISSION_DIR = BASE_DIR / "FINAL_SUBMISSION"


def normalise_id(case_id: str) -> str:
    """Strip .txt suffix for consistent matching."""
    return case_id.replace(".txt", "")


# ──────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────
def load_gold(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        raw = json.load(f)
    return {
        normalise_id(k): [normalise_id(v) for v in vs]
        for k, vs in raw.items()
    }


def load_submission(path: Path) -> dict[str, list[str]]:
    preds = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid = normalise_id(parts[0])
                cid = normalise_id(parts[1])
                preds[qid].append(cid)
    return dict(preds)


# ──────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────
def evaluate(preds: dict, gold: dict) -> dict:
    """Compute micro-averaged and per-query metrics."""
    total_tp = total_fp = total_fn = 0
    per_query = {}

    for qid, gold_cases in gold.items():
        pred_set = set(preds.get(qid, []))
        gold_set = set(gold_cases)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if tp > 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        else:
            p = r = f1 = 0.0

        per_query[qid] = {
            "f1": f1, "precision": p, "recall": r,
            "tp": tp, "fp": fp, "fn": fn,
            "n_gold": len(gold_set), "n_pred": len(pred_set),
        }

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0

    f1_arr = np.array([v["f1"] for v in per_query.values()])

    return {
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_query": per_query,
        "f1_mean": float(f1_arr.mean()),
        "f1_median": float(np.median(f1_arr)),
        "f1_std": float(f1_arr.std()),
        "zero_f1_count": int((f1_arr == 0).sum()),
        "perfect_f1_count": int((f1_arr == 1.0).sum()),
        "num_queries": len(gold),
    }


# ──────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────
def print_summary(name: str, result: dict):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Micro F1:      {result['micro_f1']:.4f}")
    print(f"  Precision:     {result['micro_precision']:.4f}")
    print(f"  Recall:        {result['micro_recall']:.4f}")
    print(f"  TP={result['total_tp']}  FP={result['total_fp']}  FN={result['total_fn']}")
    print(f"  Per-query F1:  mean={result['f1_mean']:.4f}  "
          f"median={result['f1_median']:.4f}  std={result['f1_std']:.4f}")
    print(f"  Queries F1=0:  {result['zero_f1_count']}/{result['num_queries']}")
    print(f"  Queries F1=1:  {result['perfect_f1_count']}/{result['num_queries']}")


def print_comparison_table(results: dict[str, dict]):
    """Side-by-side comparison of multiple runs."""
    names = list(results.keys())
    print(f"\n{'=' * 70}")
    print("  Comparison Table")
    print(f"{'=' * 70}")
    header = f"  {'Metric':<20}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("  " + "-" * (20 + 12 * len(names)))

    rows = [
        ("Micro F1", "micro_f1"),
        ("Precision", "micro_precision"),
        ("Recall", "micro_recall"),
        ("TP", "total_tp"),
        ("FP", "total_fp"),
        ("FN", "total_fn"),
        ("Mean F1", "f1_mean"),
        ("Median F1", "f1_median"),
        ("Std F1", "f1_std"),
        ("Queries F1=0", "zero_f1_count"),
        ("Queries F1=1", "perfect_f1_count"),
    ]

    for label, key in rows:
        vals = [results[n][key] for n in names]
        if isinstance(vals[0], int):
            row = f"  {label:<20}" + "".join(f"{v:>12d}" for v in vals)
        else:
            row = f"  {label:<20}" + "".join(f"{v:>12.4f}" for v in vals)
        # Mark the best value
        print(row)


def error_analysis(name: str, result: dict, gold: dict):
    """Detailed error analysis: what characterises failing queries?"""
    pq = result["per_query"]
    print(f"\n{'=' * 60}")
    print(f"  Error Analysis: {name}")
    print(f"{'=' * 60}")

    # Group queries by F1 range
    buckets = {"F1=0": [], "0<F1<0.25": [], "0.25<=F1<0.5": [],
               "0.5<=F1<0.75": [], "0.75<=F1<1": [], "F1=1": []}
    for qid, m in pq.items():
        f1 = m["f1"]
        if f1 == 0:
            buckets["F1=0"].append(qid)
        elif f1 < 0.25:
            buckets["0<F1<0.25"].append(qid)
        elif f1 < 0.5:
            buckets["0.25<=F1<0.5"].append(qid)
        elif f1 < 0.75:
            buckets["0.5<=F1<0.75"].append(qid)
        elif f1 < 1.0:
            buckets["0.75<=F1<1"].append(qid)
        else:
            buckets["F1=1"].append(qid)

    print("\n  F1 Distribution:")
    for bucket, qids in buckets.items():
        bar = "#" * (len(qids) // 2)
        print(f"    {bucket:<16} {len(qids):>4} queries  {bar}")

    # Analyse zero-F1 queries by number of gold citations
    zero_qids = buckets["F1=0"]
    if zero_qids:
        zero_gold_counts = [len(gold[q]) for q in zero_qids]
        print(f"\n  Zero-F1 queries ({len(zero_qids)}):")
        print(f"    Gold citations: mean={np.mean(zero_gold_counts):.1f}  "
              f"median={np.median(zero_gold_counts):.1f}  "
              f"min={min(zero_gold_counts)}  max={max(zero_gold_counts)}")

    # All queries: gold count vs performance
    all_gold_counts = [len(gold[q]) for q in gold]
    all_f1 = [pq[q]["f1"] for q in gold]

    # Correlation between number of gold citations and F1
    corr = np.corrcoef(all_gold_counts, all_f1)[0, 1]
    print(f"\n  Correlation (n_gold, F1): {corr:.3f}")

    # Performance by gold count bucket
    print("\n  Performance by number of gold citations:")
    gc_buckets = defaultdict(list)
    for qid in gold:
        n = len(gold[qid])
        if n <= 2:
            gc_buckets["1-2"].append(pq[qid]["f1"])
        elif n <= 5:
            gc_buckets["3-5"].append(pq[qid]["f1"])
        elif n <= 10:
            gc_buckets["6-10"].append(pq[qid]["f1"])
        else:
            gc_buckets["11+"].append(pq[qid]["f1"])

    for bucket in ["1-2", "3-5", "6-10", "11+"]:
        if bucket in gc_buckets:
            vals = gc_buckets[bucket]
            print(f"    {bucket:>5} gold:  {len(vals):>3} queries  "
                  f"mean F1={np.mean(vals):.4f}  zero-F1={sum(1 for v in vals if v==0)}")

    # Pairwise overlap: which queries does each run get right?
    print(f"\n  Queries with at least one TP: "
          f"{sum(1 for q in pq if pq[q]['tp'] > 0)}/{len(pq)}")


def pairwise_analysis(results: dict[str, dict]):
    """Show which queries one run gets right that another misses."""
    names = list(results.keys())
    print(f"\n{'=' * 60}")
    print("  Pairwise Run Differences")
    print(f"{'=' * 60}")

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            pq_a = results[a]["per_query"]
            pq_b = results[b]["per_query"]
            a_better = sum(1 for q in pq_a if pq_a[q]["f1"] > pq_b[q]["f1"])
            b_better = sum(1 for q in pq_a if pq_b[q]["f1"] > pq_a[q]["f1"])
            equal = sum(1 for q in pq_a if pq_a[q]["f1"] == pq_b[q]["f1"])
            a_tp_not_b = sum(1 for q in pq_a if pq_a[q]["tp"] > 0 and pq_b[q]["tp"] == 0)
            b_tp_not_a = sum(1 for q in pq_a if pq_b[q]["tp"] > 0 and pq_a[q]["tp"] == 0)

            print(f"\n  {a} vs {b}:")
            print(f"    {a} better: {a_better}  |  {b} better: {b_better}  |  equal: {equal}")
            print(f"    {a} has TP where {b} has 0: {a_tp_not_b}")
            print(f"    {b} has TP where {a} has 0: {b_tp_not_a}")

            # F1 difference distribution
            diffs = [pq_a[q]["f1"] - pq_b[q]["f1"] for q in pq_a]
            print(f"    F1 diff ({a}-{b}): mean={np.mean(diffs):.4f}  "
                  f"std={np.std(diffs):.4f}  "
                  f"max_gain={max(diffs):.4f}  max_loss={min(diffs):.4f}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="COLIEE 2026 Task 1 Evaluation")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH,
                        help="Path to gold labels JSON")
    parser.add_argument("--pred", type=Path, nargs="*",
                        help="Custom prediction file(s) to evaluate")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Run detailed error analysis on DU3 (or first --pred)")
    parser.add_argument("--json-out", type=Path,
                        help="Write results to JSON file")
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"ERROR: Gold labels not found at {args.gold}", file=sys.stderr)
        sys.exit(1)

    gold = load_gold(args.gold)
    print(f"Gold labels: {len(gold)} queries, "
          f"{sum(len(v) for v in gold.values())} total relevant cases")

    results = {}

    if args.pred:
        # Evaluate custom prediction files
        for p in args.pred:
            p = Path(p)
            if not p.exists():
                print(f"WARNING: {p} not found, skipping", file=sys.stderr)
                continue
            preds = load_submission(p)
            r = evaluate(preds, gold)
            name = p.stem
            results[name] = r
            print_summary(name, r)
    else:
        # Evaluate DU1, DU2, DU3
        for run in ["DU1", "DU2", "DU3"]:
            path = SUBMISSION_DIR / f"{run}.txt"
            if not path.exists():
                print(f"WARNING: {path} not found, skipping", file=sys.stderr)
                continue
            preds = load_submission(path)
            r = evaluate(preds, gold)
            results[run] = r
            print_summary(run, r)

    if len(results) > 1:
        print_comparison_table(results)
        pairwise_analysis(results)

    if args.error_analysis:
        # Run error analysis on the best run (DU3) or the first custom pred
        target = "DU3" if "DU3" in results else list(results.keys())[0]
        error_analysis(target, results[target], gold)

    if args.json_out:
        # Serialise (drop per_query for brevity)
        out = {}
        for name, r in results.items():
            out[name] = {k: v for k, v in r.items() if k != "per_query"}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
