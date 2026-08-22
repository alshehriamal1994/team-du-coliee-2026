"""Four-year trend of the Task 2 answer-count distribution and its ceilings (expU).

Pure label arithmetic, no model and no inference. For each COLIEE Task 2 test collection
from 2023 to 2026: the gold count distribution, the perfect-ranker micro-F1 ceiling at
each fixed answer count k, the best constant k*, and the ceiling of single selection.

Provenance of the labels. The 2025 and 2026 test collections have official label files
on disk. The 2023 and 2024 test collections are recovered from the archived training
releases, which COLIEE builds append-only: each year's training labels are the previous
year's training labels plus the previous year's test block, so the 2025 training release
(cases 1 to 825) ends in the 2024 test block (726 to 825), which in turn follows the
2023 test block (626 to 725). The append-only structure is verified where an official
file exists to check it against, which is the gate below.

Verification gates, before anything is written:
  1. The 2025 test block (cases 826 to 925) inside the 2026 training labels must
     reproduce the official 2025 test label file exactly, case for case.
  2. The training-release sizes must chain: 925 = 825 + 100 and 825 = 725 + 100.
  3. The computed 2026 single-selection ceiling must equal the paper's 0.508.

Writes expU_numbers.json.
"""
import json
import os
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

HERE = Path(__file__).parent
OUT = HERE / "expU_numbers.json"
P25TR = Path(ROOT) / "task2_train_labels_2025.json"
P26TR = Path(ROOT) / "task2_train_labels_2026.json"
P25TE = Path(ROOT) / "task2_test_labels_2025.json"
P26TE = Path(ROOT) / "task2_test_labels_2026(1).json"


def parse(v):
    items = v if isinstance(v, list) else str(v).split(",")
    return sorted(x.strip() for x in items if str(x).strip())


def block(labels, lo, hi):
    return {k: parse(v) for k, v in labels.items() if k.isdigit() and lo <= int(k) <= hi}


def ceiling_stats(gold):
    counts = [len(v) for v in gold.values()]
    n, tot = len(counts), sum(counts)
    f1 = {}
    for k in range(1, 8):
        tp = sum(min(k, c) for c in counts)
        P, R = tp / (k * n), tp / tot
        f1[k] = round(2 * P * R / (P + R), 4)
    kstar = max(f1, key=f1.get)
    return {
        "n_cases": n, "gold_total": tot, "gold_mean": round(tot / n, 2),
        "share_multi_answer": round(sum(1 for c in counts if c > 1) / n, 2),
        "ceiling_by_k": f1, "best_constant": kstar,
        "ceiling_at_best_constant": f1[kstar],
        "ceiling_single_selection": f1[1],
    }


def main():
    tr25 = json.load(open(P25TR))
    tr26 = json.load(open(P26TR))
    te25 = {k: parse(v) for k, v in json.load(open(P25TE)).items()}
    te26 = {k: parse(v) for k, v in json.load(open(P26TE)).items()}

    blk25 = block(tr26, 826, 925)
    gates = {
        "2025_block_reproduces_official_file": blk25 == te25,
        "release_sizes_chain": len(tr26) == 925 and len(tr25) == 825,
    }
    years = {
        "2023": block(tr25, 626, 725),
        "2024": block(tr25, 726, 825),
        "2025": te25,
        "2026": te26,
    }
    stats = {y: ceiling_stats(g) for y, g in years.items()}
    gates["2026_single_ceiling_is_0.508"] = abs(stats["2026"]["ceiling_single_selection"] - 0.5077) < 6e-4
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    res = {
        "experiment": "expU_ceiling_trend",
        "method": ("pure label arithmetic over the four Task 2 test collections; 2023 "
                   "and 2024 recovered from the append-only training releases by ID "
                   "block, the recovery method validated by gate 1"),
        "gates": gates,
        "years": stats,
        "reading": ("The gold mean rises monotonically across the four years, the best "
                    "constant drifts from one to three, and the ceiling of single "
                    "selection falls from 0.909 to 0.508. Single selection was "
                    "ceiling-optimal in 2023 and 2024. The trend is in the drift of "
                    "the optimal constant, not in irreducibility: each year's own best "
                    "constant keeps a high ceiling."),
    }
    OUT.write_text(json.dumps(res, indent=2))
    for y, s in stats.items():
        print(f"  {y}: mean {s['gold_mean']:.2f}  multi {s['share_multi_answer']:.0%}  "
              f"k* {s['best_constant']}  ceil(k*) {s['ceiling_at_best_constant']:.3f}  "
              f"ceil(1) {s['ceiling_single_selection']:.3f}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
