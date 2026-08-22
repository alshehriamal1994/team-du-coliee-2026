"""Check the reranker-variant ledger behind Table 5 of the journal article.

expF_numbers.json records the two monoT5 variants on the Task 2 test collection
with candidates and ensemble weights held fixed: v1 with negatives mined by BM25,
v2 with negatives re-mined from the pipeline's own top 20 and a widened input
window. The raw scores live with the licensed collection, so this script does not
recompute them. It checks that every precision, recall and F1 triple in the ledger
is internally consistent, and that the difference column printed in the article
follows from the two arms.

Writes nothing. Exits non-zero if any check fails.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "expF_numbers.json"

# The difference column as printed in the article's Table 5.
PRINTED = {
    "recall@10": 0.092,
    "recall@20": 0.065,
    "fixed_k1": 0.056,
    "fixed_k2": 0.077,
    "fixed_k3": 0.094,
    "oracle_count": 0.095,
}


def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def main():
    d = json.load(open(LEDGER))
    failures = []

    for arm in ("v1", "v2"):
        for key, cell in d[arm].items():
            if isinstance(cell, dict) and {"P", "R", "F1"} <= set(cell):
                got = f1(cell["P"], cell["R"])
                if abs(got - cell["F1"]) > 5e-4:
                    failures.append(
                        f"{arm}/{key}: F1 {cell['F1']} does not follow from "
                        f"P {cell['P']} and R {cell['R']} (computed {got:.4f})")

    for key, printed in PRINTED.items():
        a, b = d["v1"][key], d["v2"][key]
        lo = a["F1"] if isinstance(a, dict) else a
        hi = b["F1"] if isinstance(b, dict) else b
        if abs((hi - lo) - printed) > 5e-4:
            failures.append(
                f"{key}: printed difference {printed:+.3f} does not match "
                f"{hi:.4f} - {lo:.4f} = {hi - lo:+.4f}")

    if failures:
        print("FAILED")
        for f in failures:
            print("  " + f)
        sys.exit(1)

    print(f"expF ledger consistent: {len(PRINTED)} rows, "
          "every P/R/F1 triple and every printed difference checks out")
    print("note: raw scores are not recomputed here, the collection being licensed")


if __name__ == "__main__":
    main()
