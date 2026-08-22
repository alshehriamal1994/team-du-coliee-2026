#!/usr/bin/env python3
"""
Majority-vote ensemble over multiple Task 4 prediction files.

Usage:
  python3 scripts/ensemble_predictions.py \\
    --inputs experiments/runs/task4-H30.RUN1 \\
             experiments/runs/task4-H30.RUN2 \\
             experiments/runs/task4-H30.RUN3 \\
    --output experiments/runs/task4-H30.ENS-V1 \\
    --run-tag ENS-V1 \\
    --tie-break Y          # if tie: default to Y (common in legal: assume entailed)

On ties (even number of inputs), --tie-break controls the fallback.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


LINE_RE = re.compile(r"^(\S+) ([YN]) (\S+)$")


def load_pred(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            m = LINE_RE.match(raw.strip())
            if m:
                preds[m.group(1)] = m.group(2)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Prediction files to ensemble")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--tie-break", choices=["Y", "N"], default="Y",
                        help="Label to use on ties (even number of inputs)")
    args = parser.parse_args()

    all_preds: list[dict[str, str]] = []
    for p in args.inputs:
        d = load_pred(p)
        print(f"  Loaded {len(d)} predictions from {p.name}")
        all_preds.append(d)

    # Union of all query IDs
    all_ids = set()
    for d in all_preds:
        all_ids |= set(d.keys())

    # Vote
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agreed = tied = missing = 0
    with args.output.open("w", encoding="utf-8") as out:
        for qid in sorted(all_ids):
            votes = [d[qid] for d in all_preds if qid in d]
            if not votes:
                missing += 1
                continue
            counts = Counter(votes)
            y_count = counts.get("Y", 0)
            n_count = counts.get("N", 0)
            if y_count > n_count:
                label = "Y"
                agreed += 1
            elif n_count > y_count:
                label = "N"
                agreed += 1
            else:
                label = args.tie_break
                tied += 1
            out.write(f"{qid} {label} {args.run_tag}\n")

    total = agreed + tied
    print(f"Ensemble complete: {total} predictions, {tied} ties (→ {args.tie_break})")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
