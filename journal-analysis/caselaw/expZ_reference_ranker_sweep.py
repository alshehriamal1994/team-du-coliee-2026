"""The post-competition Task 1 configuration sweep behind the reference ranker.

Every post-competition Task 1 figure in the article is computed on one ranker,
which the article calls DU9. That configuration was not chosen in advance: six
configurations were trained after the competition and scored against the released
2026 test labels, and DU9 was the best of them. This script records the sweep so
that the selection disclosed in the article can be checked, and reports the spread
against which the effects attributed to the answer count should be read.

Reads the two training summaries under COLIEE_ROOT and writes expZ_numbers.json.
"""

import json
import os
import sys
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")
HERE = Path(__file__).resolve().parent
SUMMARIES = {
    "du4_bigger": Path(ROOT) / "TASK1/runs/du4_bigger/summary.json",
    "du7_tuning": Path(ROOT) / "TASK1/runs/du7_tuning/summary.json",
}
OUT = HERE / "expZ_numbers.json"

DEPLOYED = "DU9"
EXPECT_DEPLOYED_F1 = 0.3456


def main():
    scores, configs = {}, {}
    for group, path in SUMMARIES.items():
        if not path.exists():
            sys.exit(f"Set COLIEE_ROOT to the working tree; expected {path}")
        for name, rec in json.load(open(path)).items():
            if not isinstance(rec, dict) or "step8" not in rec:
                continue
            scores[name] = round(float(rec["step8"]["micro_f1"]), 4)
            cfg = rec.get("cfg", {})
            configs[name] = {k: cfg[k] for k in sorted(cfg) if k != "note"}

    if DEPLOYED not in scores:
        sys.exit(f"GATE FAILED: {DEPLOYED} absent from the summaries")
    if abs(scores[DEPLOYED] - EXPECT_DEPLOYED_F1) > 5e-5:
        sys.exit(f"GATE FAILED: {DEPLOYED} scores {scores[DEPLOYED]}, "
                 f"expected {EXPECT_DEPLOYED_F1}")
    if max(scores, key=scores.get) != DEPLOYED:
        sys.exit(f"GATE FAILED: the best configuration is "
                 f"{max(scores, key=scores.get)}, not {DEPLOYED}")

    lo, hi = min(scores.values()), max(scores.values())
    out = {
        "question": "how was the post-competition Task 1 reference ranker chosen, "
                    "and how wide is the field it was chosen from",
        "protocol": {
            "selection": "argmax of step-8 micro-F1 on the released 2026 test "
                         "labels; the configurations differ in capacity, learning "
                         "rate, sampling and regularisation only",
            "caveat": "selection on the evaluation collection. The ranker is a "
                      "reference held fixed across the stopping analyses, not a "
                      "system that could have been submitted.",
        },
        "n_configurations": len(scores),
        "scores": dict(sorted(scores.items())),
        "selected": DEPLOYED,
        "spread": {"min": lo, "max": hi, "range_pp": round(100 * (hi - lo), 2)},
        "configurations": configs,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    for name in sorted(scores):
        mark = "  <- selected" if name == DEPLOYED else ""
        print(f"  {name}: {scores[name]:.4f}{mark}")
    print(f"{len(scores)} configurations, spread {lo:.4f} to {hi:.4f} "
          f"({100 * (hi - lo):.1f} points)")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
