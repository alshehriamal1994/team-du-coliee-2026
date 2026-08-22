"""Wilson score intervals for the Task 4 test accuracies.

The test split holds 82 questions, so a single question moves accuracy by 1.22 points and
the paper repeatedly declines to read small differences. This script attaches an interval
to each headline accuracy so a reader can see directly which contrasts the split can
resolve and which it cannot, rather than taking that on assertion.

The Wilson score interval is used rather than the normal approximation because it behaves
correctly near the boundary, which matters at 79 of 82. Intervals are two-sided at 95%.

The comparison the paper rests on and the comparison it declines to make are both reported:
the ensemble against the deployable single model, and the ensemble against the strongest
competing run.

Verification gate: the correct counts must reproduce the accuracies stated in the official
results table before anything is written.

Writes wilson_intervals_numbers.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "wilson_intervals_numbers.json"
N = 82
Z = 1.959963985

RUNS = {
    "du1_du2_submitted": (79, "DU1 and DU2, the submitted meta-ensemble and deliberation vote"),
    "best_competitor": (78, "the strongest competing runs, JNLP and IAI"),
    "du3_plain_vote": (77, "DU3, the plain nine-expert vote"),
    "validation_selected_single": (68, "the model a practitioner would have deployed"),
    "hindsight_best_single": (78, "the best single expert in hindsight, not identifiable in advance"),
}
STATED = {"du1_du2_submitted": 96.3, "best_competitor": 95.1, "du3_plain_vote": 93.9,
          "validation_selected_single": 82.9, "hindsight_best_single": 95.1}


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def main():
    res, gates = {}, {}
    for key, (k, desc) in RUNS.items():
        acc = 100 * k / N
        lo, hi = wilson(k, N)
        gates[f"{key}_accuracy_matches_table"] = abs(round(acc, 1) - STATED[key]) < 0.06
        res[key] = {
            "description": desc,
            "correct": k,
            "of": N,
            "accuracy_pc": round(acc, 1),
            "wilson95_pc": [round(lo, 1), round(hi, 1)],
        }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    def overlap(a, b):
        la, ha = res[a]["wilson95_pc"]
        lb, hb = res[b]["wilson95_pc"]
        return la < hb and lb < ha

    res["contrasts"] = {
        "ensemble_vs_best_competitor": {
            "intervals_overlap": overlap("du1_du2_submitted", "best_competitor"),
            "reading": ("The split cannot resolve this contrast, which is why the paper "
                        "reports the placement and declines to claim superiority."),
        },
        "ensemble_vs_deployable_single": {
            "intervals_overlap": overlap("du1_du2_submitted", "validation_selected_single"),
            "reading": ("The split does resolve this contrast, which is the comparison the "
                        "paper's argument rests on."),
        },
    }
    res["method"] = ("two-sided 95% Wilson score intervals on 82 test questions; Wilson "
                     "rather than the normal approximation because of behaviour near the "
                     "boundary at 79 of 82")
    res["gates"] = gates
    OUT.write_text(json.dumps(res, indent=2))

    for k in RUNS:
        r = res[k]
        print(f"  {k:28s} {r['correct']}/{N} = {r['accuracy_pc']:5.1f}%  "
              f"95% CI [{r['wilson95_pc'][0]}, {r['wilson95_pc'][1]}]")
    print()
    for k, v in res["contrasts"].items():
        print(f"  {k:34s} intervals overlap: {v['intervals_overlap']}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
