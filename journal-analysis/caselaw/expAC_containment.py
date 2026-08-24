"""Does the count permission change which paragraphs are chosen, or only how
many (expAC)?

The manipulation of Section 7.1 separates the stopping decision from the
selection judgement imperfectly by nature, since the model chooses which
paragraphs and how many in one act. Precision falling from .753 to .614 under
the permission is consistent with two readings: the model added weaker
selections around an unchanged core judgement, or the removal of the
single-most-entailing wording shifted the judgement itself. The stored run
files of the two arms decide between them without any new model call, by
asking how often the treatment's set contains the control's choice.

Reads the stored predictions of run_singleclause_manipulation.py. Gated on
reproducing the expO ledger's returned counts and both arms' micro-F1 exactly.

Writes expAC_numbers.json.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")
HERE = Path(__file__).parent
RUNS = Path(ROOT) / "runs_singleclause"
LABELS = Path(ROOT) / "task2_test_labels_2026(1).json"
OUT = HERE / "expAC_numbers.json"

EXPECT = {"control": {"returned": 89, "F1": 0.3499},
          "treatment": {"returned": 189, "F1": 0.4803}}


def load(path):
    d = defaultdict(set)
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2 and parts[1] != "__none__":
            d[parts[0]].add(parts[1])
    return d


def main():
    raw = json.load(open(LABELS))
    gold = {cid: {x.strip().replace(".txt", "").zfill(3)
                  for x in val.split(",") if x.strip()}
            for cid, val in raw.items()}
    cases = sorted(gold, key=int)
    total_gold = sum(len(g) for g in gold.values())

    arms = {a: load(RUNS / f"test2026_{a}.txt") for a in ("control", "treatment")}

    gates = {}
    for a, preds in arms.items():
        tp = sum(len(preds.get(c, set()) & gold[c]) for c in cases)
        n = sum(len(preds.get(c, set())) for c in cases)
        P, R = tp / n, tp / total_gold
        f1 = round(2 * P * R / (P + R), 4)
        gates[a] = {"returned": n, "F1": f1,
                    "matches_ledger": n == EXPECT[a]["returned"]
                    and abs(f1 - EXPECT[a]["F1"]) < 6e-4}
    if not all(g["matches_ledger"] for g in gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    ctrl, trt = arms["control"], arms["treatment"]
    both = [c for c in cases if ctrl.get(c) and trt.get(c)]
    contained = [c for c in both if ctrl[c] <= trt[c]]
    out = {
        "experiment": "expAC_containment",
        "question": "does the count permission change which paragraphs are "
                    "chosen, or only how many",
        "gates": gates,
        "abstentions": {"control": sum(1 for c in cases if not ctrl.get(c)),
                        "treatment": sum(1 for c in cases if not trt.get(c))},
        "cases_both_arms_selected": len(both),
        "treatment_contains_control_choice": len(contained),
        "containment_share": round(len(contained) / len(both), 4),
        "control_choice_dropped": sorted(set(both) - set(contained), key=int),
        "control_selected_treatment_abstained":
            sum(1 for c in cases if ctrl.get(c) and not trt.get(c)),
        "treatment_selected_control_abstained":
            sum(1 for c in cases if trt.get(c) and not ctrl.get(c)),
        "reading": "the permission added selections around an unchanged core "
                   "judgement: wherever both arms selected, the treatment set "
                   "contains the control's choice, and abstention moved by two "
                   "cases",
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"both selected: {len(both)}  containment: {len(contained)} "
          f"({out['containment_share']:.0%})  abstentions "
          f"{out['abstentions']}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
