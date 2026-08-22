"""Scores and paired intervals for the 2025 replication (expQ).

Reads the raw predictions of run_replication_2025.py, recomputes both arms' micro P, R
and F1 from the files (never trusting the runner's own printout), and attaches a paired
case-level bootstrap to the difference, exactly as expP does for the 2026 experiments.

Verification gates, all of which must pass before anything is written:
  1. Both arms cover all 71 untouched cases with no permanent-failure gaps.
  2. Every predicted paragraph id belongs to its case's BM25 top-20 shortlist.
  3. No evaluated case appears in the prompt-development or stress splits.

Context the reading requires: the untouched subset averages 1.72 gold paragraphs per
case, between the development split's 1.22 and the 2026 test's 2.94, and the paper's
mechanism account therefore predicts an effect between zero and the 13.0 points measured
on the 2026 test. The prediction was registered in the runner's docstring before the
run. The result is reported whichever way it lands.

Writes expQ_numbers.json.
"""
import csv
import json
from collections import defaultdict
import os
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

import numpy as np

HERE = Path(__file__).parent
RUN_DIR = Path(ROOT) / "runs_replication_2025"
LABELS = Path(ROOT) / "task2_test_labels_2025.json"
UNTOUCHED = HERE / "replication2025_untouched_ids.json"
LTR = Path(ROOT) / "ltr_data"
OUT = HERE / "expQ_numbers.json"
B = 10_000
SEED = 20260820


def load_arm(arm):
    preds = defaultdict(set)
    covered = set()
    for line in (RUN_DIR / f"test2025_{arm}.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            covered.add(parts[0])
            if parts[1] != "__none__":
                preds[parts[0]].add(parts[1].zfill(3))
    return preds, covered


def micro(preds, gold, cases):
    tp = pn = gn = 0
    for c in cases:
        p, g = preds.get(c, set()), gold[c]
        tp += len(p & g); pn += len(p); gn += len(g)
    if pn == 0 or gn == 0 or tp == 0:
        return 0.0, 0.0, 0.0
    P, R = tp / pn, tp / gn
    return P, R, 2 * P * R / (P + R)


def main():
    cases = sorted(json.load(open(UNTOUCHED)), key=int)
    raw = json.load(open(LABELS))
    gold = {}
    for c in cases:
        v = raw[c]
        paras = v if isinstance(v, list) else [x for x in str(v).split(",") if x.strip()]
        gold[c] = {p.strip().replace(".txt", "").zfill(3) for p in paras}
    shortlists = json.load(open(RUN_DIR / "bm25_top20_2025.json"))

    touched = set()
    for name in ["dev_ltr_features.csv", "devstress_ltr_features.csv"]:
        with open(LTR / name) as fh:
            r = csv.DictReader(fh)
            key = [k for k in r.fieldnames
                   if "case" in k.lower() or "query" in k.lower() or k.lower() == "qid"][0]
            for row in r:
                touched.add(row[key])

    arms, gates = {}, {}
    for arm in ("control", "treatment"):
        preds, covered = load_arm(arm)
        arms[arm] = preds
        gates[f"{arm}_covers_all_71"] = covered >= set(cases)
        gates[f"{arm}_preds_within_shortlist"] = all(
            p in set(shortlists[c][:20]) for c, ps in preds.items() for p in ps)
    gates["no_case_prompt_touched"] = not (set(cases) & touched)
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    full = {}
    for arm, preds in arms.items():
        P, R, F = micro(preds, gold, cases)
        full[arm] = {"P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
                     "avg_returned": round(float(np.mean(
                         [len(preds.get(c, set())) for c in cases])), 2)}

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(cases))
    diffs = np.empty(B)
    for t in range(B):
        s = [cases[i] for i in rng.choice(idx, size=len(cases), replace=True)]
        diffs[t] = micro(arms["treatment"], gold, s)[2] - micro(arms["control"], gold, s)[2]
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    res = {
        "experiment": "expQ_replication_2025",
        "design": ("frozen expO prompt pair on the COLIEE 2025 test collection, 71 cases "
                   "untouched by prompt development, BM25 top-20 candidates, BM25 worked "
                   "examples from a bank purged of all 2025 cases; no trained component"),
        "collection": {"n_cases": len(cases), "gold_total": sum(len(g) for g in gold.values()),
                       "gold_mean": round(sum(len(g) for g in gold.values()) / len(cases), 2),
                       "bm25_top20_recall": 0.828},
        "registered_prediction": ("intermediate effect, between zero and the 13.0 points "
                                  "of the 2026 test, since the gold mean 1.72 sits between "
                                  "the development split's 1.22 and the 2026 test's 2.94"),
        "gates": gates,
        "arms": full,
        "paired_difference": {
            "diff_pp": round(100 * (full["treatment"]["F1"] - full["control"]["F1"]), 2),
            "boot_ci95_pp": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "prob_positive": round(float((diffs > 0).mean()), 4),
            "B": B, "seed": SEED,
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    for arm, v in full.items():
        print(f"  {arm:10s} P={v['P']:.4f} R={v['R']:.4f} F1={v['F1']:.4f} "
              f"avg={v['avg_returned']}")
    d = res["paired_difference"]
    print(f"  paired diff {d['diff_pp']:+.2f}pp  95% CI [{d['boot_ci95_pp'][0]:+.2f}, "
          f"{d['boot_ci95_pp'][1]:+.2f}]  P(>0) {d['prob_positive']:.3f}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
