"""Case-level bootstrap for the leakage-free Task 2 figures (journal article, Section 8).

Resamples the 100 test cases with replacement, B = 10,000, recomputing micro-averaged
precision, recall and F1 per replicate. The unit of resampling is the case, cases being
the independent observations, and the denominator is all 100 cases, since a case for
which a run selects nothing still contributes its gold to recall. Point estimates must
reproduce the expI ledger exactly before any interval is computed.

Writes expJ_bootstrap_numbers.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRED = Path.home() / "Desktop/TASK2_code/runs_multiselect_noprior"
GOLD = Path.home() / "Desktop/TASK2_code/task2_test_labels_2026(1).json"
LEDGER_I = HERE / "expI_numbers.json"
OUT = HERE / "expJ_bootstrap_numbers.json"

B_BOOT = 10_000
SEED = 20260810
WINNER = 0.4899          # official Task 2 winner, IAI
BEST_FIXED = 0.4646      # best fixed cutoff, k = 3

RUNS = {
    "v3_multiselect_rag": "test2026_v3_multiselect_rag.txt",
    "v3_multiselect_zero": "test2026_v3_multiselect_zero.txt",
    "r1_multiselect_rag": "test2026_r1_multiselect_rag.txt",
}


def load_gold():
    raw = json.load(open(GOLD))
    return {cid: {x.strip().replace(".txt", "").zfill(3)
                  for x in val.split(",") if x.strip()}
            for cid, val in raw.items()}


def load_preds(path):
    d = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]].add(parts[1].replace(".txt", "").zfill(3))
    return d


def micro_f1(cases, preds, gold):
    tp = sum(len(preds.get(c, set()) & gold[c]) for c in cases)
    npred = sum(len(preds.get(c, set())) for c in cases)
    ngold = sum(len(gold[c]) for c in cases)
    if npred == 0 or ngold == 0:
        return 0.0, 0.0, 0.0
    p, r = tp / npred, tp / ngold
    return p, r, (0.0 if p + r == 0 else 2 * p * r / (p + r))


def main():
    gold_all = load_gold()
    runs = {k: load_preds(PRED / v) for k, v in RUNS.items()}
    # all 100 gold cases, including any for which a run selected nothing:
    # such a case contributes no true positives but its gold still counts to recall
    cases = sorted(gold_all, key=int)
    gold = gold_all
    print(f"  {len(cases)} test cases, {sum(len(v) for v in gold.values())} gold paragraphs")

    # majority vote over the three runs, 2 of 3
    vote = {}
    for c in cases:
        cnt = defaultdict(int)
        for d in runs.values():
            for p in d.get(c, set()):
                cnt[p] += 1
        vote[c] = {p for p, n in cnt.items() if n >= 2}
    runs["majority_vote_2of3"] = vote

    # ---- gate against the expI ledger ------------------------------------
    led = json.load(open(LEDGER_I))["prior_removed"]
    ok = True
    point = {}
    for name, d in runs.items():
        p, r, f = micro_f1(cases, d, gold)
        point[name] = (p, r, f)
        want = led[name]["F1"]
        good = abs(round(f, 4) - want) < 5e-4
        ok &= good
        print(f"  gate {name:22s} F1 {f:.4f}  ledger {want:.4f}  "
              f"{'PASS' if good else 'FAIL'}")
    if not ok:
        print("GATE FAILED - no intervals computed.")
        sys.exit(1)

    # ---- case-level bootstrap --------------------------------------------
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(cases))
    draws = rng.integers(0, len(cases), size=(B_BOOT, len(cases)))
    out = {
        "protocol": {
            "resampling_unit": "test case",
            "n_cases": len(cases), "B": B_BOOT, "seed": SEED,
            "reference_winner_IAI": WINNER,
            "reference_best_fixed_cutoff": BEST_FIXED,
        },
        "runs": {},
    }
    for name, d in runs.items():
        vals = np.empty(B_BOOT)
        for b in range(B_BOOT):
            sel = [cases[i] for i in draws[b]]
            vals[b] = micro_f1(sel, d, gold)[2]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out["runs"][name] = {
            "F1": round(point[name][2], 4),
            "precision": round(point[name][0], 4),
            "recall": round(point[name][1], 4),
            "boot_se": round(float(vals.std()), 4),
            "boot_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ci_contains_winner": bool(lo <= WINNER <= hi),
            "prob_exceeds_winner": round(float((vals > WINNER).mean()), 3),
            "prob_exceeds_best_fixed": round(float((vals > BEST_FIXED).mean()), 3),
        }
        v = out["runs"][name]
        print(f"  {name:22s} F1 {v['F1']:.4f}  SE {v['boot_se']:.4f}  "
              f"95% [{v['boot_ci95'][0]:.4f}, {v['boot_ci95'][1]:.4f}]  "
              f"contains winner: {v['ci_contains_winner']}  "
              f"P(>winner) {v['prob_exceeds_winner']:.2f}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
