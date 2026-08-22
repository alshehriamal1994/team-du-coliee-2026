"""Bootstrap CI for the oracle-vs-retrieved Q-statistic null (Section 6).

Same nine experts, same 82 R07 questions; oracle (gold-article) predictions
from TASK4/experiments/runs_ensemble/R07.*, retrieved-input predictions from
the post-competition Task 3 rerun. Reproduces mean pairwise Q of 0.900
(oracle) and 0.893 (retrieved) as a gate, then bootstraps the paired
difference over questions.
"""

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

from selection_policy_analysis import load_gold, load_predictions, DU3_EXPERTS

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
RETR = ROOT / "TASK33/Post_Competition_Analysis_Task4_Ensemble_on_Task3/expert_predictions"
OUT = Path(__file__).parent / "deltaq_ci_numbers.json"

RETR_FILE = {
    "deepseek-r1-zs": "DSR1.jsonl",
    "llama-4-maverick_standard_v1": "L4-M.jsonl",
    "llama-4-scout_standard_v1": "L4-S.jsonl",
    "llama-3.3-70b-instruct_concise_v1": "L-con.jsonl",
    "llama-3.3-70b-instruct_cot_strict_v1": "L-cot.jsonl",
    "qwen-2.5-72b-instruct_cot_strict_v1": "Q2-cot.jsonl",
    "qwen3-235b-a22b_standard_v1": "Q3.jsonl",
    "qwen3-235b-a22b_irac_v1": "Q3-irac.jsonl",
    "qwen3-235b-a22b_sc3_v1": "Q3-sc3.jsonl",
}


def mean_q(correct):
    """Mean pairwise Q-statistic over expert rows (bool matrix E x N)."""
    qs = []
    for a, b in combinations(range(correct.shape[0]), 2):
        n11 = float(np.sum(correct[a] & correct[b]))
        n00 = float(np.sum(~correct[a] & ~correct[b]))
        n10 = float(np.sum(correct[a] & ~correct[b]))
        n01 = float(np.sum(~correct[a] & correct[b]))
        num = n11 * n00 - n01 * n10
        den = n11 * n00 + n01 * n10
        if den > 0:
            qs.append(num / den)
    return float(np.mean(qs)) if qs else float("nan")


def main():
    gold = load_gold()
    test_ids = sorted(q for q in gold if q.startswith("R07"))

    oracle = []
    retrieved = []
    for e in DU3_EXPERTS:
        po = load_predictions(e, "R07")
        oracle.append([po[q] == gold[q] for q in test_ids])
        pr = {}
        with open(RETR / RETR_FILE[e]) as f:
            for line in f:
                d = json.loads(line)
                pr[d["id"]] = d["label"]
        retrieved.append([pr[q] == gold[q] for q in test_ids])
    oracle = np.array(oracle)
    retrieved = np.array(retrieved)

    q_o = mean_q(oracle)
    q_r = mean_q(retrieved)
    print(f"oracle meanQ={q_o:.3f}  retrieved meanQ={q_r:.3f}  "
          f"dQ={q_r - q_o:+.3f}")
    assert round(q_o, 3) == 0.900 and round(q_r, 3) == 0.893, "gate failed"
    print("gate meanQ 0.900/0.893: PASS")

    rng = np.random.default_rng(20260704)
    n = len(test_ids)
    dqs = []
    for _ in range(10_000):
        idx = rng.integers(0, n, size=n)
        dqs.append(mean_q(retrieved[:, idx]) - mean_q(oracle[:, idx]))
    dqs = np.array(dqs)
    dqs = dqs[~np.isnan(dqs)]
    res = {
        "meanQ_oracle": round(q_o, 3),
        "meanQ_retrieved": round(q_r, 3),
        "deltaQ": round(q_r - q_o, 3),
        "deltaQ_95ci": [round(float(np.percentile(dqs, 2.5)), 3),
                        round(float(np.percentile(dqs, 97.5)), 3)],
        "n_bootstrap_valid": int(len(dqs)),
        "prob_deltaQ_ge_0.10": round(float((dqs >= 0.10).mean()), 4),
    }
    print(json.dumps(res, indent=2))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
