"""Task 3 deep dive: could the Task 4 committee have won Task 3?

Answers the conference question ("why not apply the ensemble to Task 3 and
win?") with a decomposition. The post-competition nine-expert vote on Task 3
read the top-5 articles of the submitted DU2 retrieval run. We compute, per
question: whether the top-5 context contained every gold article
(sufficiency@5), the vote's correctness under oracle and retrieved input,
and the attribution of the vote's errors and of its oracle-to-retrieved
flips to (a) missing gold articles vs (b) entailment failure despite a
sufficient context. Also places the ensemble's 86.6% in the official
leaderboard excerpt (ranks 1, 2, 6 known: .951, .902, .829).

Gate: vote on retrieved input must reproduce 71/82; oracle vote 77/82.
Writes task3_gap_numbers.json.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from selection_policy_analysis import load_gold, build_pool, DU3_EXPERTS
from deltaq_ci_analysis import RETR_FILE, RETR

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
IR_FILE = ROOT / ("TASK33/FINAL_SENT_TO_COMMITTEE_TASK3_20260306/"
                  "05_INPUTS_CACHES_ARTIFACTS/task3-IR.DU2")
TESTJ = ROOT / "TASK4/experiments/datasets/test_R07/test.jsonl"
OUT = HERE / "task3_gap_numbers.json"


def main():
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    du3 = [pool.index(e) for e in DU3_EXPERTS]

    # gold articles per question
    gold_articles = {}
    for line in open(TESTJ, encoding="utf-8"):
        d = json.loads(line)
        gold_articles[d["id"]] = set(d["articles"])

    # top-5 retrieved articles per question (TREC format, DU2 run)
    retrieved = defaultdict(list)
    with open(IR_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and len(retrieved[parts[0]]) < 5:
                retrieved[parts[0]].append(parts[2])
    sufficient = {q: gold_articles[q] <= set(retrieved[q])
                  for q in test_ids}

    # committee correctness under both conditions
    oracle_vote = c_test[du3].sum(0) > 4.5
    retr_correct = []
    for e in DU3_EXPERTS:
        pr = {}
        for line in open(RETR / RETR_FILE[e]):
            d = json.loads(line)
            pr[d["id"]] = d["label"]
        retr_correct.append([pr.get(q) == gold[q] for q in test_ids])
    retr_vote = np.array(retr_correct).sum(0) > 4.5

    if not (int(oracle_vote.sum()) == 77 and int(retr_vote.sum()) == 71):
        print("GATE FAILED", int(oracle_vote.sum()), int(retr_vote.sum()))
        sys.exit(1)
    print("gate oracle 77 / retrieved 71: PASS")

    suff = np.array([sufficient[q] for q in test_ids])
    n_suff = int(suff.sum())

    errors = ~retr_vote
    err_suff = int((errors & suff).sum())
    err_insuff = int((errors & ~suff).sum())
    flips = oracle_vote & ~retr_vote
    flip_suff = int((flips & suff).sum())
    flip_insuff = int((flips & ~suff).sum())

    acc_on_suff = float(retr_vote[suff].mean())
    results = {
        "context": "nine-expert vote on top-5 articles of the submitted DU2 "
                    "retrieval run (the post-competition configuration)",
        "sufficiency_at_5": {
            "questions_with_all_gold_in_top5": n_suff,
            "of": 82,
            "rate_pc": round(n_suff / 82 * 100, 1),
        },
        "vote_retrieved_errors_decomposed": {
            "total_errors": int(errors.sum()),
            "errors_with_sufficient_context": err_suff,
            "errors_with_missing_gold": err_insuff,
        },
        "oracle_to_retrieved_flips": {
            "total_flips_right_to_wrong": int(flips.sum()),
            "flips_with_sufficient_context": flip_suff,
            "flips_with_missing_gold": flip_insuff,
            "flip_ids": [test_ids[i] for i in np.where(flips)[0]],
        },
        "accuracy_on_sufficient_context_pc": round(acc_on_suff * 100, 1),
        "ceiling_with_this_retrieval_pc": round(
            (n_suff + int(retr_vote[~suff].sum())) / 82 * 100, 1),
        "leaderboard_placement": {
            "ensemble_pc": 86.6,
            "official_top5": {"1_NOWJ_run1": 95.1, "2_JNLP1": 90.2,
                              "3_NOWJ_run2": 89.0, "4_NOWJ_run3": 89.0,
                              "5_codyrag": 85.4, "14_DU2_ours": 79.3},
            "placement": "exactly four official runs score above 86.6, so "
                         "the ensemble would have placed 5th of 23 runs, "
                         "not 1st (top-5 verified from the COLIEE 2026 "
                         "proceedings, NOWJ paper Table 4)",
        },
        "gap_to_winner_questions": int(78 - retr_vote.sum()),
    }

    print(json.dumps(results, indent=2))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
