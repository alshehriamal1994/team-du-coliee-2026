"""Does the selection failure hold when a different examination year is the target?

Sections 5.1 and 5.3 select on the 258 validation questions and deploy on R07. A reader
may reasonably ask whether the failure is specific to that pairing, since R07 is one year
and the validation set is three particular years. The pool is scored on four independent
examination years in total, so the question can be answered without any new inference by
rotating which year plays the part of the deployment target.

For each held-out year Y the procedure is the one Section 5 studies, applied unchanged:
rank the thirty experts on the questions of the remaining years, deploy the highest-ranked
single expert, and record what it scores on Y. Three quantities are reported per rotation,
matching the main text. The deployed single's accuracy on Y, the accuracy of drawing an
expert uniformly at random, which needs no selection data at all, and the accuracy of an
unweighted majority vote over all thirty, which also needs no selection. The rank the
deployed expert achieves among the thirty on Y states how far the choice missed.

The R07 rotation reproduces the main text and acts as the verification gate: selecting on
H30, R01 and R02 must place the deployed expert 28th of thirty on R07 and score 82.9%,
otherwise nothing is written.

This is a replication of the selection procedure across deployment targets, not an
independent dataset. All four years come from the same competition and the same Civil
Code, so it bounds the within-benchmark generality of the finding and no more.

Writes cross_year_selection_numbers.json.
"""
import json
from pathlib import Path

import numpy as np

import selection_policy_analysis as base

HERE = Path(__file__).parent
OUT = HERE / "cross_year_selection_numbers.json"


def year_of(qid):
    prefix = qid.split("-")[0]
    return "R01" if prefix in ("R01", "R1") else prefix


def main():
    gold = base.load_gold()
    names, val, test, vids, tids = base.build_pool(gold)

    # one matrix over all four years, experts by questions
    allc = np.concatenate([val, test], axis=1)
    allq = list(vids) + list(tids)
    years = np.array([year_of(q) for q in allq])
    order = ["H30", "R01", "R02", "R07"]

    rows = {}
    for held in order:
        mask = years == held
        train, target = allc[:, ~mask], allc[:, mask]
        sel = int(np.argmax(train.mean(1)))                 # the deployed single
        acc = float(target[sel].mean() * 100)
        per_expert = target.mean(1) * 100
        rank = int((per_expert > acc).sum() + 1)
        vote = float((target.sum(0) > len(names) / 2).mean() * 100)
        rows[held] = {
            "n_questions": int(mask.sum()),
            "selected_on": [y for y in order if y != held],
            "deployed_expert": names[sel],
            "deployed_accuracy_pc": round(acc, 1),
            "deployed_rank_of_30": rank,
            "random_draw_accuracy_pc": round(float(per_expert.mean()), 1),
            "selection_minus_random_pp": round(acc - float(per_expert.mean()), 1),
            "full_pool_vote_accuracy_pc": round(vote, 1),
            "vote_minus_deployed_pp": round(vote - acc, 1),
            "best_expert_on_target_pc": round(float(per_expert.max()), 1),
        }

    r07 = rows["R07"]
    gates = {
        "r07_rotation_reproduces_rank_28": r07["deployed_rank_of_30"] == 28,
        "r07_rotation_reproduces_82.9": abs(r07["deployed_accuracy_pc"] - 82.9) < 0.06,
        "four_years_present": len(rows) == 4,
        "question_total_is_340": sum(v["n_questions"] for v in rows.values()) == 340,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}\n{json.dumps(rows, indent=2)}")

    beats = sum(1 for v in rows.values() if v["vote_minus_deployed_pp"] > 0)
    worse_than_random = sum(1 for v in rows.values() if v["selection_minus_random_pp"] < 0)
    res = {
        "purpose": "does the selection failure survive rotating the deployment target?",
        "method": ("leave-one-examination-year-out: rank the thirty experts on the other "
                   "three years, deploy the best, score it on the held-out year; no new "
                   "inference, the pool is already scored on all four years"),
        "gates": gates,
        "rotations": rows,
        "summary": {
            "rotations": len(rows),
            "rotations_where_vote_beats_deployed_single": beats,
            "rotations_where_selection_worse_than_random": worse_than_random,
            "mean_selection_minus_random_pp":
                round(float(np.mean([v["selection_minus_random_pp"] for v in rows.values()])), 1),
            "mean_vote_minus_deployed_pp":
                round(float(np.mean([v["vote_minus_deployed_pp"] for v in rows.values()])), 1),
            "deployed_rank_range":
                [min(v["deployed_rank_of_30"] for v in rows.values()),
                 max(v["deployed_rank_of_30"] for v in rows.values())],
        },
        "scope": ("All four years come from the same competition and the same Civil Code, "
                  "so this bounds the within-benchmark generality of the finding and does "
                  "not establish it on other jurisdictions or tasks."),
    }
    OUT.write_text(json.dumps(res, indent=2))

    print(f"  {'held out':8s} {'n':>4s} {'deployed':>9s} {'rank':>6s} {'random':>7s} "
          f"{'vs rand':>8s} {'vote':>6s} {'vote-dep':>9s}")
    for y in order:
        v = rows[y]
        print(f"  {y:8s} {v['n_questions']:4d} {v['deployed_accuracy_pc']:8.1f}% "
              f"{v['deployed_rank_of_30']:4d}/30 {v['random_draw_accuracy_pc']:6.1f}% "
              f"{v['selection_minus_random_pp']:+7.1f} {v['full_pool_vote_accuracy_pc']:5.1f}% "
              f"{v['vote_minus_deployed_pp']:+8.1f}")
    s = res["summary"]
    print(f"\n  vote beats the deployed single in {beats} of 4 rotations; "
          f"selection trails a random draw in {worse_than_random} of 4")
    print(f"  mean selection minus random {s['mean_selection_minus_random_pp']:+.1f} pp, "
          f"mean vote minus deployed {s['mean_vote_minus_deployed_pp']:+.1f} pp")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
