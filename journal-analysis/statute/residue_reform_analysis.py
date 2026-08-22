"""The residue and statutory reform (Section 6 of the manuscript).

Defines the residue as the questions on which at most a quarter of the
30-expert pool (at most 7 of 30) is correct, across all 340 labelled
questions, and characterises it: size, distribution over examination years,
the questions missed by every expert, and the gold articles involved. The
2020 Civil Code reform enters through specific residue questions keyed to
pre-reform law (Art. 604 lease term; the pre-reform sub-agency liability
provisions).

Gate: reproduces pool and DU3 before computing. Writes
residue_reform_numbers.json.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

from selection_policy_analysis import (load_gold, build_pool,
                                       committee_correct, DU3_EXPERTS)

HERE = Path(__file__).parent
OUT = HERE / "residue_reform_numbers.json"
ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
DATA = ROOT / "TASK4/experiments/datasets"


def main():
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    du3 = np.array([[pool.index(e) for e in DU3_EXPERTS]])
    if not (len(pool) == 30
            and int(committee_correct(c_test, du3)[0].sum()) == 77):
        print("GATE FAILED")
        sys.exit(1)
    print("gate: PASS")

    all_ids = val_ids + test_ids
    C = np.concatenate([c_val, c_test], axis=1)
    counts = C.sum(0)

    articles = {}
    for split in ("H30_formal", "R01_formal", "R02_formal", "test_R07"):
        f = DATA / split / ("test.jsonl")
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            articles[d["id"]] = d["articles"]

    def year(qid):
        y = qid.split("-")[0]
        return "R01" if y in ("R01", "R1") else y

    residue_idx = [i for i in range(len(all_ids)) if counts[i] <= 7]
    residue = [{"id": all_ids[i], "year": year(all_ids[i]),
                "correct_of_30": int(counts[i]),
                "gold_articles": articles[all_ids[i]]}
               for i in residue_idx]
    by_year = {}
    for r in residue:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1

    year_base = {}
    for q in all_ids:
        year_base[year(q)] = year_base.get(year(q), 0) + 1

    results = {
        "definition": "residue = questions with at most 7 of 30 experts "
                      "correct, over all 340 labelled questions",
        "residue_size": len(residue),
        "by_year": by_year,
        "base_counts": year_base,
        "r01_share_pc": round(by_year.get("R01", 0) / len(residue) * 100),
        "r01_base_rate_pc": round(year_base["R01"] / 340 * 100),
        "zero_correct": [r for r in residue if r["correct_of_30"] == 0],
        "residue": residue,
        "reform_notes": {
            "R01-25-I": "Art. 604: pre-reform maximum lease term 20 years; "
                        "the 2017 amendment (effective April 2020) raised it "
                        "to 50. Gold answer keyed to pre-reform law; 0 of 30 "
                        "experts recover it.",
            "R01-3-E": "keyed to the pre-reform sub-agency provisions; the "
                       "reform deleted the former agent-liability rule and "
                       "reorganised the surrounding articles; 2 of 30 "
                       "experts recover the pre-reform answer.",
            "R07": "the 2025 test examination postdates the reform, so the "
                   "current Code and the answer key agree; the test-side "
                   "conclusions of the paper are unaffected.",
        },
    }
    print(json.dumps(results, indent=2)[:1500])
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
