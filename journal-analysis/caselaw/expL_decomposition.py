"""Decomposition of the micro-F1 gap into four components (journal article, Section 6).

Within a single reranker version, the distance from each submitted policy to a perfect
score is divided into the gain available from a better fixed count, the further gain from
the true per-query count, ranking quality within the evaluated list, and gold never
retrieved into it. Perfect selection from the list, the boundary between the last two,
returns exactly the gold present, so its precision is one and its recall is the list's
recall. It sizes the component and is not an achievable system.

The script reproduces the published anchors from the ledgers beside it before reporting
anything, and writes expL_decomposition_numbers.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import os

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))

HERE = Path(__file__).resolve().parent
GOLD1 = ROOT / "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"
DEEP1 = HERE / "expA2_step8_deep.json"
OUT = HERE / "expL_decomposition_numbers.json"


def norm(s):
    return s.replace(".txt", "")


def main():
    a2 = json.load(open(HERE / "expA2_numbers.json"))
    ec = json.load(open(HERE / "expC_numbers.json"))
    ef = json.load(open(HERE / "expF_numbers.json"))
    ei = json.load(open(HERE / "expI_numbers.json"))

    # ---- gate on the published anchors ------------------------------------
    checks = {
        "T1 fixed5 0.3456": abs(a2["fixed_k"]["k5"]["F1"] - 0.3456) < 5e-5,
        "T1 oracle 0.4143": abs(a2["oracle_count"]["F1"] - 0.4143) < 5e-5,
        "T2v2 k1 0.3503": abs(ec["fixed_topk"]["k1"]["F1"] - 0.3503) < 5e-5,
        "T2v2 k3 0.4646": abs(ec["fixed_topk"]["k3"]["F1"] - 0.4646) < 5e-5,
        "T2v2 oracle 0.4932": abs(ec["oracle_cardinality"]["F1"] - 0.4932) < 5e-5,
        "expC is v2": abs(ec["fixed_topk"]["k1"]["F1"] - ef["v2"]["fixed_k1"]["F1"]) < 5e-5,
    }
    for k, v in checks.items():
        print(f"  gate {k:22s} {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        print("GATE FAILED - nothing computed.")
        sys.exit(1)

    # ---- Task 1: recall of the evaluated list -----------------------------
    gold = {norm(k): set(norm(v) for v in vs) for k, vs in json.load(open(GOLD1)).items()}
    deep = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(DEEP1)).items()}
    qs = sorted(gold)
    total = sum(len(gold[q]) for q in qs)
    present = sum(len(set(deep.get(q, [])) & gold[q]) for q in qs)
    r1 = present / total
    perfect1 = 2 * r1 / (1 + r1)                    # P = 1, R = r1

    fk = a2["fixed_k"]
    best1 = max(fk, key=lambda x: fk[x]["F1"])
    deployed1, oracle1 = fk["k5"]["F1"], a2["oracle_count"]["F1"]

    # ---- Task 2, held at v2 throughout ------------------------------------
    tk = ec["fixed_topk"]
    best2 = max(tk, key=lambda x: tk[x]["F1"])
    deployed2 = tk["k1"]["F1"]                       # the at-most-one policy we deployed
    oracle2 = ec["oracle_cardinality"]["F1"]
    r2 = ef["v2"]["recall@20"]
    perfect2 = 2 * r2 / (1 + r2)

    def block(name, deployed, best, best_k, oracle, perfect, recall):
        return {
            "deployed_constant": round(deployed, 4),
            "best_constant": round(best, 4), "best_constant_k": best_k,
            "oracle_count": round(oracle, 4),
            "perfect_selection_from_list": round(perfect, 4),
            "list_recall": round(recall, 4),
            "components_pp": {
                "1_better_constant": round((best - deployed) * 100, 1),
                "2_per_query_adaptivity": round((oracle - best) * 100, 1),
                "3_ranking_within_list": round((perfect - oracle) * 100, 1),
                "4_gold_never_retrieved": round((1.0 - perfect) * 100, 1),
            },
        }

    res = {
        "protocol": {
            "purpose": "version-matched decomposition of the micro-F1 gap, separating "
                       "the choice of a constant from per-query adaptivity",
            "version_note": "Task 2 is held at the monoT5 v2 reranker throughout, so "
                            "no part of a reranker upgrade is charged to the count "
                            "decision.",
            "perfect_selection": "returns exactly the gold present in the evaluated "
                                 "list, precision 1, recall equal to the list's recall. "
                                 "An upper bound used to size the component, not a system.",
        },
        "task1": block("T1", deployed1, fk[best1]["F1"], best1, oracle1, perfect1, r1),
        "task2_v2": block("T2", deployed2, tk[best2]["F1"], best2, oracle2, perfect2, r2),
        "version_matched_count_value_task2": {
            "v1": round((ef["v1"]["oracle_count"]["F1"]
                         - ef["v1"]["fixed_k1"]["F1"]) * 100, 1),
            "v2": round((ef["v2"]["oracle_count"]["F1"]
                         - ef["v2"]["fixed_k1"]["F1"]) * 100, 1),
            "cross_version_figure_not_used": round((oracle2
                                                  - ei["reference_points"]["official_DU3"]) * 100, 1),
        },
    }

    for t in ("task1", "task2_v2"):
        b = res[t]["components_pp"]
        print(f"\n  {t}: best constant k={res[t]['best_constant_k']}, "
              f"list recall {res[t]['list_recall']}")
        print(f"    better constant      {b['1_better_constant']:+6.1f}")
        print(f"    per-query adaptivity {b['2_per_query_adaptivity']:+6.1f}")
        print(f"    ranking in the list  {b['3_ranking_within_list']:+6.1f}")
        print(f"    never retrieved      {b['4_gold_never_retrieved']:+6.1f}")
    print(f"\n  Task 2 count value, version-matched: "
          f"v1 {res['version_matched_count_value_task2']['v1']:+.1f}, "
          f"v2 {res['version_matched_count_value_task2']['v2']:+.1f}  "
          f"(cross-version figure, not used: "
          f"{res['version_matched_count_value_task2']['cross_version_figure_not_used']:+.1f})")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"written: {OUT.name}")


if __name__ == "__main__":
    main()
