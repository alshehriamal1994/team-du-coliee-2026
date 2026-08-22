"""Provenance and verification for the supervised fine-tuning baselines (journal
article, Discussion).

Establishes where each figure in the fine-tuning table comes from and recomputes, from
the retained prediction logs, each figure that can be recomputed. Recomputed values are
checked against the recorded per-year accuracies converted to the 258-question micro
average, and the script aborts rather than writing if any check fails. Runs whose
surviving records describe conflicting protocols are documented rather than tabulated.

Writes finetune_baselines_numbers.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

import selection_policy_analysis as base

OUT = Path(__file__).resolve().parent / "finetune_baselines_numbers.json"
QLORA_TRAIN = base.ROOT / "TASK4/ncc_qlora_package/train_data.jsonl"

VAL_SIZES = {"H30": 67, "R01": 110, "R02": 81}
N_VAL = sum(VAL_SIZES.values())

# per-year accuracies as recorded in DETAILED_REPORT.md, used as the
# independent cross-check on the recomputed figures
RECORDED = {
    "deberta-noleak_v1": {"H30": 55.2, "R01": 66.4, "R02": 69.1},
    "qwen7-temporal-noleak_v1": {"H30": 74.6, "R01": 70.9, "R02": 76.5},
}
QWEN32B_RECORDED = {"H30": 50.7, "R01": 48.4, "R02": 59.3}


def micro(per_year):
    return sum(per_year[s] * n for s, n in VAL_SIZES.items()) / N_VAL


def main():
    gold = base.load_gold()
    val_ids = sorted(q for q in gold if not q.startswith("R07"))

    # ---- recompute the two runs whose predictions were retained ------------
    recomputed = {}
    for name, rec in RECORDED.items():
        preds = {}
        for split in base.ALL_SPLITS:
            p = base.load_predictions(name, split)
            if p:
                preds.update(p)
        covered = [q for q in val_ids if q in preds]
        if len(covered) != N_VAL:
            print(f"FAIL {name}: covers {len(covered)}/{N_VAL} val questions")
            sys.exit(1)
        n_correct = int(sum(preds[q] == gold[q] for q in covered))
        acc = 100.0 * n_correct / N_VAL
        expected = micro(rec)
        ok = abs(acc - expected) < 0.1
        print(f"  {name:28s} logs {acc:5.1f}%  records->micro {expected:5.1f}%"
              f"  {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("PROVENANCE CHECK FAILED - ledger not written.")
            sys.exit(1)
        recomputed[name] = {
            "n_correct": n_correct, "n_questions": N_VAL,
            "acc_pc": round(acc, 1),
            "recorded_per_year_pc": rec,
            "recorded_micro_pc": round(expected, 1),
            "recomputed_from_logs": True,
        }

    # ---- majority-class floor and pool reference ---------------------------
    pool, c_val, c_test, _, _ = base.build_pool(gold)
    val_acc = c_val.mean(axis=1) * 100
    best_i = int(val_acc.argmax())
    du3_idx = np.array([[pool.index(e) for e in base.DU3_EXPERTS]])
    du3_val = int(base.committee_correct(c_val, du3_idx)[0].sum())

    n_y = sum(gold[q] == "Y" for q in val_ids)
    floor = 100.0 * max(n_y, N_VAL - n_y) / N_VAL

    # ---- QLoRA training-set size, verified against the file ----------------
    qlora_n = sum(1 for _ in open(QLORA_TRAIN))

    out = {
        "purpose": "provenance and verification for Table tab:finetune",
        "benchmark": {"n_questions": N_VAL, "per_year": VAL_SIZES,
                      "majority_class_floor_pc": round(floor, 1)},
        "finetuned": {
            "deberta_v2_large_japanese": {
                "params": "324M", "method": "full cross-encoder fine-tune",
                "train_examples": 965,
                "settings": {"epochs": 15, "batch": 4, "grad_accum": 4,
                             "lr": 1e-5, "schedule": "cosine with warmup",
                             "early_stopping_patience": 3},
                **recomputed["deberta-noleak_v1"],
            },
            "qwen2.5_7b_instruct_qlora": {
                "params": "7B",
                "method": "QLoRA 4-bit, r=16, alpha=32, attention-only",
                "train_examples": 965,
                "settings": {"epochs": 5, "batch": 1, "grad_accum": 16,
                             "lr": 2e-4},
                **recomputed["qwen7-temporal-noleak_v1"],
            },
            "qwen2.5_32b_instruct_qlora": {
                "params": "32B",
                "method": "QLoRA 4-bit NF4, r=64, alpha=128, all projections",
                "train_examples": qlora_n,
                "train_examples_source": str(QLORA_TRAIN),
                "train_composition": {"task4": 965, "task3": qlora_n - 965},
                "settings": {"steps": 640, "hardware": "1x A100-80GB",
                             "wall_clock_h": 8.3},
                "recorded_per_year_pc": QWEN32B_RECORDED,
                "acc_pc": round(sum(QWEN32B_RECORDED.values()) / 3, 1),
                "acc_basis": "mean of recorded per-year accuracies",
                "recomputed_from_logs": False,
                "note": "prediction file not retained; 77 of 82 test "
                        "predictions on a single label per DETAILED_REPORT 5.3.3",
            },
        },
        "not_tabulated": {
            "elyza_jp_8b_lora": {
                "pooled_correct_per_year": [53, 84, 62],
                "acc_pc": round(100.0 * (53 + 84 + 62) / N_VAL, 1),
                "source": "TASK4/experiments/RESULTS.md (per-year adapters)",
                "conflicting_record": "DETAILED_REPORT.md 5.3.4 reports "
                                      "62.7/59.1/65.4 for a full-data adapter",
            },
            "llmjp_3_13b_lora": {
                "pooled_correct_per_year": [46, 68, 63],
                "acc_pc": round(100.0 * (46 + 68 + 63) / N_VAL, 1),
                "source": "TASK4/experiments/RESULTS.md (per-year adapters)",
            },
            "reason": "two surviving records describe different protocols; "
                      "the manuscript quotes only the higher pooled value",
        },
        "zero_shot_reference": {
            "best_expert": pool[best_i],
            "best_expert_val_acc_pc": round(float(val_acc[best_i]), 1),
            "pool_size": len(pool),
            "du3_vote_val": f"{du3_val}/{N_VAL}",
            "du3_vote_val_acc_pc": round(100.0 * du3_val / N_VAL, 1),
        },
    }

    print(f"  majority-class floor {floor:.1f}%  "
          f"best zero-shot {val_acc[best_i]:.1f}%  "
          f"DU3 vote {100.0 * du3_val / N_VAL:.1f}%")
    print(f"  QLoRA train file rows: {qlora_n} "
          f"(965 Task 4 + {qlora_n - 965} Task 3)")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
