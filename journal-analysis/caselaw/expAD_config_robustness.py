"""Is the count's worth a property of DU9, or of the whole configuration
family (expAD)?

Section 5.2 discloses that DU9 was the best of six post-competition
configurations scored on the test labels, and Section 7.3 measures the count's
worth on DU9 alone. If the 6.9 points were an artefact of that selection, the
disclosure would be load-bearing. This experiment re-evaluates the stopping
decision on all six configurations from their stored predictions: each
configuration's top-50 ranking is passed through the deployed postprocessing at
depth 30, exactly the protocol behind the ledgered DU9 figures, and the fixed
five is compared with the true per-query count on each.

No model is run. Everything is re-evaluation of predictions stored at training
time. Gated on DU9 reproducing its ledgered fixed-five and oracle scores
exactly before the other five are computed.

Writes expAD_numbers.json.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")
HERE = Path(__file__).parent
RUNS = Path(ROOT) / "TASK1/runs"
STEP8 = Path(ROOT) / "TASK1/ARCHIVE/code_AUTHORITY_v2/step8_postprocess_filters_v2.py"
CORPUS = Path(ROOT) / ("TASK1/ARCHIVE/task_one_ready_to_use/data/"
                       "task1_test_files_2026/task1_test_files_2026")
CACHE = Path(ROOT) / "TASK1/ARCHIVE/cache_2026/test_cache_dotxt.pkl"
GOLD = Path(ROOT) / "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"
OUT = HERE / "expAD_numbers.json"

CONFIGS = {
    "DU4": RUNS / "du4_bigger/du4/preds_for_step8.json",
    "DU5": RUNS / "du4_bigger/du5/preds_for_step8.json",
    "DU6": RUNS / "du4_bigger/du6/preds_for_step8.json",
    "DU7": RUNS / "du7_tuning/du7/preds_for_step8.json",
    "DU8": RUNS / "du7_tuning/du8/preds_for_step8.json",
    "DU9": RUNS / "du7_tuning/du9/preds_for_step8.json",
}
EXPECT_DU9 = {"fixed5": 0.3456, "oracle": 0.4143}


def norm(s):
    return s.replace(".txt", "")


def main():
    gold_raw = json.load(open(GOLD))
    gold = {norm(k): {norm(x) for x in (v if isinstance(v, list)
                                       else str(v).split(","))}
            for k, v in gold_raw.items()}
    total_gold = sum(len(v) for v in gold.values())

    def micro(preds, k_of=None):
        tp = n = 0
        for q, lst in preds.items():
            kq = k_of(q) if k_of else 5
            sel = lst[:kq]
            n += len(sel)
            tp += len(set(sel) & gold.get(q, set()))
        P, R = tp / n, tp / total_gold
        return round(2 * P * R / (P + R), 4)

    results = {}
    with tempfile.TemporaryDirectory() as td:
        for name, pred_path in CONFIGS.items():
            if not pred_path.exists():
                sys.exit(f"missing predictions for {name}: {pred_path}")
            out_path = Path(td) / f"step8_{name}.json"
            subprocess.run([sys.executable, str(STEP8),
                            "--corpus", str(CORPUS), "--cache", str(CACHE),
                            "--base_preds", str(pred_path),
                            "--out", str(out_path),
                            "--out_k", "30", "--rrf_k", "5",
                            "--remove_query_cases", "--filter_future"],
                           check=True, capture_output=True, text=True)
            filt = {norm(k): [norm(v) for v in vs]
                    for k, vs in json.load(open(out_path)).items()}
            fixed5 = micro(filt)
            oracle = micro(filt, k_of=lambda q: len(gold.get(q, ())))
            results[name] = {"fixed5_F1": fixed5, "oracle_F1": oracle,
                             "delta_stop_pp": round(100 * (oracle - fixed5), 2)}
            print(f"  {name}: fixed5 {fixed5:.4f}  oracle {oracle:.4f}  "
                  f"delta {results[name]['delta_stop_pp']:.2f} pp")

    du9 = results["DU9"]
    gates = {
        "du9_fixed5_reproduces_ledger": abs(du9["fixed5_F1"]
                                            - EXPECT_DU9["fixed5"]) < 6e-4,
        "du9_oracle_reproduces_ledger": abs(du9["oracle_F1"]
                                            - EXPECT_DU9["oracle"]) < 6e-4,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}, {du9}")

    deltas = [r["delta_stop_pp"] for r in results.values()]
    out = {
        "experiment": "expAD_config_robustness",
        "question": "is the count's worth a property of the selected DU9 "
                    "configuration or of the whole family",
        "protocol": "each configuration's stored top-50 ranking through the "
                    "deployed postprocessing at depth 30, fixed five against "
                    "the true per-query count, identical to the DU9 ladder",
        "gates": gates,
        "per_configuration": results,
        "delta_stop_band_pp": {"min": min(deltas), "max": max(deltas),
                               "spread": round(max(deltas) - min(deltas), 2)},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\ndelta_stop across six configurations: "
          f"{min(deltas):.2f} to {max(deltas):.2f} pp "
          f"(spread {out['delta_stop_band_pp']['spread']:.2f})")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
