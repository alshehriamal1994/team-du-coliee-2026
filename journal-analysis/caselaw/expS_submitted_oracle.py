"""The submitted ranker's own oracle-count score (expS).

Section 6.3 separates ranking and stopping because the oracle figure of 0.414 belongs to
the post-competition ranker, the submitted run surviving only as its top-5 output. The
saved booster of the submitted DU3 configuration turns out to exist
(ARCHIVE/runs/du3_bigmodel/model.txt), so the separation can be closed from the other
side: rebuild the submitted ranker's full ranking, verify it reproduces the official
submission, and supply the true count to the ranking we actually submitted.

Verification gates, before anything is written:
  1. The rebuilt pipeline's top five, under the original step8 postprocess, must score
     exactly the official 0.3141.
  2. Per-query, the rebuilt top five must match the official DU3.txt sets; the match
     fraction is reported and must be at least 0.99.

Then, on the depth-30 filtered ranking of that same booster: the oracle-count score,
its query-level bootstrap interval, and the paired delta against the submitted fixed
five, using the identical design as expR.

Writes expS_numbers.json.
"""
import json
import subprocess
import sys
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
ARCHIVE = ROOT / "TASK1/ARCHIVE"
MODEL = ARCHIVE / "runs/du3_bigmodel/model.txt"
FINAL = ROOT / "TASK1/FINAL_SUBMISSION"
STEP8 = ARCHIVE / "code_AUTHORITY_v2/step8_postprocess_filters_v2.py"
CORPUS = ARCHIVE / "task_one_ready_to_use/data/task1_test_files_2026/task1_test_files_2026"
CACHE = ARCHIVE / "cache_2026/test_cache_dotxt.pkl"
OUT = HERE / "expS_numbers.json"
B = 10_000
SEED = 20260822


def _jsonable(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not serialisable: {type(o)}")


def norm(s):
    return s.replace(".txt", "")


def main():
    gold = {norm(k): {norm(v) for v in vs}
            for k, vs in json.load(open(FINAL / "task1_test_labels_2026.json")).items()}
    qlist = sorted(gold)
    total = sum(len(g) for g in gold.values())

    def ev(preds):
        tp = sum(len(set(p) & gold[q]) for q, p in preds.items() if q in gold)
        n = sum(len(p) for p in preds.values())
        P, R = (tp / n if n else 0.0), tp / total
        return round(2 * P * R / (P + R), 4) if P + R else 0.0

    booster = lgb.Booster(model_file=str(MODEL))
    d = np.load(ARCHIVE / "cache_features/feature_matrix_test2026.npz", allow_pickle=True)
    X, qids, cids = d["X"].astype(np.float32), d["qids"].tolist(), d["cids"].tolist()
    scores = booster.predict(X)
    per_query = {}
    for s, q, c in zip(scores, qids, cids):
        per_query.setdefault(q, []).append((c, float(s)))
    ranked = {q: [norm(c) for c, _ in sorted(p, key=lambda x: x[1], reverse=True)]
              for q, p in per_query.items()}

    # The submitted runs fed step 8 the ranker's top five, so a filtered candidate
    # was refilled from the stage-one fusion rather than from further down the
    # learned ranking. Reproducing the submission therefore requires depth 5; the
    # oracle needs a deeper list, which is a separate call.
    def write_base(depth, name):
        bp = HERE / name
        bp.write_text(json.dumps(
            {q + ".txt": [c + ".txt" for c in r[:depth]] for q, r in ranked.items()}))
        return bp

    def run_step8(base_path, out_k, out_name):
        s8 = HERE / out_name
        subprocess.run([sys.executable, str(STEP8), "--corpus", str(CORPUS),
                        "--cache", str(CACHE), "--base_preds", str(base_path),
                        "--out", str(s8),
                        "--out_k", str(out_k), "--rrf_k", "5", "--remove_query_cases",
                        "--filter_future"], check=True, capture_output=True, text=True)
        return {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(s8)).items()}

    top5 = run_step8(write_base(5, "expS_base_top5.json"), 5, "expS_step8_top5.json")
    deep = run_step8(write_base(100, "expS_base_deep.json"), 30, "expS_step8_deep.json")

    du3 = {}
    for line in open(FINAL / "DU3.txt"):
        parts = line.split()
        du3.setdefault(norm(parts[0]), set()).add(norm(parts[1]))

    f1_top5 = ev({q: r[:5] for q, r in top5.items()})
    match = np.mean([set(top5.get(q, [])[:5]) == du3.get(q, set()) for q in qlist])
    gates = {
        "rebuilt_top5_scores_official_0.3141": bool(abs(f1_top5 - 0.3141) < 2e-4),
        "per_query_exact_match_ge_0.99": bool(match >= 0.99),
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates} "
                         f"(F1 {f1_top5}, match {match:.4f})")

    fixed5 = {q: set(deep.get(q, [])[:5]) for q in qlist}
    oracle = {q: set(deep.get(q, [])[:len(gold[q])]) for q in qlist}
    f1_fixed5, f1_oracle = ev(fixed5), ev(oracle)

    def arrs(preds):
        return (np.array([len(preds[q] & gold[q]) for q in qlist]),
                np.array([len(preds[q]) for q in qlist]),
                np.array([len(gold[q]) for q in qlist]))

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(qlist), size=(B, len(qlist)))

    def boot(preds):
        tp, npred, ngold = arrs(preds)
        TP, NP, NG = tp[idx].sum(1), npred[idx].sum(1), ngold[idx].sum(1)
        P, R = TP / np.maximum(NP, 1), TP / np.maximum(NG, 1)
        return 2 * P * R / np.maximum(P + R, 1e-12)

    b5, bo = boot(fixed5), boot(oracle)
    diff = bo - b5
    lo, hi = np.percentile(diff, [2.5, 97.5])
    olo, ohi = np.percentile(bo, [2.5, 97.5])

    res = {
        "experiment": "expS_submitted_oracle",
        "model": "saved submitted DU3 booster (ARCHIVE/runs/du3_bigmodel/model.txt)",
        "gates": {**gates, "per_query_exact_match_fraction": round(float(match), 4)},
        "protocol": {"B": B, "seed": SEED, "n_queries": len(qlist)},
        "submitted_fixed5_F1": f1_fixed5,
        "submitted_oracle_F1": f1_oracle,
        "submitted_oracle_ci95": [round(float(olo), 4), round(float(ohi), 4)],
        "stopping_on_submitted_pp": {
            "point": round(100 * (f1_oracle - f1_fixed5), 2),
            "ci95": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "prob_positive": round(float((diff > 0).mean()), 4),
        },
        "references": {"official_DU3": 0.3141, "winner": 0.4220,
                       "du9_oracle": 0.4143, "du9_fixed5": 0.3456},
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  gate: rebuilt top5 F1 {f1_top5}, per-query exact match {match:.4f}")
    print(f"  submitted ranker fixed-5 (depth-30 list): {f1_fixed5}")
    print(f"  SUBMITTED RANKER ORACLE: {f1_oracle}  CI {res['submitted_oracle_ci95']}")
    s = res["stopping_on_submitted_pp"]
    print(f"  stopping on the submitted system: +{s['point']}pp CI {s['ci95']} "
          f"P(>0) {s['prob_positive']}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
