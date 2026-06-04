#!/usr/bin/env python3
"""
DU7-DU9: Fine-tuning around DU4's sweet spot.
DU4 (F1=0.3451) is the best so far. These runs explore whether
different regularisation or subsampling can squeeze out more.

Expected time: ~15 min each, ~45 min total.
"""

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np

# ── Paths ──
ARCHIVE = Path("./ARCHIVE")
CACHE_DIR = ARCHIVE / "cache_features"
STEP8_SCRIPT = ARCHIVE / "code_AUTHORITY_v2/step8_postprocess_filters_v2.py"
CORPUS_2026 = ARCHIVE / "task_one_ready_to_use/data/task1_test_files_2026/task1_test_files_2026"
STAGE1_CACHE = ARCHIVE / "cache_2026/test_cache_dotxt.pkl"
GOLD_PATH = Path("./FINAL_SUBMISSION/task1_test_labels_2026.json")
OUT_DIR = Path("./runs/du7_tuning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configs ──
# DU4 baseline: 4000 trees, 255 leaves, LR=0.02, subsample=0.8, colsample=0.8, reg=0.1/0.1
CONFIGS = {
    "DU7": {
        "n_estimators": 4000, "num_leaves": 255, "learning_rate": 0.02,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.5, "reg_lambda": 0.5,
        "note": "DU4 + stronger regularisation + less sampling"
    },
    "DU8": {
        "n_estimators": 5000, "num_leaves": 255, "learning_rate": 0.015,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "note": "DU4 + more rounds at lower LR (converge slower)"
    },
    "DU9": {
        "n_estimators": 4000, "num_leaves": 255, "learning_rate": 0.02,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_alpha": 0.05, "reg_lambda": 0.05,
        "note": "DU4 + less regularisation + more data per tree"
    },
}

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["labels"].astype(np.int8),
            d["qids"].tolist(), d["cids"].tolist(), d["feature_names"].tolist())

def norm_id(s):
    return s.replace(".txt", "")

def evaluate_against_gold(preds, gold_path):
    with open(gold_path) as f:
        gold_raw = json.load(f)
    gold = {norm_id(k): [norm_id(v) for v in vs] for k, vs in gold_raw.items()}
    tp = fp = fn = 0
    per_query_f1 = []
    for qid, gc in gold.items():
        pc = set(preds.get(qid, []))
        gs = set(gc)
        t = len(pc & gs)
        tp += t; fp += len(pc - gs); fn += len(gs - pc)
        if t > 0:
            p = t / (t + len(pc - gs)); r = t / (t + len(gs - pc))
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        per_query_f1.append(f1)
    micro_p = tp / (tp + fp) if tp + fp else 0
    micro_r = tp / (tp + fn) if tp + fn else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0
    arr = np.array(per_query_f1)
    return {"micro_f1": micro_f1, "precision": micro_p, "recall": micro_r,
            "tp": tp, "fp": fp, "fn": fn, "zero_f1": int((arr == 0).sum())}

# ── Load data ──
print("Loading feature matrices...")
t0 = time.time()
X1, y1, q1, c1, fnames = load_npz(CACHE_DIR / "feature_matrix_train2025.npz")
X2, y2, q2, c2, _ = load_npz(CACHE_DIR / "feature_matrix_test2025.npz")
X3, y3, q3, c3, _ = load_npz(CACHE_DIR / "feature_matrix_train2026_clean.npz")
Xt, yt, qt, ct, _ = load_npz(CACHE_DIR / "feature_matrix_test2026.npz")

X_train = np.vstack([X1, X2, X3])
y_train = np.concatenate([y1, y2, y3])
q_train = q1 + q2 + q3
mask = y_train >= 0
X_train, y_train = X_train[mask], y_train[mask]
q_filtered = [q for q, m in zip(q_train, mask) if m]
qid_order = list(dict.fromkeys(q_filtered))
counts = Counter(q_filtered)
groups = [counts[q] for q in qid_order]
print(f"  Loaded in {time.time()-t0:.1f}s")
print()

results = {}

for run_name, cfg in CONFIGS.items():
    print(f"{'='*60}")
    print(f"  {run_name}: {cfg['note']}")
    print(f"  {cfg['n_estimators']} trees, {cfg['num_leaves']} leaves, LR={cfg['learning_rate']}")
    print(f"  subsample={cfg['subsample']}, colsample={cfg['colsample_bytree']}, reg={cfg['reg_alpha']}/{cfg['reg_lambda']}")
    print(f"{'='*60}")

    run_dir = OUT_DIR / run_name.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
        "num_leaves": cfg["num_leaves"], "learning_rate": cfg["learning_rate"],
        "min_child_samples": 20,
        "subsample": cfg["subsample"], "colsample_bytree": cfg["colsample_bytree"],
        "reg_alpha": cfg["reg_alpha"], "reg_lambda": cfg["reg_lambda"],
        "verbose": -1, "seed": 42,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, group=groups,
                            feature_name=fnames, free_raw_data=False)
    t1 = time.time()
    model = lgb.train(params, lgb_train, num_boost_round=cfg["n_estimators"],
                      valid_sets=[lgb_train], callbacks=[lgb.log_evaluation(period=500)])
    train_time = time.time() - t1
    print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    model.save_model(str(run_dir / "model.txt"))

    # Predict
    scores = model.predict(Xt)
    per_query = {}
    for score, qid, cid in zip(scores, qt, ct):
        per_query.setdefault(qid, []).append((cid, float(score)))

    top5_raw = {qid: [norm_id(c) for c, _ in
                      sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
                for qid, pairs in per_query.items()}

    raw_metrics = evaluate_against_gold(top5_raw, GOLD_PATH)
    print(f"  Raw:    F1={raw_metrics['micro_f1']:.4f}  P={raw_metrics['precision']:.4f}  R={raw_metrics['recall']:.4f}  zero-F1={raw_metrics['zero_f1']}")

    # Step8
    preds_dotxt = {k + ".txt": [v + ".txt" for v in vs] for k, vs in
                   {qid: [norm_id(c) for c, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:50]]
                    for qid, pairs in per_query.items()}.items()}
    preds_dotxt_path = run_dir / "preds_for_step8.json"
    with open(preds_dotxt_path, "w") as f:
        json.dump(preds_dotxt, f, indent=2)

    step8_output = run_dir / "step8_output.json"
    try:
        subprocess.run([sys.executable, str(STEP8_SCRIPT),
            "--corpus", str(CORPUS_2026), "--cache", str(STAGE1_CACHE),
            "--base_preds", str(preds_dotxt_path), "--out", str(step8_output),
            "--out_k", "5", "--rrf_k", "5", "--remove_query_cases", "--filter_future",
        ], check=True, capture_output=True, text=True)

        with open(step8_output) as f:
            step8_preds_raw = json.load(f)
        step8_preds = {norm_id(k): [norm_id(v) for v in vs] for k, vs in step8_preds_raw.items()}
        step8_metrics = evaluate_against_gold(step8_preds, GOLD_PATH)
        print(f"  Step8:  F1={step8_metrics['micro_f1']:.4f}  P={step8_metrics['precision']:.4f}  R={step8_metrics['recall']:.4f}  zero-F1={step8_metrics['zero_f1']}")
        results[run_name] = {"raw": raw_metrics, "step8": step8_metrics, "time": train_time, "cfg": cfg}
    except Exception as e:
        print(f"  Step8 failed: {e}")
        results[run_name] = {"raw": raw_metrics, "step8": None, "time": train_time, "cfg": cfg}
    print()

# ── Summary ──
print("=" * 70)
print("  SUMMARY — All runs vs DU4 (F1=0.3451)")
print("=" * 70)
print(f"  {'Run':<6} {'Trees':>6} {'Leaves':>7} {'LR':>6} {'sub':>5} {'col':>5} {'reg':>7} {'Step8 F1':>9} {'vs DU4':>8}")
print("  " + "-" * 68)
print(f"  {'DU3':<6} {'2000':>6} {'127':>7} {'0.03':>6} {'0.8':>5} {'0.8':>5} {'0.1/0.1':>7} {'0.3141':>9} {'-0.0310':>8}")
print(f"  {'DU4':<6} {'4000':>6} {'255':>7} {'0.02':>6} {'0.8':>5} {'0.8':>5} {'0.1/0.1':>7} {'0.3451':>9} {'---':>8}")

for name, r in results.items():
    c = r["cfg"]
    s8 = r["step8"]
    f1_str = f"{s8['micro_f1']:.4f}" if s8 else "N/A"
    best = s8['micro_f1'] if s8 else r['raw']['micro_f1']
    delta = f"{best - 0.3451:+.4f}"
    reg = f"{c['reg_alpha']}/{c['reg_lambda']}"
    print(f"  {name:<6} {c['n_estimators']:>6} {c['num_leaves']:>7} {c['learning_rate']:>6} {c['subsample']:>5} {c['colsample_bytree']:>5} {reg:>7} {f1_str:>9} {delta:>8}")

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT_DIR}")
