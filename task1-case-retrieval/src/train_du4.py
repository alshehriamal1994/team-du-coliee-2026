#!/usr/bin/env python3
"""
DU4: bigger model experiment.
4000 trees, 255 leaves, LR 0.02, double DU3's capacity.
Evaluates against gold labels directly.

Usage: python3 train_du4.py
Expected time: ~40-50 minutes on CPU
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

# paths
ARCHIVE = Path("./ARCHIVE")
CACHE_DIR = ARCHIVE / "cache_features"
STEP8_SCRIPT = ARCHIVE / "code_AUTHORITY_v2/step8_postprocess_filters_v2.py"
CORPUS_2026 = ARCHIVE / "task_one_ready_to_use/data/task1_test_files_2026/task1_test_files_2026"
STAGE1_CACHE = ARCHIVE / "cache_2026/test_cache_dotxt.pkl"
GOLD_PATH = Path("./FINAL_SUBMISSION/task1_test_labels_2026.json")
OUT_DIR = Path("./runs/du4_bigger")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# hyperparameters
CONFIGS = {
    "DU4": {"n_estimators": 4000, "num_leaves": 255, "learning_rate": 0.02},
    "DU5": {"n_estimators": 4000, "num_leaves": 255, "learning_rate": 0.01},
    "DU6": {"n_estimators": 6000, "num_leaves": 511, "learning_rate": 0.01},
}

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32),
            d["labels"].astype(np.int8),
            d["qids"].tolist(),
            d["cids"].tolist(),
            d["feature_names"].tolist())

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
        tp += t
        fp += len(pc - gs)
        fn += len(gs - pc)
        if t > 0:
            p = t / (t + len(pc - gs))
            r = t / (t + len(gs - pc))
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        per_query_f1.append(f1)

    micro_p = tp / (tp + fp) if tp + fp else 0
    micro_r = tp / (tp + fn) if tp + fn else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0
    arr = np.array(per_query_f1)
    return {
        "micro_f1": micro_f1, "precision": micro_p, "recall": micro_r,
        "tp": tp, "fp": fp, "fn": fn,
        "zero_f1": int((arr == 0).sum()),
        "mean_f1": float(arr.mean()),
    }

# load data
print("Loading feature matrices...")
t0 = time.time()

X1, y1, q1, c1, fnames = load_npz(CACHE_DIR / "feature_matrix_train2025.npz")
X2, y2, q2, c2, _ = load_npz(CACHE_DIR / "feature_matrix_test2025.npz")
X3, y3, q3, c3, _ = load_npz(CACHE_DIR / "feature_matrix_train2026_clean.npz")
Xt, yt, qt, ct, _ = load_npz(CACHE_DIR / "feature_matrix_test2026.npz")

# Merge training data
X_train = np.vstack([X1, X2, X3])
y_train = np.concatenate([y1, y2, y3])
q_train = q1 + q2 + q3

# Filter labeled pairs
mask = y_train >= 0
X_train = X_train[mask]
y_train = y_train[mask]
q_filtered = [q for q, m in zip(q_train, mask) if m]

# Build group sizes
from collections import Counter
qid_order = []
seen = set()
for q in q_filtered:
    if q not in seen:
        qid_order.append(q)
        seen.add(q)
counts = Counter(q_filtered)
groups = [counts[q] for q in qid_order]

print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Training: {X_train.shape[0]:,} pairs, {len(groups):,} queries")
print(f"  Test: {Xt.shape[0]:,} pairs")
print(f"  Features: {len(fnames)}")
print()

# train and evaluate each config
results = {}

for run_name, cfg in CONFIGS.items():
    print(f"{'='*60}")
    print(f"  Training {run_name}: {cfg['n_estimators']} trees, {cfg['num_leaves']} leaves, LR={cfg['learning_rate']}")
    print(f"{'='*60}")

    run_dir = OUT_DIR / run_name.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, group=groups,
                            feature_name=fnames, free_raw_data=False)

    t1 = time.time()
    model = lgb.train(params, lgb_train,
                      num_boost_round=cfg["n_estimators"],
                      valid_sets=[lgb_train],
                      callbacks=[lgb.log_evaluation(period=500)])
    train_time = time.time() - t1
    print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    # Save model
    model.save_model(str(run_dir / "model.txt"))

    # Predict on test
    scores = model.predict(Xt)
    per_query = {}
    for score, qid, cid in zip(scores, qt, ct):
        per_query.setdefault(qid, []).append((cid, float(score)))

    # Top-5 raw
    top5_raw = {qid: [norm_id(c) for c, _ in
                      sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
                for qid, pairs in per_query.items()}

    # Save raw predictions
    preds_path = run_dir / "test_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(top5_raw, f, indent=2)

    # Evaluate raw (no step8)
    raw_metrics = evaluate_against_gold(top5_raw, GOLD_PATH)
    print(f"  Raw (no step8):  F1={raw_metrics['micro_f1']:.4f}  P={raw_metrics['precision']:.4f}  R={raw_metrics['recall']:.4f}  zero-F1={raw_metrics['zero_f1']}")

    # Step8 postprocessing
    preds_dotxt = {k + ".txt": [v + ".txt" for v in vs] for k, vs in
                   {qid: [norm_id(c) for c, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:50]]
                    for qid, pairs in per_query.items()}.items()}
    preds_dotxt_path = run_dir / "preds_for_step8.json"
    with open(preds_dotxt_path, "w") as f:
        json.dump(preds_dotxt, f, indent=2)

    step8_output = run_dir / "step8_output.json"
    try:
        subprocess.run([
            sys.executable, str(STEP8_SCRIPT),
            "--corpus", str(CORPUS_2026),
            "--cache", str(STAGE1_CACHE),
            "--base_preds", str(preds_dotxt_path),
            "--out", str(step8_output),
            "--out_k", "5",
            "--rrf_k", "5",
            "--remove_query_cases",
            "--filter_future",
        ], check=True, capture_output=True, text=True)

        with open(step8_output) as f:
            step8_preds_raw = json.load(f)
        step8_preds = {norm_id(k): [norm_id(v) for v in vs] for k, vs in step8_preds_raw.items()}

        step8_metrics = evaluate_against_gold(step8_preds, GOLD_PATH)
        print(f"  Step8:           F1={step8_metrics['micro_f1']:.4f}  P={step8_metrics['precision']:.4f}  R={step8_metrics['recall']:.4f}  zero-F1={step8_metrics['zero_f1']}")

        # Save final
        with open(run_dir / "step8_final_predictions.json", "w") as f:
            json.dump(step8_preds, f, indent=2)

        results[run_name] = {"raw": raw_metrics, "step8": step8_metrics, "time": train_time}
    except Exception as e:
        print(f"  Step8 failed: {e}")
        print("  Using raw predictions only.")
        results[run_name] = {"raw": raw_metrics, "step8": None, "time": train_time}

    print()

print("=" * 70)
print("  Summary: all runs vs DU3 baseline (F1=0.3141)")
print("=" * 70)
print(f"  {'Run':<8} {'Trees':>6} {'Leaves':>7} {'LR':>6} {'Raw F1':>8} {'Step8 F1':>9} {'Time':>8} {'vs DU3':>8}")
print("  " + "-" * 65)

# DU3 reference
print(f"  {'DU3':<8} {'2000':>6} {'127':>7} {'0.03':>6} {'---':>8} {'0.3141':>9} {'22min':>8} {'---':>8}")

for name, r in results.items():
    cfg = CONFIGS[name]
    raw_f1 = f"{r['raw']['micro_f1']:.4f}"
    s8_f1 = f"{r['step8']['micro_f1']:.4f}" if r['step8'] else "N/A"
    time_str = f"{r['time']/60:.0f}min"
    best_f1 = r['step8']['micro_f1'] if r['step8'] else r['raw']['micro_f1']
    delta = f"{best_f1 - 0.3141:+.4f}"
    print(f"  {name:<8} {cfg['n_estimators']:>6} {cfg['num_leaves']:>7} {cfg['learning_rate']:>6} {raw_f1:>8} {s8_f1:>9} {time_str:>8} {delta:>8}")

# Save summary
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUT_DIR}")
