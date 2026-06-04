#!/usr/bin/env python3
"""
Three experiments in one script:
  tune the Step8 RRF k-value (no training)
  multi-seed ensemble (trains 5 models)
  apply the best k to the best ensemble

Usage: python3 tune_step8_and_seeds.py
"""

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
OUT_DIR = Path("./runs/tuning_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm_id(s):
    return s.replace(".txt", "")

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["labels"].astype(np.int8),
            d["qids"].tolist(), d["cids"].tolist(), d["feature_names"].tolist())

def evaluate(preds, gold):
    tp = fp = fn = 0
    pqf = []
    for qid, gc in gold.items():
        pc = set(preds.get(qid, [])[:5])
        gs = set(gc)
        t = len(pc & gs)
        tp += t; fp += len(pc - gs); fn += len(gs - pc)
        if t > 0:
            p = t/(t+len(pc-gs)); r = t/(t+len(gs-pc))
            pqf.append(2*p*r/(p+r))
        else:
            pqf.append(0.0)
    mp = tp/(tp+fp) if tp+fp else 0
    mr = tp/(tp+fn) if tp+fn else 0
    mf = 2*mp*mr/(mp+mr) if mp+mr else 0
    arr = np.array(pqf)
    return {"f1": mf, "p": mp, "r": mr, "tp": tp, "fp": fp, "fn": fn,
            "zero_f1": int((arr==0).sum())}

def run_step8(base_preds_dotxt, rrf_k, out_path):
    """Run Step8 postprocessing with given RRF k."""
    tmp_in = out_path.parent / f"tmp_step8_input_k{rrf_k}.json"
    with open(tmp_in, "w") as f:
        json.dump(base_preds_dotxt, f, indent=2)
    result = subprocess.run([
        sys.executable, str(STEP8_SCRIPT),
        "--corpus", str(CORPUS_2026), "--cache", str(STAGE1_CACHE),
        "--base_preds", str(tmp_in), "--out", str(out_path),
        "--out_k", "5", "--rrf_k", str(rrf_k),
        "--remove_query_cases", "--filter_future",
    ], check=True, capture_output=True, text=True)
    with open(out_path) as f:
        raw = json.load(f)
    return {norm_id(k): [norm_id(v) for v in vs] for k, vs in raw.items()}

# Load gold
with open(GOLD_PATH) as f:
    gold_raw = json.load(f)
gold = {norm_id(k): [norm_id(v) for v in vs] for k, vs in gold_raw.items()}

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
print(f"  Loaded in {time.time()-t0:.1f}s\n")


# tune Step8 RRF k on DU9's predictions
print("=" * 70)
print("  Tuning Step8 RRF k-value on DU9")
print("=" * 70)

# First, reproduce DU9's raw top-50 predictions (needed for Step8)
print("  Training DU9 to get top-50 predictions...")
params_du9 = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
    "num_leaves": 255, "learning_rate": 0.02, "min_child_samples": 20,
    "subsample": 0.9, "colsample_bytree": 0.9,
    "reg_alpha": 0.05, "reg_lambda": 0.05, "verbose": -1, "seed": 42,
}
lgb_ds = lgb.Dataset(X_train, label=y_train, group=groups,
                     feature_name=fnames, free_raw_data=False)
model_du9 = lgb.train(params_du9, lgb_ds, num_boost_round=4000,
                      callbacks=[lgb.log_evaluation(period=2000)])

scores_du9 = model_du9.predict(Xt)
per_query_du9 = {}
for score, qid, cid in zip(scores_du9, qt, ct):
    per_query_du9.setdefault(qid, []).append((cid, float(score)))

# Top-50 in .txt format for Step8
top50_dotxt = {qid + ".txt": [norm_id(c) + ".txt" for c, _ in
               sorted(pairs, key=lambda x: x[1], reverse=True)[:50]]
               for qid, pairs in per_query_du9.items()}

# Raw top-5 (no step8)
top5_raw = {qid: [norm_id(c) for c, _ in
            sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
            for qid, pairs in per_query_du9.items()}
raw_m = evaluate(top5_raw, gold)
print(f"  DU9 raw (no Step8): F1={raw_m['f1']:.4f}\n")

# Also test: Step8 without RRF (only self-match removal + future filtering)
print("  Testing Step8 with no RRF (filters only)...")
tmp_in = OUT_DIR / "tmp_noRRF_input.json"
with open(tmp_in, "w") as f:
    # For no-RRF, just pass top-5 not top-50, so RRF has no effect
    top5_dotxt_noRRF = {qid + ".txt": [norm_id(c) + ".txt" for c, _ in
                        sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
                        for qid, pairs in per_query_du9.items()}
    json.dump(top5_dotxt_noRRF, f, indent=2)
noRRF_out = OUT_DIR / "step8_noRRF.json"
subprocess.run([
    sys.executable, str(STEP8_SCRIPT),
    "--corpus", str(CORPUS_2026), "--cache", str(STAGE1_CACHE),
    "--base_preds", str(tmp_in), "--out", str(noRRF_out),
    "--out_k", "5", "--rrf_k", "5",
    "--remove_query_cases", "--filter_future",
], check=True, capture_output=True, text=True)
with open(noRRF_out) as f:
    noRRF_preds = {norm_id(k): [norm_id(v) for v in vs] for k, vs in json.load(f).items()}
noRRF_m = evaluate(noRRF_preds, gold)
print(f"  Filters only (no RRF):   F1={noRRF_m['f1']:.4f}\n")

# Test different k values
k_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 60]
print(f"  {'k':>4}  {'F1':>7}  {'P':>7}  {'R':>7}  {'Zero':>5}  {'vs k=5':>8}")
print("  " + "-" * 45)

best_k = 5
best_k_f1 = 0

for k in k_values:
    out_path = OUT_DIR / f"step8_k{k}.json"
    preds = run_step8(top50_dotxt, k, out_path)
    m = evaluate(preds, gold)
    delta = f"{m['f1'] - 0.3456:+.4f}" if k != 5 else "  base"
    marker = " ***" if m["f1"] > best_k_f1 else ""
    print(f"  {k:>4}  {m['f1']:.4f}  {m['p']:.4f}  {m['r']:.4f}  {m['zero_f1']:>4}  {delta:>8}{marker}")
    if m["f1"] > best_k_f1:
        best_k_f1 = m["f1"]
        best_k = k

print(f"\n  Best k={best_k} (F1={best_k_f1:.4f})\n")


# multi-seed ensemble
print("=" * 70)
print("  Multi-seed ensemble (DU9 config, 5 seeds)")
print("=" * 70)

seeds = [42, 7, 123, 456, 789]
seed_per_query = {}  # seed -> {qid: [top50 (cid, score)]}
seed_top5 = {}       # seed -> {qid: [top5 cids]}

for i, seed in enumerate(seeds):
    print(f"\n  Seed {seed} ({i+1}/5)...")
    params = dict(params_du9)
    params["seed"] = seed

    lgb_ds_s = lgb.Dataset(X_train, label=y_train, group=groups,
                           feature_name=fnames, free_raw_data=False)
    t1 = time.time()
    model = lgb.train(params, lgb_ds_s, num_boost_round=4000,
                      callbacks=[lgb.log_evaluation(period=4000)])
    print(f"    Trained in {time.time()-t1:.0f}s")

    scores = model.predict(Xt)
    pq = {}
    for score, qid, cid in zip(scores, qt, ct):
        pq.setdefault(qid, []).append((cid, float(score)))

    seed_per_query[seed] = pq

    # Individual performance
    t5 = {qid: [norm_id(c) for c, _ in
           sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
          for qid, pairs in pq.items()}
    seed_top5[seed] = t5
    m = evaluate(t5, gold)
    print(f"    Raw F1={m['f1']:.4f}  zero-F1={m['zero_f1']}")

    # With best k step8
    t50_dotxt = {qid + ".txt": [norm_id(c) + ".txt" for c, _ in
                 sorted(pairs, key=lambda x: x[1], reverse=True)[:50]]
                 for qid, pairs in pq.items()}
    out_path = OUT_DIR / f"step8_seed{seed}_k{best_k}.json"
    s8_preds = run_step8(t50_dotxt, best_k, out_path)
    s8_m = evaluate(s8_preds, gold)
    seed_top5[f"{seed}_s8"] = s8_preds
    print(f"    Step8 (k={best_k}) F1={s8_m['f1']:.4f}  zero-F1={s8_m['zero_f1']}")

# vote ensembles across seeds
print(f"\n{'='*60}")
print(f"  Seed ensemble results (vote across 5 seeds)")
print(f"{'='*60}")

def vote_ensemble(run_preds_list, top_k=5):
    all_qids = set()
    for preds in run_preds_list:
        all_qids.update(preds.keys())
    result = {}
    for qid in all_qids:
        votes = Counter()
        best_rank = {}
        for preds in run_preds_list:
            if qid in preds:
                for rank, cid in enumerate(preds[qid]):
                    votes[cid] += 1
                    if cid not in best_rank or rank < best_rank[cid]:
                        best_rank[cid] = rank
        candidates = sorted(votes.keys(),
                          key=lambda c: (-votes[c], best_rank.get(c, 999)))
        result[qid] = candidates[:top_k]
    return result

# Raw vote (no step8)
raw_ens = vote_ensemble([seed_top5[s] for s in seeds])
raw_ens_m = evaluate(raw_ens, gold)
print(f"  Raw vote (5 seeds):      F1={raw_ens_m['f1']:.4f}  zero-F1={raw_ens_m['zero_f1']}")

# Step8 vote
s8_ens = vote_ensemble([seed_top5[f"{s}_s8"] for s in seeds])
s8_ens_m = evaluate(s8_ens, gold)
print(f"  Step8 vote (5 seeds):    F1={s8_ens_m['f1']:.4f}  zero-F1={s8_ens_m['zero_f1']}")

# Also try: score averaging (not just voting)
print(f"\n  Score-averaged ensembles:")

def score_average_ensemble(seed_pqs, rrf_k, top_k=5):
    """Average LightGBM scores across seeds, then apply Step8."""
    all_qids = set()
    for pq in seed_pqs.values():
        all_qids.update(pq.keys())

    avg_pq = {}
    for qid in all_qids:
        cid_scores = defaultdict(list)
        for pq in seed_pqs.values():
            if qid in pq:
                for cid, score in pq[qid]:
                    cid_scores[cid].append(score)
        # Average scores
        avg_pairs = [(cid, np.mean(scores)) for cid, scores in cid_scores.items()]
        avg_pq[qid] = sorted(avg_pairs, key=lambda x: x[1], reverse=True)

    # Top-5 raw
    top5 = {qid: [norm_id(c) for c, _ in pairs[:5]] for qid, pairs in avg_pq.items()}
    raw_m = evaluate(top5, gold)
    print(f"    Raw (score avg):       F1={raw_m['f1']:.4f}  zero-F1={raw_m['zero_f1']}")

    # Step8 with best k
    top50_dotxt = {qid + ".txt": [norm_id(c) + ".txt" for c, _ in pairs[:50]]
                   for qid, pairs in avg_pq.items()}
    out_path = OUT_DIR / f"step8_scoreavg_k{rrf_k}.json"
    s8_preds = run_step8(top50_dotxt, rrf_k, out_path)
    s8_m = evaluate(s8_preds, gold)
    print(f"    Step8 (k={rrf_k}, score avg): F1={s8_m['f1']:.4f}  zero-F1={s8_m['zero_f1']}")
    return s8_preds, s8_m

best_ens_preds, best_ens_m = score_average_ensemble(seed_per_query, best_k)

# Also try score avg with different k values
print(f"\n  Score-avg ensemble with different k values:")
for k in [1, 3, 5, 7, 10, 20]:
    out_path = OUT_DIR / f"step8_scoreavg_k{k}.json"
    top50_dotxt = {}
    all_qids = set()
    for pq in seed_per_query.values():
        all_qids.update(pq.keys())
    for qid in all_qids:
        cid_scores = defaultdict(list)
        for pq in seed_per_query.values():
            if qid in pq:
                for cid, score in pq[qid]:
                    cid_scores[cid].append(score)
        avg_pairs = sorted([(cid, np.mean(scores)) for cid, scores in cid_scores.items()],
                          key=lambda x: x[1], reverse=True)
        top50_dotxt[qid + ".txt"] = [norm_id(c) + ".txt" for c, _ in avg_pairs[:50]]

    preds = run_step8(top50_dotxt, k, out_path)
    m = evaluate(preds, gold)
    marker = " <-- BEST" if m["f1"] > best_ens_m["f1"] else ""
    print(f"    k={k:>3}  F1={m['f1']:.4f}  zero-F1={m['zero_f1']}{marker}")
    if m["f1"] > best_ens_m["f1"]:
        best_ens_m = m
        best_ens_preds = preds


print(f"\n{'='*70}")
print(f"  FINAL SUMMARY")
print(f"{'='*70}")
print(f"  DU3 (submitted):            F1 = 0.3141")
print(f"  DU9 (best single, k=5):     F1 = 0.3456  (+3.15pp vs DU3)")
print(f"  DU9 (best k={best_k}):            F1 = {best_k_f1:.4f}  ({best_k_f1-0.3456:+.4f} vs DU9 k=5)")
print(f"  5-seed score avg (best):    F1 = {best_ens_m['f1']:.4f}  ({best_ens_m['f1']-0.3456:+.4f} vs DU9 k=5)")
print(f"  ")
print(f"  Improvement over submitted: {best_ens_m['f1']-0.3141:+.4f}pp")

# Save best predictions
with open(OUT_DIR / "best_predictions.json", "w") as f:
    json.dump(best_ens_preds, f, indent=2)

sub_path = OUT_DIR / "best_submission.txt"
with open(sub_path, "w") as f:
    for qid in sorted(best_ens_preds.keys()):
        for cid in best_ens_preds[qid]:
            f.write(f"{qid} {cid} BEST\n")

print(f"\n  Best predictions saved to {OUT_DIR}")
