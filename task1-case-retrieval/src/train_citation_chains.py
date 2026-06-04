#!/usr/bin/env python3
"""
Citation chain features over the citation graph.

Builds a citation graph from all training labels, then computes 5 features
for each (query, candidate) pair:
  1. shared_citers: cases that cite both query and candidate
  2. two_hop_score: among cases the query cites, how many cite the candidate
  3. reverse_two_hop: among cases that cite the query, how many the candidate cites
  4. cocitation_jaccard: Jaccard of their cited-by sets
  5. shared_citations: cases cited by both query and candidate

Retrains LightGBM with 39 features (34 original + 5 new) using DU9's
config (4000 trees, 255 leaves, LR=0.02, sub=0.9, col=0.9, reg=0.05).

Usage: python3 train_citation_chains.py
Expected time: ~20 minutes (5 min features + 15 min training)
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
OUT_DIR = Path("./runs/citation_chains")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Label files
LABELS_2025_TRAIN = ARCHIVE / "task_one_ready_to_use/TASK1-raw-data/task1_train_labels_2025.json"
LABELS_2025_TEST = ARCHIVE / "task_one_ready_to_use/TASK1-raw-data/task1_test_labels_2025.json"
LABELS_2026_TRAIN = ARCHIVE / "task_one_ready_to_use/data/task1_train_files_2026/clean_task1_train_labels_2026.json"

def norm_id(s):
    return s.replace(".txt", "")

def load_labels(path):
    with open(path) as f:
        raw = json.load(f)
    return {norm_id(k): [norm_id(v) for v in vs] for k, vs in raw.items()}

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["labels"].astype(np.int8),
            d["qids"].tolist(), d["cids"].tolist(), d["feature_names"].tolist())

def evaluate_against_gold(preds, gold_path):
    with open(gold_path) as f:
        gold_raw = json.load(f)
    gold = {norm_id(k): [norm_id(v) for v in vs] for k, vs in gold_raw.items()}
    tp = fp = fn = 0
    pqf = []
    for qid, gc in gold.items():
        pc = set(preds.get(qid, []))
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
    return {"f1": mf, "p": mp, "r": mr, "tp": tp, "zero_f1": int((arr==0).sum())}


# build the citation graph from all training labels
print("building citation graph from all training labels...")
t0 = time.time()

# Load all labels
labels_all = {}
for lpath in [LABELS_2025_TRAIN, LABELS_2025_TEST, LABELS_2026_TRAIN]:
    if lpath.exists():
        labs = load_labels(lpath)
        labels_all.update(labs)
        print(f"  Loaded {len(labs)} queries from {lpath.name}")

print(f"  Total: {len(labels_all)} queries, {sum(len(v) for v in labels_all.values())} edges")

# cites: outgoing edges, cited_by: incoming edges
cites = defaultdict(set)
cited_by = defaultdict(set)

for qid, cited_cases in labels_all.items():
    for cid in cited_cases:
        cites[qid].add(cid)
        cited_by[cid].add(qid)

all_cases = set(cites.keys()) | set(cited_by.keys())
print(f"  Graph: {len(all_cases)} unique cases, "
      f"{len(cites)} with outgoing edges, {len(cited_by)} with incoming edges")
print(f"  Built in {time.time()-t0:.1f}s\n")


def compute_chain_features(qids, cids):
    """Compute 5 citation chain features for arrays of (qid, cid) pairs."""
    n = len(qids)
    feats = np.zeros((n, 5), dtype=np.float32)

    for i in range(n):
        q = qids[i]
        c = cids[i]

        q_cites = cites.get(q, set())
        q_cited_by = cited_by.get(q, set())
        c_cites = cites.get(c, set())
        c_cited_by = cited_by.get(c, set())

        # shared_citers: cases that cite both q and c
        feats[i, 0] = len(q_cited_by & c_cited_by)

        # two_hop_score: among cases q cites, how many cite c
        two_hop = 0
        for a in q_cites:
            if c in cites.get(a, set()):
                two_hop += 1
        feats[i, 1] = two_hop

        # reverse_two_hop: among cases that cite q, how many c cites
        rev_hop = 0
        for a in q_cited_by:
            if a in c_cites:
                rev_hop += 1
        feats[i, 2] = rev_hop

        # cocitation_jaccard: Jaccard of their cited-by sets
        if len(q_cited_by) > 0 or len(c_cited_by) > 0:
            feats[i, 3] = len(q_cited_by & c_cited_by) / len(q_cited_by | c_cited_by)

        # shared_citations: cases cited by both q and c
        feats[i, 4] = len(q_cites & c_cites)

    return feats

print("computing citation chain features...")

# Process each split
splits = {
    "train2025": CACHE_DIR / "feature_matrix_train2025.npz",
    "test2025": CACHE_DIR / "feature_matrix_test2025.npz",
    "train2026": CACHE_DIR / "feature_matrix_train2026_clean.npz",
    "test2026": CACHE_DIR / "feature_matrix_test2026.npz",
}

chain_feature_names = [
    "chain_shared_citers",
    "chain_two_hop",
    "chain_reverse_two_hop",
    "chain_cocitation_jaccard",
    "chain_shared_citations",
]

data = {}
for split_name, npz_path in splits.items():
    print(f"  Processing {split_name}...", end=" ", flush=True)
    t1 = time.time()

    X, labels, qids, cids, fnames = load_npz(npz_path)

    chain_feats = compute_chain_features(qids, cids)

    X_aug = np.hstack([X, chain_feats])
    fnames_aug = fnames + chain_feature_names

    data[split_name] = {
        "X": X_aug, "labels": labels, "qids": qids, "cids": cids,
        "fnames": fnames_aug
    }

    nonzero_pct = [(chain_feats[:, j] > 0).mean() * 100 for j in range(5)]
    print(f"{time.time()-t1:.0f}s  |  nonzero%: " +
          "  ".join(f"{chain_feature_names[j].replace('chain_','')}={nonzero_pct[j]:.1f}%" for j in range(5)))

print()


print("training LightGBM with 39 features (34 + 5 chain)...")

# merge training data
X_train = np.vstack([data["train2025"]["X"], data["test2025"]["X"], data["train2026"]["X"]])
y_train = np.concatenate([data["train2025"]["labels"], data["test2025"]["labels"], data["train2026"]["labels"]])
q_train = data["train2025"]["qids"] + data["test2025"]["qids"] + data["train2026"]["qids"]
fnames = data["train2025"]["fnames"]

# keep labelled pairs
mask = y_train >= 0
X_train, y_train = X_train[mask], y_train[mask]
q_filtered = [q for q, m in zip(q_train, mask) if m]
qid_order = list(dict.fromkeys(q_filtered))
counts = Counter(q_filtered)
groups = [counts[q] for q in qid_order]

Xt = data["test2026"]["X"]
qt = data["test2026"]["qids"]
ct = data["test2026"]["cids"]

print(f"  Training: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features")
print(f"  Test: {Xt.shape[0]:,} pairs")

# Configs to try
CONFIGS = {
    "CC1": {
        "n_estimators": 4000, "num_leaves": 255, "learning_rate": 0.02,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_alpha": 0.05, "reg_lambda": 0.05,
        "note": "DU9 config + 5 chain features"
    },
    "CC2": {
        "n_estimators": 4000, "num_leaves": 255, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "note": "DU4 config + 5 chain features"
    },
    "CC3": {
        "n_estimators": 5000, "num_leaves": 255, "learning_rate": 0.015,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_alpha": 0.05, "reg_lambda": 0.05,
        "note": "DU9 config + more rounds + 5 chain features"
    },
}

results = {}
for run_name, cfg in CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  {run_name}: {cfg['note']}")
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

    # feature importance for the chain features
    importance = dict(zip(fnames, model.feature_importance(importance_type='gain')))
    print(f"\n  Chain feature importance (gain):")
    for cf in chain_feature_names:
        gain = importance.get(cf, 0)
        rank = sorted(importance.values(), reverse=True).index(gain) + 1 if gain > 0 else "N/A"
        print(f"    {cf:<30} gain={gain:>10,.0f}  rank={rank}")

    # Predict
    scores = model.predict(Xt)
    per_query = {}
    for score, qid, cid in zip(scores, qt, ct):
        per_query.setdefault(qid, []).append((cid, float(score)))

    top5_raw = {qid: [norm_id(c) for c, _ in
                      sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
                for qid, pairs in per_query.items()}

    raw_m = evaluate_against_gold(top5_raw, GOLD_PATH)
    print(f"\n  Raw:    F1={raw_m['f1']:.4f}  P={raw_m['p']:.4f}  R={raw_m['r']:.4f}  zero-F1={raw_m['zero_f1']}")

    # Step8
    preds_dotxt = {k + ".txt": [v + ".txt" for v in vs] for k, vs in
                   {qid: [norm_id(c) for c, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:50]]
                    for qid, pairs in per_query.items()}.items()}
    preds_path = run_dir / "preds_for_step8.json"
    with open(preds_path, "w") as f:
        json.dump(preds_dotxt, f, indent=2)

    step8_out = run_dir / "step8_output.json"
    try:
        subprocess.run([sys.executable, str(STEP8_SCRIPT),
            "--corpus", str(CORPUS_2026), "--cache", str(STAGE1_CACHE),
            "--base_preds", str(preds_path), "--out", str(step8_out),
            "--out_k", "5", "--rrf_k", "5", "--remove_query_cases", "--filter_future",
        ], check=True, capture_output=True, text=True)

        with open(step8_out) as f:
            s8_raw = json.load(f)
        s8_preds = {norm_id(k): [norm_id(v) for v in vs] for k, vs in s8_raw.items()}
        s8_m = evaluate_against_gold(s8_preds, GOLD_PATH)
        print(f"  Step8:  F1={s8_m['f1']:.4f}  P={s8_m['p']:.4f}  R={s8_m['r']:.4f}  zero-F1={s8_m['zero_f1']}")

        with open(run_dir / "step8_final_predictions.json", "w") as f:
            json.dump(s8_preds, f, indent=2)
        results[run_name] = {"raw": raw_m, "step8": s8_m, "time": train_time}
    except Exception as e:
        print(f"  Step8 failed: {e}")
        results[run_name] = {"raw": raw_m, "step8": None, "time": train_time}


print(f"\n{'='*70}")
print(f"  Summary: citation chain features vs DU9 baseline (F1=0.3456)")
print(f"{'='*70}")
print(f"  {'Run':<6} {'Features':>8} {'Step8 F1':>9} {'vs DU9':>8} {'Time':>8}")
print("  " + "-" * 50)
print(f"  {'DU9':<6} {'34':>8} {'0.3456':>9} {'---':>8} {'14min':>8}")

for name, r in results.items():
    s8 = r["step8"]
    f1_str = f"{s8['f1']:.4f}" if s8 else f"{r['raw']['f1']:.4f}"
    best = s8['f1'] if s8 else r['raw']['f1']
    delta = f"{best - 0.3456:+.4f}"
    print(f"  {name:<6} {'39':>8} {f1_str:>9} {delta:>8} {r['time']/60:.0f}min")

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT_DIR}")
