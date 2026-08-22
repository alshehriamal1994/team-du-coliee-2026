#!/usr/bin/env python3
"""
Vote ensemble across all available runs.
For each query, counts how many runs predict each candidate,
then picks the top-5 by vote count (ties broken by best run's rank).

No training needed, just combines existing predictions.

Usage: python3 vote_ensemble.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(".")
GOLD_PATH = BASE / "FINAL_SUBMISSION" / "task1_test_labels_2026.json"

def norm_id(s):
    return s.replace(".txt", "")

def load_submission(path):
    preds = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                preds[norm_id(parts[0])].append(norm_id(parts[1]))
    return dict(preds)

def load_json_preds(path):
    with open(path) as f:
        raw = json.load(f)
    return {norm_id(k): [norm_id(v) for v in vs] for k, vs in raw.items()}

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
    return {"f1": mf, "p": mp, "r": mr, "tp": tp, "zero_f1": int((arr==0).sum())}

# load gold
with open(GOLD_PATH) as f:
    gold_raw = json.load(f)
gold = {norm_id(k): [norm_id(v) for v in vs] for k, vs in gold_raw.items()}
print(f"Gold: {len(gold)} queries, {sum(len(v) for v in gold.values())} relevant\n")

# load all available runs
runs = {}

# competition submissions
for name in ["DU1", "DU2", "DU3"]:
    p = BASE / "FINAL_SUBMISSION" / f"{name}.txt"
    if p.exists():
        runs[name] = load_submission(p)

# post-competition runs (step8 predictions)
for name in ["du4", "du5", "du6"]:
    p = BASE / "runs" / "du4_bigger" / name / "step8_final_predictions.json"
    if p.exists():
        runs[name.upper()] = load_json_preds(p)

# DU7-DU9 if they exist
for name in ["du7", "du8", "du9"]:
    p = BASE / "runs" / "du7_tuning" / name / "step8_final_predictions.json"
    if p.exists():
        runs[name.upper()] = load_json_preds(p)
    # Also try step8_output.json
    p2 = BASE / "runs" / "du7_tuning" / name / "step8_output.json"
    if p2.exists() and name.upper() not in runs:
        runs[name.upper()] = load_json_preds(p2)

print(f"Loaded {len(runs)} runs: {', '.join(runs.keys())}\n")

# evaluate individual runs
print("Individual run performance:")
print(f"  {'Run':<6} {'F1':>7} {'P':>7} {'R':>7} {'Zero':>5}")
print("  " + "-" * 35)
run_scores = {}
for name, preds in runs.items():
    m = evaluate(preds, gold)
    run_scores[name] = m["f1"]
    print(f"  {name:<6} {m['f1']:.4f}  {m['p']:.4f}  {m['r']:.4f}  {m['zero_f1']:>4}")

# sort runs by F1 (best first) for tie-breaking
runs_sorted = sorted(runs.keys(), key=lambda n: run_scores[n], reverse=True)
print(f"\nRun ranking by F1: {' > '.join(runs_sorted)}\n")

# vote ensembles
def vote_ensemble(run_names, runs_dict, top_k=5):
    """For each query, count votes across runs. Break ties by best run's rank position."""
    all_qids = set()
    for preds in runs_dict.values():
        all_qids.update(preds.keys())

    ensemble_preds = {}
    for qid in all_qids:
        # Count votes
        vote_counts = Counter()
        # Also track best rank position for tie-breaking
        best_rank = {}  # candidate -> best rank across runs
        for rname in run_names:
            if qid in runs_dict[rname]:
                for rank, cid in enumerate(runs_dict[rname][qid]):
                    vote_counts[cid] += 1
                    if cid not in best_rank or rank < best_rank[cid]:
                        best_rank[cid] = rank

        # Sort by: (1) vote count descending, (2) best rank ascending
        candidates = sorted(vote_counts.keys(),
                          key=lambda c: (-vote_counts[c], best_rank.get(c, 999)))
        ensemble_preds[qid] = candidates[:top_k]

    return ensemble_preds

# Try different ensemble combinations
print("=" * 60)
print("  Vote Ensemble Results")
print("=" * 60)

combos = [
    ("Top2", ["DU4", "DU3"]),
    ("Top3", ["DU4", "DU6", "DU3"]),
    ("Top3+DU2", ["DU4", "DU6", "DU2"]),
    ("All_submitted", ["DU1", "DU2", "DU3"]),
    ("All_post", [n for n in ["DU4", "DU5", "DU6"] if n in runs]),
    ("Best4", [n for n in runs_sorted[:4] if n in runs]),
    ("Best5", [n for n in runs_sorted[:5] if n in runs]),
    ("ALL", list(runs.keys())),
]

# Also try DU7-DU9 combos if they exist
if "DU7" in runs:
    combos.append(("DU4+DU7", ["DU4", "DU7"]))
if "DU8" in runs:
    combos.append(("DU4+DU8", ["DU4", "DU8"]))
if "DU9" in runs:
    combos.append(("DU4+DU9", ["DU4", "DU9"]))
if all(n in runs for n in ["DU4", "DU7", "DU8", "DU9"]):
    combos.append(("DU4+DU7-9", ["DU4", "DU7", "DU8", "DU9"]))

best_f1 = 0
best_name = ""

print(f"\n  {'Ensemble':<15} {'Runs':>4} {'F1':>7} {'P':>7} {'R':>7} {'Zero':>5} {'vs DU4':>8}")
print("  " + "-" * 55)

for combo_name, combo_runs in combos:
    valid_runs = [r for r in combo_runs if r in runs]
    if len(valid_runs) < 2:
        continue
    ens_preds = vote_ensemble(valid_runs, runs)
    m = evaluate(ens_preds, gold)
    delta = f"{m['f1'] - 0.3451:+.4f}"
    marker = " ***" if m["f1"] > best_f1 else ""
    print(f"  {combo_name:<15} {len(valid_runs):>4} {m['f1']:.4f}  {m['p']:.4f}  {m['r']:.4f}  {m['zero_f1']:>4}  {delta:>8}{marker}")
    if m["f1"] > best_f1:
        best_f1 = m["f1"]
        best_name = combo_name
        best_preds = ens_preds

print(f"\n  Best ensemble: {best_name} (F1={best_f1:.4f})")

# save best ensemble
out_path = BASE / "runs" / "vote_ensemble_best.json"
with open(out_path, "w") as f:
    json.dump(best_preds, f, indent=2)

# Also save as submission format
sub_path = BASE / "runs" / "vote_ensemble_best.txt"
with open(sub_path, "w") as f:
    for qid in sorted(best_preds.keys()):
        for cid in best_preds[qid]:
            f.write(f"{qid} {cid} VOTE_ENS\n")

print(f"  Saved to {out_path}")
print(f"  Submission: {sub_path}")
