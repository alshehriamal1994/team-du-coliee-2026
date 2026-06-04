#!/usr/bin/env python3
"""
Comprehensive post-hoc experiments for COLIEE 2026 Task 2 paper.
Uses gold labels + cached reranker scores + individual LLM run files.
"""
import json
import pickle
import numpy as np
from collections import defaultdict, Counter

# ============================================================
# LOAD DATA
# ============================================================

# Gold labels
with open("./task2_test_labels_2026(1).json") as f:
    raw_gold = json.load(f)
gold = {}
for cid, val in raw_gold.items():
    gold[cid] = {x.strip().replace(".txt", "").zfill(3) for x in val.split(",")}
ALL_CASES = sorted(gold.keys(), key=int)
TOTAL_RELEVANT = sum(len(g) for g in gold.values())

# Reranker cache (MonoT5-v2 + Qwen3 scores for all 100 candidates per case)
with open("cache/runs_final_2026/test_cache_monot5v2.pkl", "rb") as f:
    cache_v2 = pickle.load(f)

# Also load MonoT5-v1 cache if available
try:
    with open("cache/runs_final_2026/test_cache_stage1top100_qwen2048.pkl", "rb") as f:
        cache_v1 = pickle.load(f)
    # v1 and v2 have same structure, rows is the key
except:
    cache_v1 = None

# Build per-case reranker score dicts
def build_score_index(cache):
    """Build {case_id: {para_id: (m5_score, q3_score)}}"""
    index = {}
    for row in cache['rows']:
        cid = row['cid']
        scores = {}
        for i, pid in enumerate(row['cand_ids']):
            pid_norm = pid.zfill(3)
            scores[pid_norm] = (row['m5'][i], row['q3'][i])
        index[cid] = scores
    return index

scores_v2 = build_score_index(cache_v2)
scores_v1 = build_score_index(cache_v1) if cache_v1 else scores_v2

# Load individual LLM run files
def load_run(path):
    preds = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                preds[parts[0]].add(parts[1].zfill(3))
    return dict(preds)

BASE = "cache/runs_experiments/"
FINAL = "predictions/"

# Individual model runs on test set
v3_rag = load_run(BASE + "test2026_v3_rag_k20.txt")
r1_rag = load_run(BASE + "test2026_r1_rag_k20.txt")
v3_zero = load_run(BASE + "test2026_v3_forced_top1_k20.txt")
r1_zero = load_run(BASE + "test2026_r1_forced_top1_k20.txt")
llama_zero = load_run(BASE + "test2026_llama33_forced_top1_k20.txt")

# Try to load QWQ too
try:
    qwq_zero = load_run(BASE + "test2026_qwq_forced_top1_k20.txt")
except: qwq_zero = {}

try:
    llama_zero_alt = load_run(BASE + "test2026_llama_zero_k20.txt")
except: llama_zero_alt = {}

# Final submissions
du1 = load_run(FINAL + "DU1/task2_DU1.txt")
du2 = load_run(FINAL + "DU2/task2_DU2.txt")
du3 = load_run(FINAL + "DU3/task2_DU3.txt")

# DU2 v3-rag with monot5v2
du2_v3rag_monot5v2 = load_run("cache/runs_final_2026/test_v3_rag_monot5v2.txt")

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(preds, name=None, verbose=False):
    correct = retrieved = 0
    for cid in ALL_CASES:
        g = gold[cid]
        p = preds.get(cid, set())
        correct += len(g & p)
        retrieved += len(p)
    micro_p = correct / retrieved if retrieved else 0
    micro_r = correct / TOTAL_RELEVANT
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) else 0
    if verbose and name:
        print(f"  {name:55s}  P={micro_p:.4f} R={micro_r:.4f} F1={micro_f1:.4f}  (c={correct} ret={retrieved} rel={TOTAL_RELEVANT})")
    return micro_p, micro_r, micro_f1

# ============================================================
# EXPERIMENT 1: BASELINE — individual model runs scored against gold
# ============================================================

print("=" * 90)
print("  EXPERIMENT 1: Individual Model Runs on Test Set")
print("=" * 90)
for name, run in [
    ("V3 zero-shot k=20", v3_zero),
    ("R1 zero-shot k=20", r1_zero),
    ("LLaMA-3.3 zero-shot k=20", llama_zero),
    ("QWQ zero-shot k=20", qwq_zero),
    ("LLaMA zero-shot (alt) k=20", llama_zero_alt),
    ("V3 Legal-RAG k=20 (MonoT5-v1 reranker)", v3_rag),
    ("R1 Legal-RAG k=20 (MonoT5-v1 reranker)", r1_rag),
    ("V3 Legal-RAG k=20 (MonoT5-v2 reranker)", du2_v3rag_monot5v2),
    ("--- DU1 (submitted)", du1),
    ("--- DU2 (submitted)", du2),
    ("--- DU3 (submitted)", du3),
]:
    if run:
        evaluate(run, name, verbose=True)

# ============================================================
# EXPERIMENT 2: Multi-model combination strategies (from existing runs)
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 2: Multi-Model Combination Strategies")
print("=" * 90)

def combine_runs(runs, strategy="union"):
    result = {}
    for cid in ALL_CASES:
        sets = [r.get(cid, set()) for r in runs]
        if strategy == "union":
            result[cid] = set().union(*sets)
        elif strategy == "intersection":
            non_empty = [s for s in sets if s]
            result[cid] = set.intersection(*non_empty) if non_empty else set()
        elif strategy == "majority":
            counts = Counter(pid for s in sets for pid in s)
            threshold = len(runs) // 2 + 1
            result[cid] = {pid for pid, cnt in counts.items() if cnt >= threshold}
    return result

# All possible pairwise and triple unions
print("\n  --- Union strategies ---")
for name, runs in [
    ("V3_rag ∪ R1_rag", [v3_rag, r1_rag]),
    ("V3_zero ∪ R1_zero", [v3_zero, r1_zero]),
    ("V3_zero ∪ R1_zero ∪ LLaMA_zero", [v3_zero, r1_zero, llama_zero]),
    ("V3_rag ∪ R1_rag ∪ LLaMA_zero", [v3_rag, r1_rag, llama_zero]),
    ("DU1 ∪ DU2 ∪ DU3", [du1, du2, du3]),
    ("All 5 models union (V3r,R1r,V3z,R1z,LL)", [v3_rag, r1_rag, v3_zero, r1_zero, llama_zero]),
]:
    evaluate(combine_runs(runs, "union"), name, verbose=True)

print("\n  --- Intersection strategies ---")
for name, runs in [
    ("V3_rag ∩ R1_rag", [v3_rag, r1_rag]),
    ("V3_zero ∩ R1_zero", [v3_zero, r1_zero]),
    ("V3_zero ∩ R1_zero ∩ LLaMA_zero", [v3_zero, r1_zero, llama_zero]),
]:
    evaluate(combine_runs(runs, "intersection"), name, verbose=True)

print("\n  --- Majority voting ---")
for name, runs in [
    ("Majority(V3z, R1z, LLaMA)", [v3_zero, r1_zero, llama_zero]),
    ("Majority(V3r, R1r, LLaMA)", [v3_rag, r1_rag, llama_zero]),
    ("Majority(V3r, R1r, V3z, R1z, LLaMA)", [v3_rag, r1_rag, v3_zero, r1_zero, llama_zero]),
]:
    evaluate(combine_runs(runs, "majority"), name, verbose=True)

# ============================================================
# EXPERIMENT 3: Reranker-only baselines (no LLM)
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 3: Reranker-Only Baselines (MonoT5-v2 + Qwen3 ensemble)")
print("=" * 90)

def reranker_topN(scores_index, n=1, w_m5=0.8):
    """Predict top-N by ensemble score."""
    preds = {}
    for cid in ALL_CASES:
        if cid not in scores_index:
            continue
        sc = scores_index[cid]
        # Min-max normalize
        m5_scores = np.array([v[0] for v in sc.values()])
        q3_scores = np.array([v[1] for v in sc.values()])
        pids = list(sc.keys())

        r1 = m5_scores.max() - m5_scores.min()
        r2 = q3_scores.max() - q3_scores.min()
        n1 = np.ones_like(m5_scores) if r1 < 1e-9 else (m5_scores - m5_scores.min()) / r1
        n2 = np.ones_like(q3_scores) if r2 < 1e-9 else (q3_scores - q3_scores.min()) / r2

        combined = w_m5 * n1 + (1 - w_m5) * n2
        order = np.argsort(-combined)
        preds[cid] = {pids[i] for i in order[:n]}
    return preds

def reranker_dynamic_margin(scores_index, margin=0.10, w_m5=0.8, max_preds=5):
    """Predict with dynamic margin threshold."""
    preds = {}
    for cid in ALL_CASES:
        if cid not in scores_index:
            continue
        sc = scores_index[cid]
        m5_scores = np.array([v[0] for v in sc.values()])
        q3_scores = np.array([v[1] for v in sc.values()])
        pids = list(sc.keys())

        r1 = m5_scores.max() - m5_scores.min()
        r2 = q3_scores.max() - q3_scores.min()
        n1 = np.ones_like(m5_scores) if r1 < 1e-9 else (m5_scores - m5_scores.min()) / r1
        n2 = np.ones_like(q3_scores) if r2 < 1e-9 else (q3_scores - q3_scores.min()) / r2

        combined = w_m5 * n1 + (1 - w_m5) * n2
        order = np.argsort(-combined)

        chosen = {pids[order[0]]}
        s_top = combined[order[0]]
        for i in order[1:]:
            if s_top - combined[i] < margin and len(chosen) < max_preds:
                chosen.add(pids[i])
            else:
                break
        preds[cid] = chosen
    return preds

print("\n  --- Fixed top-N ---")
for n in [1, 2, 3, 4, 5]:
    evaluate(reranker_topN(scores_v2, n), f"Reranker top-{n} (MonoT5-v2, w=0.8)", verbose=True)

print("\n  --- Dynamic margin (MonoT5-v2, w=0.8) ---")
for margin in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
    r = reranker_dynamic_margin(scores_v2, margin)
    avg_preds = np.mean([len(v) for v in r.values()])
    p, rc, f1 = evaluate(r, verbose=False)
    print(f"  margin={margin:.2f}  P={p:.4f} R={rc:.4f} F1={f1:.4f}  avg_preds={avg_preds:.2f}")

# ============================================================
# EXPERIMENT 4: Recall@K analysis — are gold paragraphs in the reranker top-K?
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 4: Recall@K of Reranker Ensemble (how many golds in top-K?)")
print("=" * 90)

def recall_at_k(scores_index, k, w_m5=0.8):
    found = 0
    total = 0
    cases_perfect = 0
    for cid in ALL_CASES:
        if cid not in scores_index:
            total += len(gold[cid])
            continue
        sc = scores_index[cid]
        m5_scores = np.array([v[0] for v in sc.values()])
        q3_scores = np.array([v[1] for v in sc.values()])
        pids = list(sc.keys())

        r1 = m5_scores.max() - m5_scores.min()
        r2 = q3_scores.max() - q3_scores.min()
        n1 = np.ones_like(m5_scores) if r1 < 1e-9 else (m5_scores - m5_scores.min()) / r1
        n2 = np.ones_like(q3_scores) if r2 < 1e-9 else (q3_scores - q3_scores.min()) / r2

        combined = w_m5 * n1 + (1 - w_m5) * n2
        order = np.argsort(-combined)
        topk = {pids[i] for i in order[:k]}

        g = gold[cid]
        hits = g & topk
        found += len(hits)
        total += len(g)
        if hits == g:
            cases_perfect += 1

    recall = found / total if total else 0
    print(f"  Recall@{k:3d}: {recall:.4f} ({found}/{total} paragraphs found, {cases_perfect}/100 cases fully covered)")

for k in [5, 10, 15, 20, 30, 50, 100]:
    recall_at_k(scores_v2, k)

# ============================================================
# EXPERIMENT 5: LLM + Reranker hybrid — use LLM pick + reranker top-N
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 5: Hybrid LLM + Reranker (augment LLM picks with reranker top-N)")
print("=" * 90)

def hybrid_llm_reranker(llm_run, scores_index, extra_n=2, w_m5=0.8):
    """Take LLM predictions + add reranker top-N if LLM predicted fewer than extra_n."""
    preds = {}
    reranker_top = reranker_topN(scores_index, extra_n + 1, w_m5)
    for cid in ALL_CASES:
        llm_preds = llm_run.get(cid, set())
        rr_preds = reranker_top.get(cid, set())
        # Always include LLM picks; fill up to extra_n+1 from reranker
        combined = set(llm_preds)
        for pid in sorted(rr_preds):  # add from reranker
            if len(combined) >= extra_n + 1:
                break
            combined.add(pid)
        preds[cid] = combined
    return preds

def llm_union_plus_reranker_fill(llm_runs, scores_index, target_n=3, w_m5=0.8):
    """Union of LLM runs, then fill remaining slots from reranker."""
    preds = {}
    reranker_ranked = {}
    for cid in ALL_CASES:
        sc = scores_index.get(cid, {})
        if not sc:
            reranker_ranked[cid] = []
            continue
        m5_scores = np.array([v[0] for v in sc.values()])
        q3_scores = np.array([v[1] for v in sc.values()])
        pids = list(sc.keys())
        r1 = m5_scores.max() - m5_scores.min()
        r2 = q3_scores.max() - q3_scores.min()
        n1 = np.ones_like(m5_scores) if r1 < 1e-9 else (m5_scores - m5_scores.min()) / r1
        n2 = np.ones_like(q3_scores) if r2 < 1e-9 else (q3_scores - q3_scores.min()) / r2
        combined = w_m5 * n1 + (1 - w_m5) * n2
        order = np.argsort(-combined)
        reranker_ranked[cid] = [pids[i] for i in order]

    for cid in ALL_CASES:
        llm_union = set()
        for run in llm_runs:
            llm_union |= run.get(cid, set())
        # Fill from reranker
        result = set(llm_union)
        for pid in reranker_ranked.get(cid, []):
            if len(result) >= target_n:
                break
            result.add(pid)
        preds[cid] = result
    return preds

print("\n  --- LLM pick + reranker fill (single LLM) ---")
for name, llm_run in [("V3_rag", v3_rag), ("R1_rag", r1_rag), ("V3_zero", v3_zero)]:
    for extra in [1, 2, 3, 4]:
        total_n = extra + 1
        r = hybrid_llm_reranker(llm_run, scores_v2, extra, 0.8)
        avg_preds = np.mean([len(v) for v in r.values()])
        p, rc, f1 = evaluate(r, verbose=False)
        print(f"  {name} + reranker fill to {total_n}:  P={p:.4f} R={rc:.4f} F1={f1:.4f}  avg={avg_preds:.1f}")

print("\n  --- Union of LLMs + reranker fill ---")
for target in [2, 3, 4, 5]:
    r = llm_union_plus_reranker_fill([v3_rag, r1_rag], scores_v2, target, 0.8)
    avg_preds = np.mean([len(v) for v in r.values()])
    p, rc, f1 = evaluate(r, verbose=False)
    print(f"  V3rag∪R1rag + fill to {target}:  P={p:.4f} R={rc:.4f} F1={f1:.4f}  avg={avg_preds:.1f}")

for target in [2, 3, 4, 5]:
    r = llm_union_plus_reranker_fill([v3_rag, r1_rag, llama_zero], scores_v2, target, 0.8)
    avg_preds = np.mean([len(v) for v in r.values()])
    p, rc, f1 = evaluate(r, verbose=False)
    print(f"  V3rag∪R1rag∪LLaMA + fill to {target}:  P={p:.4f} R={rc:.4f} F1={f1:.4f}  avg={avg_preds:.1f}")

for target in [2, 3, 4, 5]:
    r = llm_union_plus_reranker_fill([v3_rag, r1_rag, v3_zero, r1_zero, llama_zero], scores_v2, target, 0.8)
    avg_preds = np.mean([len(v) for v in r.values()])
    p, rc, f1 = evaluate(r, verbose=False)
    print(f"  All5LLM∪ + fill to {target}:  P={p:.4f} R={rc:.4f} F1={f1:.4f}  avg={avg_preds:.1f}")

# ============================================================
# EXPERIMENT 6: "None" case recovery — fill skipped cases with reranker
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 6: Recovery of 'None' Cases (fill skipped with reranker top-1)")
print("=" * 90)

def fill_none_cases(llm_run, scores_index, fill_n=1, w_m5=0.8):
    preds = {}
    reranker_top = reranker_topN(scores_index, fill_n, w_m5)
    for cid in ALL_CASES:
        if cid in llm_run and llm_run[cid]:
            preds[cid] = llm_run[cid]
        else:
            preds[cid] = reranker_top.get(cid, set())
    return preds

for name, run in [("DU1", du1), ("DU2", du2), ("DU3", du3),
                  ("V3_rag", v3_rag), ("R1_rag", r1_rag)]:
    skipped = sum(1 for c in ALL_CASES if c not in run or not run[c])
    filled = fill_none_cases(run, scores_v2, fill_n=1)
    p0, r0, f0 = evaluate(run, verbose=False)
    p1, r1_, f1_ = evaluate(filled, verbose=False)
    print(f"  {name:20s}  skipped={skipped:2d}  before: F1={f0:.4f}  after fill: F1={f1_:.4f}  Δ={f1_-f0:+.4f}")

# ============================================================
# EXPERIMENT 7: What if we had used top-all prompt? Simulate multi-select
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 7: Simulated Multi-Select — what if LLMs returned top-3?")
print("=" * 90)
print("  (Oracle: for each case, take all LLM-selected paras across all runs)")

# Oracle: best possible from all available LLM data
all_llm_runs = [v3_rag, r1_rag, v3_zero, r1_zero, llama_zero]
if qwq_zero: all_llm_runs.append(qwq_zero)
if llama_zero_alt: all_llm_runs.append(llama_zero_alt)

# Oracle: union of ALL LLM runs, filtered to gold-only
oracle_preds = {}
for cid in ALL_CASES:
    all_selected = set()
    for run in all_llm_runs:
        all_selected |= run.get(cid, set())
    oracle_preds[cid] = all_selected

evaluate(oracle_preds, "Oracle: union of ALL LLM runs (unfiltered)", verbose=True)

# Oracle filtered to only correct predictions
oracle_correct_only = {}
for cid in ALL_CASES:
    all_selected = set()
    for run in all_llm_runs:
        all_selected |= run.get(cid, set())
    oracle_correct_only[cid] = all_selected & gold[cid]

evaluate(oracle_correct_only, "Oracle: union of ALL LLM (only correct kept)", verbose=True)

# How many unique correct paragraphs did the LLMs find across all runs?
total_unique_correct = sum(len(v) for v in oracle_correct_only.values())
print(f"\n  Across all LLM runs, {total_unique_correct}/{TOTAL_RELEVANT} gold paragraphs were found by at least one model")
print(f"  That's {total_unique_correct/TOTAL_RELEVANT*100:.1f}% coverage of all gold paragraphs")

# ============================================================
# EXPERIMENT 8: Reranker weight sensitivity
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 8: Ensemble Weight Sensitivity (MonoT5 weight)")
print("=" * 90)

for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for n in [1, 2, 3]:
        p, r_, f1_ = evaluate(reranker_topN(scores_v2, n, w), verbose=False)
        print(f"  w_m5={w:.1f} top-{n}:  P={p:.4f} R={r_:.4f} F1={f1_:.4f}")
    print()

# ============================================================
# EXPERIMENT 9: Best achievable from our components (upper bound)
# ============================================================

print("=" * 90)
print("  EXPERIMENT 9: Upper Bounds & Comparative Analysis")
print("=" * 90)

# Perfect selection from reranker top-20
topk_preds = reranker_topN(scores_v2, 20)
oracle_from_top20 = {}
for cid in ALL_CASES:
    candidates = topk_preds.get(cid, set())
    oracle_from_top20[cid] = gold[cid] & candidates

evaluate(oracle_from_top20, "Oracle from reranker top-20 (perfect LLM selection)", verbose=True)

# Perfect selection from reranker top-100 (all BM25 candidates)
topk100_preds = reranker_topN(scores_v2, 100)
oracle_from_top100 = {}
for cid in ALL_CASES:
    candidates = topk100_preds.get(cid, set())
    oracle_from_top100[cid] = gold[cid] & candidates

evaluate(oracle_from_top100, "Oracle from reranker top-100 (perfect from full BM25 pool)", verbose=True)

# How many gold paragraphs are NOT even in BM25 top-100?
missed_by_bm25 = 0
missed_cases = []
for cid in ALL_CASES:
    candidates = topk100_preds.get(cid, set())
    missed = gold[cid] - candidates
    if missed:
        missed_by_bm25 += len(missed)
        missed_cases.append((cid, missed, gold[cid]))

print(f"\n  Gold paragraphs NOT in BM25 top-100: {missed_by_bm25}/{TOTAL_RELEVANT} ({missed_by_bm25/TOTAL_RELEVANT*100:.1f}%)")
if missed_cases:
    print(f"  Affected cases ({len(missed_cases)}):")
    for cid, missed, g in missed_cases:
        print(f"    Case {cid}: missed {sorted(missed)} (gold={sorted(g)})")

# ============================================================
# EXPERIMENT 10: Competition comparison context
# ============================================================

print("\n" + "=" * 90)
print("  EXPERIMENT 10: How our best post-hoc strategies compare to competition winner")
print("=" * 90)

winner_f1 = 0.4899
print(f"\n  Competition winner (IAI run2): F1 = {winner_f1:.4f}")
print()

best_strategies = []

# Test many combinations
configs = [
    ("DU1 (submitted)", du1),
    ("DU2 (submitted)", du2),
    ("DU3 (submitted)", du3),
    ("Reranker top-3", reranker_topN(scores_v2, 3)),
    ("V3rag ∪ R1rag", combine_runs([v3_rag, r1_rag], "union")),
    ("V3rag∪R1rag + fill to 3", llm_union_plus_reranker_fill([v3_rag, r1_rag], scores_v2, 3)),
    ("All5LLM ∪ fill to 3", llm_union_plus_reranker_fill([v3_rag, r1_rag, v3_zero, r1_zero, llama_zero], scores_v2, 3)),
    ("V3rag + fill to 3", hybrid_llm_reranker(v3_rag, scores_v2, 2)),
    ("Majority(V3r,R1r,LLaMA)", combine_runs([v3_rag, r1_rag, llama_zero], "majority")),
    ("DU2 + none-fill top1", fill_none_cases(du2, scores_v2, 1)),
]

for name, preds in configs:
    p, r_, f1_ = evaluate(preds, verbose=False)
    best_strategies.append((f1_, name, p, r_))

best_strategies.sort(reverse=True)
print(f"  {'Rank':>4s}  {'Strategy':55s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  {'vs Winner':>10s}")
print(f"  {'-'*4}  {'-'*55}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")
for i, (f1_, name, p, r_) in enumerate(best_strategies, 1):
    delta = f1_ - winner_f1
    marker = "BEATS!" if delta > 0 else ""
    print(f"  {i:4d}  {name:55s}  {p:.4f}  {r_:.4f}  {f1_:.4f}  {delta:+.4f} {marker}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("  SUMMARY: Key Numbers for the Paper")
print("=" * 90)

gold_dist = Counter(len(g) for g in gold.values())
print(f"\n  Test set: {len(gold)} cases, {TOTAL_RELEVANT} relevant paragraphs")
print(f"  Avg gold/case: {TOTAL_RELEVANT/len(gold):.2f}")
print(f"  Gold distribution: {dict(sorted(gold_dist.items()))}")
print(f"  Forced-top-1 ceiling: F1={2*1.0*(100/TOTAL_RELEVANT)/(1.0+100/TOTAL_RELEVANT):.4f}")
print(f"  Competition winner: F1={winner_f1}")
print(f"  DU2 precision rank: 2nd/35 (0.7529)")
print(f"  DU runs predicted: DU1={len(du1)}, DU2={len(du2)}, DU3={len(du3)} cases out of 100")
