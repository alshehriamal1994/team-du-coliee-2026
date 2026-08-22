#!/usr/bin/env python3
"""Diagnostic analysis of the Task 2 submissions."""
import json
import pickle
import numpy as np
from collections import defaultdict, Counter

# Load data
with open("../data/task2/task2_test_labels_2026.json") as f:
    raw_gold = json.load(f)
gold = {}
for cid, val in raw_gold.items():
    gold[cid] = {x.strip().replace(".txt", "").zfill(3) for x in val.split(",")}
ALL_CASES = sorted(gold.keys(), key=int)
TOTAL_RELEVANT = sum(len(g) for g in gold.values())

with open("cache/runs_final_2026/test_cache_monot5v2.pkl", "rb") as f:
    cache_v2 = pickle.load(f)

def build_score_index(cache):
    index = {}
    for row in cache['rows']:
        cid = row['cid']
        scores = {}
        for i, pid in enumerate(row['cand_ids']):
            scores[pid.zfill(3)] = (row['m5'][i], row['q3'][i])
        index[cid] = scores
    return index

scores_v2 = build_score_index(cache_v2)

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

v3_rag = load_run(BASE + "test2026_v3_rag_k20.txt")
r1_rag = load_run(BASE + "test2026_r1_rag_k20.txt")
v3_zero = load_run(BASE + "test2026_v3_forced_top1_k20.txt")
r1_zero = load_run(BASE + "test2026_r1_forced_top1_k20.txt")
llama_zero = load_run(BASE + "test2026_llama33_forced_top1_k20.txt")
du1 = load_run(FINAL + "DU1/task2_DU1.txt")
du2 = load_run(FINAL + "DU2/task2_DU2.txt")
du3 = load_run(FINAL + "DU3/task2_DU3.txt")

def evaluate(preds):
    correct = retrieved = 0
    for cid in ALL_CASES:
        g = gold[cid]
        p = preds.get(cid, set())
        correct += len(g & p)
        retrieved += len(p)
    micro_p = correct / retrieved if retrieved else 0
    micro_r = correct / TOTAL_RELEVANT
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) else 0
    return micro_p, micro_r, micro_f1, correct, retrieved

# A. Dev vs test gold distribution
print("  A. Dev vs test gold distribution")
# Dev set had 100 cases, 122 relevant = 1.22 avg
# Test set has 100 cases, 294 relevant = 2.94 avg
print("""
  Development set:
    - 100 cases, 122 relevant paragraphs, 1.22 gold per case
    - Most cases had exactly 1 gold paragraph
    - Forced-top-1 maximised F1 under this distribution

  Test set:
    - 100 cases, 294 relevant paragraphs, 2.94 gold per case
    - Only 5 cases (5%) had 1 gold paragraph
    - 43 cases (43%) had exactly 3 gold paragraphs
    - 25 cases (25%) had 4-6 gold paragraphs

  The gold distribution differs between development and test. The development
  set rewarded precision; the test set rewards recall.
""")

# B. Decomposing the F1 gap
print("  B. Decomposing the F1 gap between DU2 and the winner")

# DU2: P=0.7529, R=0.2177, F1=0.3377
# IAI:  P=0.4501, R=0.5374, F1=0.4899
print("""
  DU2 (best DU run): P=0.7529  R=0.2177  F1=0.3377  (64 correct / 85 predicted)
  IAI (winner):      P=0.4501  R=0.5374  F1=0.4899

  The gap is in recall, not precision. DU2 leads on precision by +0.30 and
  trails on recall by -0.32.

  In concrete numbers:
    - DU2 identified 64 correct paragraphs
    - The winner identified about 158 correct paragraphs (0.5374 x 294)
    - That is 94 more correct paragraphs than DU2
    - The winner retrieved about 351 predictions (158/0.4501)
    - So the winner predicted about 3.5 paragraphs per case against DU2's 0.85

  The winner traded precision for recall, which at 2.94 gold per case is optimal.
""")

# C. Case-by-case categories
print("  C. Case-by-case diagnostic: three categories of DU2 cases")

correct_cases = []  # Predicted correctly (hit at least 1 gold)
wrong_cases = []    # Predicted but wrong
skipped_cases = []  # Returned "none"

for cid in ALL_CASES:
    p = du2.get(cid, set())
    g = gold[cid]
    if not p:
        skipped_cases.append(cid)
    elif p & g:
        correct_cases.append(cid)
    else:
        wrong_cases.append(cid)

print(f"  Correct predictions (hit at least 1 gold): {len(correct_cases)} cases")
print(f"  Wrong predictions (missed all gold):        {len(wrong_cases)} cases")
print(f"  Skipped ('none' returned):                  {len(skipped_cases)} cases")
print()

# For correct cases: how many gold did we miss?
correct_missed = 0
correct_total_gold = 0
for cid in correct_cases:
    g = gold[cid]
    correct_total_gold += len(g)
    correct_missed += len(g) - 1  # one predicted and correct, the rest missed
print(f"  In the {len(correct_cases)} correct cases:")
print(f"    Total gold paragraphs: {correct_total_gold}")
print(f"    Found: {len(correct_cases)} (one per case)")
print(f"    Missed: {correct_missed} gold paragraphs (forced-top-1)")
print(f"    Even when correct, only {len(correct_cases)/correct_total_gold*100:.1f}% of the gold is captured")
print()

# For skipped cases: gold paragraphs not attempted
skipped_gold = sum(len(gold[c]) for c in skipped_cases)
print(f"  In the {len(skipped_cases)} skipped cases:")
print(f"    Gold paragraphs not attempted: {skipped_gold}")
print(f"    These contribute nothing to recall or precision")
for cid in skipped_cases:
    print(f"    Case {cid}: {len(gold[cid])} gold paragraphs not attempted")
print()

# For wrong cases: distance from gold
print(f"  In the {len(wrong_cases)} wrong cases:")
wrong_gold = sum(len(gold[c]) for c in wrong_cases)
print(f"    Gold paragraphs: {wrong_gold}")
for cid in wrong_cases:
    p = sorted(du2.get(cid, set()))
    g = sorted(gold[cid])
    # Check if prediction is near gold
    near = False
    for pid in p:
        for gid in g:
            if abs(int(pid) - int(gid)) <= 2:
                near = True
    marker = " [near-miss]" if near else ""
    print(f"    Case {cid}: predicted={p} gold={g}{marker}")

# D. Where gold paragraphs rank in the reranker ensemble
print()
print("  D. Where gold paragraphs rank in the reranker ensemble")

gold_ranks = []
gold_rank_dist = Counter()
for cid in ALL_CASES:
    if cid not in scores_v2:
        continue
    sc = scores_v2[cid]
    m5 = np.array([v[0] for v in sc.values()])
    q3 = np.array([v[1] for v in sc.values()])
    pids = list(sc.keys())
    r1 = m5.max() - m5.min()
    r2 = q3.max() - q3.min()
    n1 = np.ones_like(m5) if r1 < 1e-9 else (m5 - m5.min()) / r1
    n2 = np.ones_like(q3) if r2 < 1e-9 else (q3 - q3.min()) / r2
    combined = 0.8 * n1 + 0.2 * n2
    order = np.argsort(-combined)
    pid_to_rank = {pids[order[i]]: i+1 for i in range(len(order))}

    for gpid in gold[cid]:
        if gpid in pid_to_rank:
            rank = pid_to_rank[gpid]
            gold_ranks.append(rank)
            if rank <= 3:
                gold_rank_dist['top-3'] += 1
            elif rank <= 5:
                gold_rank_dist['4-5'] += 1
            elif rank <= 10:
                gold_rank_dist['6-10'] += 1
            elif rank <= 20:
                gold_rank_dist['11-20'] += 1
            elif rank <= 50:
                gold_rank_dist['21-50'] += 1
            else:
                gold_rank_dist['51+'] += 1
        else:
            gold_ranks.append(999)
            gold_rank_dist['not in BM25'] += 1

print(f"\n  Rank of the {TOTAL_RELEVANT} gold paragraphs in the reranker output")
print(f"  (MonoT5-v2 + Qwen3, w=0.8)")
print()
for bucket in ['top-3', '4-5', '6-10', '11-20', '21-50', '51+', 'not in BM25']:
    count = gold_rank_dist.get(bucket, 0)
    pct = count / TOTAL_RELEVANT * 100
    print(f"    {bucket:>12s}: {count:3d} ({pct:5.1f}%)")

print(f"\n  Median gold paragraph rank: {int(np.median(gold_ranks))}")
print(f"  Mean gold paragraph rank:   {np.mean(gold_ranks):.1f}")

# E. The 'none' cases DU2 skipped
print()
print("  E. The 'none' cases DU2 skipped")

# Check what the other models said for these skipped cases
print(f"\n  DU2 skipped {len(skipped_cases)} cases. Other model predictions:")
print(f"  {'Case':>6s}  {'V3_rag':>8s}  {'R1_rag':>8s}  {'V3_zero':>8s}  {'R1_zero':>8s}  {'LLaMA':>8s}  {'Gold':>20s}  {'Any_hit':>8s}")

total_recoverable = 0
for cid in skipped_cases:
    v3r = sorted(v3_rag.get(cid, set())) or ['none']
    r1r = sorted(r1_rag.get(cid, set())) or ['none']
    v3z = sorted(v3_zero.get(cid, set())) or ['none']
    r1z = sorted(r1_zero.get(cid, set())) or ['none']
    llm = sorted(llama_zero.get(cid, set())) or ['none']
    g = sorted(gold[cid])

    any_correct = False
    for run in [v3_rag, r1_rag, v3_zero, r1_zero, llama_zero]:
        if run.get(cid, set()) & gold[cid]:
            any_correct = True
            break

    if any_correct:
        total_recoverable += 1

    print(f"  {cid:>6s}  {','.join(v3r):>8s}  {','.join(r1r):>8s}  {','.join(v3z):>8s}  {','.join(r1z):>8s}  {','.join(llm):>8s}  {','.join(g):>20s}  {'YES' if any_correct else 'no':>8s}")

print(f"\n  Of {len(skipped_cases)} skipped cases, {total_recoverable} had a correct answer from another model")
print(f"  A fallback to any other model would recover these")

# F. Confidence vs correctness
print()
print("  F. Confidence vs correctness: do high reranker scores predict gold?")

# For each case, get the ensemble score of the top-1 candidate
# and check if it's gold
top1_scores_correct = []
top1_scores_wrong = []
for cid in ALL_CASES:
    if cid not in scores_v2:
        continue
    sc = scores_v2[cid]
    m5 = np.array([v[0] for v in sc.values()])
    q3 = np.array([v[1] for v in sc.values()])
    pids = list(sc.keys())
    r1_ = m5.max() - m5.min()
    r2_ = q3.max() - q3.min()
    n1 = np.ones_like(m5) if r1_ < 1e-9 else (m5 - m5.min()) / r1_
    n2 = np.ones_like(q3) if r2_ < 1e-9 else (q3 - q3.min()) / r2_
    combined = 0.8 * n1 + 0.2 * n2
    order = np.argsort(-combined)

    top1_pid = pids[order[0]]
    top1_score = combined[order[0]]
    # Gap between top-1 and top-2
    gap = combined[order[0]] - combined[order[1]] if len(order) > 1 else 0

    if top1_pid in gold[cid]:
        top1_scores_correct.append((top1_score, gap))
    else:
        top1_scores_wrong.append((top1_score, gap))

print(f"\n  Reranker top-1 is correct in {len(top1_scores_correct)} cases, wrong in {len(top1_scores_wrong)} cases")
print(f"  Reranker top-1 accuracy: {len(top1_scores_correct)/100*100:.1f}%")
print(f"\n  When correct:  avg_score={np.mean([s for s,g in top1_scores_correct]):.4f}  avg_gap={np.mean([g for s,g in top1_scores_correct]):.4f}")
print(f"  When wrong:    avg_score={np.mean([s for s,g in top1_scores_wrong]):.4f}  avg_gap={np.mean([g for s,g in top1_scores_wrong]):.4f}")

# G. Predicting 3 paragraphs per case
print()
print("  G. Predicting 3 paragraphs per case")

# Strategy: R1_rag pick + fill to 3 from reranker
def get_reranker_ranked(cid, w_m5=0.8):
    if cid not in scores_v2:
        return []
    sc = scores_v2[cid]
    m5 = np.array([v[0] for v in sc.values()])
    q3 = np.array([v[1] for v in sc.values()])
    pids = list(sc.keys())
    r1_ = m5.max() - m5.min()
    r2_ = q3.max() - q3.min()
    n1 = np.ones_like(m5) if r1_ < 1e-9 else (m5 - m5.min()) / r1_
    n2 = np.ones_like(q3) if r2_ < 1e-9 else (q3 - q3.min()) / r2_
    combined = w_m5 * n1 + (1 - w_m5) * n2
    order = np.argsort(-combined)
    return [(pids[i], combined[i]) for i in order]

# Build best post-hoc strategy
strategies = {}

# Strategy A: LLM pick + reranker fill to N
for target_n in [2, 3, 4]:
    for llm_name, llm_run in [("V3_rag", v3_rag), ("R1_rag", r1_rag),
                                ("V3_zero", v3_zero), ("R1_zero", r1_zero)]:
        preds = {}
        for cid in ALL_CASES:
            llm_pids = llm_run.get(cid, set())
            ranked = get_reranker_ranked(cid)
            result = set(llm_pids)
            for pid, score in ranked:
                if len(result) >= target_n:
                    break
                result.add(pid)
            preds[cid] = result
        p, r, f1, c, ret = evaluate(preds)
        strategies[f"{llm_name} + fill to {target_n}"] = (f1, p, r, c, ret)

# Strategy B: Union of 2 LLMs + fill
for target_n in [2, 3, 4]:
    for combo_name, runs in [
        ("V3r+R1r", [v3_rag, r1_rag]),
        ("V3z+R1z", [v3_zero, r1_zero]),
        ("V3r+R1z", [v3_rag, r1_zero]),
        ("V3z+R1r", [v3_zero, r1_rag]),
        ("V3r+R1r+LL", [v3_rag, r1_rag, llama_zero]),
    ]:
        preds = {}
        for cid in ALL_CASES:
            union = set()
            for run in runs:
                union |= run.get(cid, set())
            ranked = get_reranker_ranked(cid)
            result = set(union)
            for pid, score in ranked:
                if len(result) >= target_n:
                    break
                result.add(pid)
            preds[cid] = result
        p, r, f1, c, ret = evaluate(preds)
        strategies[f"{combo_name} + fill to {target_n}"] = (f1, p, r, c, ret)

# Strategy C: Pure reranker
for n in [2, 3, 4]:
    preds = {}
    for cid in ALL_CASES:
        ranked = get_reranker_ranked(cid)
        preds[cid] = {pid for pid, sc in ranked[:n]}
    p, r, f1, c, ret = evaluate(preds)
    strategies[f"Pure reranker top-{n}"] = (f1, p, r, c, ret)

# Strategy D: adaptive, use score gap to decide how many
for gap_thresh in [0.15, 0.20, 0.25, 0.30]:
    preds = {}
    for cid in ALL_CASES:
        ranked = get_reranker_ranked(cid)
        if not ranked:
            continue
        result = {ranked[0][0]}
        top_score = ranked[0][1]
        for pid, sc in ranked[1:5]:  # consider up to top-5
            if top_score - sc < gap_thresh:
                result.add(pid)
            else:
                break
        preds[cid] = result
    p, r, f1, c, ret = evaluate(preds)
    avg_n = np.mean([len(v) for v in preds.values()])
    strategies[f"Adaptive gap<{gap_thresh:.2f} (avg={avg_n:.1f})"] = (f1, p, r, c, ret)

# Sort and display
print(f"\n  {'Strategy':50s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  {'Corr':>5s}  {'Ret':>5s}")
print(f"  {'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")

winner_f1 = 0.4899
sorted_strats = sorted(strategies.items(), key=lambda x: -x[1][0])
for name, (f1, p, r, c, ret) in sorted_strats[:25]:
    delta = f1 - winner_f1
    marker = " [beats]" if delta >= 0 else ""
    print(f"  {name:50s}  {p:.4f}  {r:.4f}  {f1:.4f}  {c:5d}  {ret:5d}{marker}")

print(f"\n  Competition winner (IAI run2):                       0.4501  0.5374  0.4899")

# H. Step-by-step improvements
print()
print("  H. Step-by-step: how each improvement adds up")

# Start from DU2 and improve incrementally
print(f"\n  Starting point: DU2 (submitted)")
p, r, f1, c, ret = evaluate(du2)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

# Step 1: Fill "none" cases with reranker top-1
step1 = {}
reranker_top1 = {}
for cid in ALL_CASES:
    ranked = get_reranker_ranked(cid)
    reranker_top1[cid] = {ranked[0][0]} if ranked else set()

for cid in ALL_CASES:
    if cid in du2 and du2[cid]:
        step1[cid] = du2[cid]
    else:
        step1[cid] = reranker_top1[cid]

print(f"\n  Step 1: Fill 15 'none' cases with reranker top-1")
p, r, f1, c, ret = evaluate(step1)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

# Step 2: Add reranker top-2 to all cases
step2 = {}
for cid in ALL_CASES:
    ranked = get_reranker_ranked(cid)
    existing = step1.get(cid, set())
    result = set(existing)
    for pid, sc in ranked:
        if len(result) >= 2:
            break
        result.add(pid)
    step2[cid] = result

print(f"\n  Step 2: Ensure at least 2 predictions per case (reranker fill)")
p, r, f1, c, ret = evaluate(step2)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

# Step 3: Add reranker top-3
step3 = {}
for cid in ALL_CASES:
    ranked = get_reranker_ranked(cid)
    existing = step1.get(cid, set())
    result = set(existing)
    for pid, sc in ranked:
        if len(result) >= 3:
            break
        result.add(pid)
    step3[cid] = result

print(f"\n  Step 3: Ensure at least 3 predictions per case (reranker fill)")
p, r, f1, c, ret = evaluate(step3)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

# Step 4: Use R1_rag instead of DU2's V3_rag
step4_base = {}
for cid in ALL_CASES:
    if cid in r1_rag and r1_rag[cid]:
        step4_base[cid] = r1_rag[cid]
    else:
        step4_base[cid] = reranker_top1[cid]

step4 = {}
for cid in ALL_CASES:
    ranked = get_reranker_ranked(cid)
    existing = step4_base.get(cid, set())
    result = set(existing)
    for pid, sc in ranked:
        if len(result) >= 3:
            break
        result.add(pid)
    step4[cid] = result

print(f"\n  Step 4: Use R1_rag (instead of V3_rag) + fill to 3")
p, r, f1, c, ret = evaluate(step4)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

# Step 5: Use union of V3_rag + R1_rag
step5_base = {}
for cid in ALL_CASES:
    union = v3_rag.get(cid, set()) | r1_rag.get(cid, set())
    step5_base[cid] = union if union else reranker_top1[cid]

step5 = {}
for cid in ALL_CASES:
    ranked = get_reranker_ranked(cid)
    existing = step5_base.get(cid, set())
    result = set(existing)
    for pid, sc in ranked:
        if len(result) >= 3:
            break
        result.add(pid)
    step5[cid] = result

print(f"\n  Step 5: Union V3_rag and R1_rag + fill to 3")
p, r, f1, c, ret = evaluate(step5)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

# Step 6: Use ALL LLMs union + fill to 3
step6_base = {}
for cid in ALL_CASES:
    union = set()
    for run in [v3_rag, r1_rag, v3_zero, r1_zero, llama_zero]:
        union |= run.get(cid, set())
    step6_base[cid] = union if union else reranker_top1[cid]

step6 = {}
for cid in ALL_CASES:
    ranked = get_reranker_ranked(cid)
    existing = step6_base.get(cid, set())
    result = set(existing)
    for pid, sc in ranked:
        if len(result) >= 3:
            break
        result.add(pid)
    step6[cid] = result

print(f"\n  Step 6: Union ALL 5 LLMs + fill to 3")
p, r, f1, c, ret = evaluate(step6)
print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (correct={c}, retrieved={ret})")

print(f"\n  Competition winner: P=0.4501  R=0.5374  F1=0.4899")

# I. Requirements to beat the winner
print()
print("  I. What it would take to beat the winner")

# If we keep P=0.75 (our strength), what R do we need?
for target_f1 in [0.49, 0.50, 0.55]:
    # F1 = 2PR/(P+R), so R = F1*P / (2P - F1)
    P = 0.75
    R_needed = target_f1 * P / (2 * P - target_f1)
    preds_needed = R_needed * TOTAL_RELEVANT
    print(f"  To reach F1={target_f1:.2f} at P=0.75: need R>={R_needed:.4f}, {preds_needed:.0f} correct paragraphs")

print()
# If we keep P=0.50 (more realistic for multi-select), what R?
for target_f1 in [0.49, 0.50, 0.55]:
    P = 0.50
    R_needed = target_f1 * P / (2 * P - target_f1)
    preds_needed = R_needed * TOTAL_RELEVANT
    print(f"  To reach F1={target_f1:.2f} at P=0.50: need R>={R_needed:.4f}, {preds_needed:.0f} correct paragraphs")

print()
# How many correct do we need at different prediction levels?
print(f"  If predicting ~3 per case (300 total):")
for target_f1 in [0.49, 0.50, 0.55]:
    # F1 = 2*c/300 * c/294 / (c/300 + c/294) = 2c^2/(300*294) / (c*(300+294)/(300*294))
    # F1 = 2c / (300+294) = 2c/594
    c_needed = target_f1 * 594 / 2
    print(f"    F1={target_f1:.2f} needs {c_needed:.0f} correct out of 300 predictions ({c_needed/300*100:.1f}% precision)")
