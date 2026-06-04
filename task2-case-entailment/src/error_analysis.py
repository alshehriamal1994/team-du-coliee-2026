#!/usr/bin/env python3
"""Deep analysis of DU Task 2 submissions against gold labels."""
import json
import sys
from collections import Counter, defaultdict

# --- Load gold labels ---
with open("../data/task2/task2_test_labels_2026.json") as f:
    raw = json.load(f)

gold = {}
for cid, val in raw.items():
    pids = {x.strip().replace(".txt", "").lstrip("0") or "0" for x in val.split(",")}
    # normalize to 3-digit
    pids_norm = {x.zfill(3) for x in pids}
    gold[cid] = pids_norm

# --- Load predictions ---
def load_run(path):
    preds = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid = parts[0]
                pid = parts[1].zfill(3)
                preds[qid].add(pid)
    return dict(preds)

du1 = load_run("predictions/DU1/task2_DU1.txt")
du2 = load_run("predictions/DU2/task2_DU2.txt")
du3 = load_run("predictions/DU3/task2_DU3.txt")

all_cases = sorted(gold.keys(), key=int)

# --- Compute metrics ---
def evaluate(preds, name):
    correct = 0
    retrieved = 0
    relevant = 0
    per_case = {}

    for cid in all_cases:
        g = gold[cid]
        p = preds.get(cid, set())
        hits = g & p
        correct += len(hits)
        retrieved += len(p)
        relevant += len(g)

        # per-case F1
        prec = len(hits) / len(p) if p else 0
        rec = len(hits) / len(g) if g else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        per_case[cid] = {
            'gold': g, 'pred': p, 'hits': hits,
            'prec': prec, 'rec': rec, 'f1': f1,
            'n_gold': len(g), 'n_pred': len(p)
        }

    micro_p = correct / retrieved if retrieved else 0
    micro_r = correct / relevant if relevant else 0
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) else 0

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Correct: {correct}  Retrieved: {retrieved}  Relevant: {relevant}")
    print(f"  Micro-P: {micro_p:.4f}  Micro-R: {micro_r:.4f}  Micro-F1: {micro_f1:.4f}")
    print(f"  Cases predicted: {len(preds)}/100  Cases skipped: {100-len(preds)}")

    return per_case

pc1 = evaluate(du1, "DU1 (Legal-RAG, V3+R1 intersection, MonoT5-v1)")
pc2 = evaluate(du2, "DU2 (Legal-RAG, V3 only, MonoT5-v2)")
pc3 = evaluate(du3, "DU3 (Zero-shot, V3+R1+LLaMA tiebreak, MonoT5-v1)")

# --- Gold label distribution ---
print(f"\n{'='*70}")
print(f"  GOLD LABEL DISTRIBUTION (Test Set)")
print(f"{'='*70}")
gold_counts = [len(g) for g in gold.values()]
total_relevant = sum(gold_counts)
dist = Counter(gold_counts)
print(f"  Total cases: {len(gold)}")
print(f"  Total relevant paragraphs: {total_relevant}")
print(f"  Avg gold per case: {total_relevant/len(gold):.2f}")
print(f"  Distribution of #gold per case:")
for k in sorted(dist.keys()):
    pct = dist[k]/len(gold)*100
    print(f"    {k} gold paragraphs: {dist[k]} cases ({pct:.1f}%)")

# --- The forced-top-1 ceiling ---
print(f"\n{'='*70}")
print(f"  FORCED-TOP-1 THEORETICAL CEILING")
print(f"{'='*70}")
# If you predict exactly 1 correct para per case (perfect selection), what's your max?
# For case with N gold: you get 1 correct, precision=1.0, recall=1/N
# Total: correct=100, retrieved=100, relevant=sum(gold_counts)
perfect_top1_correct = len(gold)  # 100
perfect_top1_retrieved = len(gold)  # 100
perfect_top1_relevant = total_relevant
p_ceil = perfect_top1_correct / perfect_top1_retrieved
r_ceil = perfect_top1_correct / perfect_top1_relevant
f1_ceil = 2*p_ceil*r_ceil/(p_ceil+r_ceil)
print(f"  If EVERY prediction is correct (1 per case):")
print(f"  Precision: {p_ceil:.4f}  Recall: {r_ceil:.4f}  F1: {f1_ceil:.4f}")
print(f"  >>> Max possible F1 with forced-top-1: {f1_ceil:.4f}")

# --- Cases with 1 gold only ---
single_gold = [cid for cid in all_cases if len(gold[cid]) == 1]
multi_gold = [cid for cid in all_cases if len(gold[cid]) > 1]
print(f"\n  Single-gold cases: {len(single_gold)}")
print(f"  Multi-gold cases: {len(multi_gold)}")

# --- Performance on single vs multi gold ---
print(f"\n{'='*70}")
print(f"  SINGLE vs MULTI-GOLD PERFORMANCE")
print(f"{'='*70}")

for name, pc in [("DU1", pc1), ("DU2", pc2), ("DU3", pc3)]:
    s_correct = sum(1 for cid in single_gold if pc[cid]['hits'])
    s_total = len(single_gold)
    m_correct = sum(len(pc[cid]['hits']) for cid in multi_gold)
    m_retrieved = sum(pc[cid]['n_pred'] for cid in multi_gold)
    m_relevant = sum(pc[cid]['n_gold'] for cid in multi_gold)
    m_cases_hit = sum(1 for cid in multi_gold if pc[cid]['hits'])

    print(f"\n  {name}:")
    print(f"    Single-gold: {s_correct}/{s_total} correct ({s_correct/s_total*100:.1f}% accuracy)")
    print(f"    Multi-gold:  {m_correct} hits from {m_retrieved} preds (of {m_relevant} relevant)")
    print(f"    Multi-gold:  {m_cases_hit}/{len(multi_gold)} cases with ≥1 hit ({m_cases_hit/len(multi_gold)*100:.1f}%)")

# --- Error analysis: where all 3 runs fail ---
print(f"\n{'='*70}")
print(f"  ERROR ANALYSIS")
print(f"{'='*70}")

all_fail = []
all_correct = []
for cid in all_cases:
    h1 = bool(pc1[cid]['hits'])
    h2 = bool(pc2[cid]['hits'])
    h3 = bool(pc3[cid]['hits'])
    if not h1 and not h2 and not h3:
        all_fail.append(cid)
    if h1 and h2 and h3:
        all_correct.append(cid)

print(f"  All 3 runs correct: {len(all_correct)} cases")
print(f"  All 3 runs wrong:   {len(all_fail)} cases")

# Show the failing cases
print(f"\n  Cases where ALL runs failed (case_id: #gold, DU1_pred, DU2_pred, DU3_pred → gold):")
for cid in all_fail:
    g = sorted(gold[cid])
    p1 = sorted(du1.get(cid, set()))
    p2 = sorted(du2.get(cid, set()))
    p3 = sorted(du3.get(cid, set()))
    print(f"    {cid}: gold={g}, DU1={p1}, DU2={p2}, DU3={p3}")

# --- Overlap analysis between runs ---
print(f"\n{'='*70}")
print(f"  RUN OVERLAP ANALYSIS")
print(f"{'='*70}")

for cid in all_cases:
    p1 = du1.get(cid, set())
    p2 = du2.get(cid, set())
    p3 = du3.get(cid, set())

agree_12 = sum(1 for c in all_cases if du1.get(c, set()) == du2.get(c, set()))
agree_13 = sum(1 for c in all_cases if du1.get(c, set()) == du3.get(c, set()))
agree_23 = sum(1 for c in all_cases if du2.get(c, set()) == du3.get(c, set()))
agree_all = sum(1 for c in all_cases if du1.get(c, set()) == du2.get(c, set()) == du3.get(c, set()))
print(f"  DU1==DU2 agreement: {agree_12}/100")
print(f"  DU1==DU3 agreement: {agree_13}/100")
print(f"  DU2==DU3 agreement: {agree_23}/100")
print(f"  All 3 agree:        {agree_all}/100")

# --- Cases where runs disagree and one is right ---
print(f"\n  Cases where runs DISAGREE and correctness differs:")
disagree_interesting = []
for cid in all_cases:
    h1 = bool(pc1[cid]['hits'])
    h2 = bool(pc2[cid]['hits'])
    h3 = bool(pc3[cid]['hits'])
    if h1 != h2 or h2 != h3:
        status = f"DU1={'✓' if h1 else '✗'} DU2={'✓' if h2 else '✗'} DU3={'✓' if h3 else '✗'}"
        disagree_interesting.append((cid, status))

print(f"  {len(disagree_interesting)} cases with different correctness:")
for cid, status in disagree_interesting:
    g = sorted(gold[cid])
    p1 = sorted(du1.get(cid, set()))
    p2 = sorted(du2.get(cid, set()))
    p3 = sorted(du3.get(cid, set()))
    print(f"    {cid}: {status}  gold={g}, DU1={p1}, DU2={p2}, DU3={p3}")

# --- Oracle analysis: union of all runs ---
print(f"\n{'='*70}")
print(f"  ORACLE / UPPER-BOUND ANALYSIS")
print(f"{'='*70}")

# Union oracle
union_correct = 0
union_retrieved = 0
for cid in all_cases:
    p_union = du1.get(cid, set()) | du2.get(cid, set()) | du3.get(cid, set())
    hits = gold[cid] & p_union
    union_correct += len(hits)
    union_retrieved += len(p_union)

union_p = union_correct / union_retrieved if union_retrieved else 0
union_r = union_correct / total_relevant
union_f1 = 2*union_p*union_r/(union_p+union_r) if (union_p+union_r) else 0
print(f"  Union of DU1∪DU2∪DU3:")
print(f"    Correct: {union_correct}  Retrieved: {union_retrieved}  Relevant: {total_relevant}")
print(f"    P: {union_p:.4f}  R: {union_r:.4f}  F1: {union_f1:.4f}")

# Best-per-case oracle
oracle_correct = 0
oracle_retrieved = 0
for cid in all_cases:
    best_hits = 0
    best_pred = set()
    for preds in [du1, du2, du3]:
        p = preds.get(cid, set())
        h = len(gold[cid] & p)
        if h > best_hits or (h == best_hits and len(p) < len(best_pred)):
            best_hits = h
            best_pred = p
    oracle_correct += best_hits
    oracle_retrieved += len(best_pred)

oracle_p = oracle_correct / oracle_retrieved if oracle_retrieved else 0
oracle_r = oracle_correct / total_relevant
oracle_f1 = 2*oracle_p*oracle_r/(oracle_p+oracle_r) if (oracle_p+oracle_r) else 0
print(f"\n  Best-per-case oracle (pick best run per case):")
print(f"    Correct: {oracle_correct}  Retrieved: {oracle_retrieved}  Relevant: {total_relevant}")
print(f"    P: {oracle_p:.4f}  R: {oracle_r:.4f}  F1: {oracle_f1:.4f}")

# --- What if we predicted ALL top-1 reranker outputs? ---
# Approximate: the cases where we predicted nothing
no_pred_du1 = set(all_cases) - set(du1.keys())
no_pred_du2 = set(all_cases) - set(du2.keys())
no_pred_du3 = set(all_cases) - set(du3.keys())
print(f"\n  Cases with NO prediction:")
print(f"    DU1: {sorted(no_pred_du1, key=int)} ({len(no_pred_du1)} cases)")
print(f"    DU2: {sorted(no_pred_du2, key=int)} ({len(no_pred_du2)} cases)")
print(f"    DU3: {sorted(no_pred_du3, key=int)} ({len(no_pred_du3)} cases)")

# How many relevant paragraphs were in skipped cases?
for name, no_pred in [("DU1", no_pred_du1), ("DU2", no_pred_du2), ("DU3", no_pred_du3)]:
    missed_rel = sum(len(gold[c]) for c in no_pred)
    print(f"    {name} skipped {len(no_pred)} cases with {missed_rel} relevant paragraphs")

# --- DU1 has 2 predictions for case 1016 ---
print(f"\n{'='*70}")
print(f"  MULTI-PREDICTION CASES")
print(f"{'='*70}")
for name, preds in [("DU1", du1), ("DU2", du2), ("DU3", du3)]:
    multi = {c: p for c, p in preds.items() if len(p) > 1}
    if multi:
        for c, p in multi.items():
            hits = gold[c] & p
            print(f"  {name} case {c}: predicted {sorted(p)}, gold={sorted(gold[c])}, hits={sorted(hits)}")
    else:
        print(f"  {name}: all single-prediction")

# --- Paragraph ID analysis: are we even in the right ballpark? ---
print(f"\n{'='*70}")
print(f"  NEAR-MISS ANALYSIS (predicted adjacent paragraph)")
print(f"{'='*70}")
for name, preds in [("DU1", du1), ("DU2", du2), ("DU3", du3)]:
    near_misses = 0
    for cid in all_cases:
        p = preds.get(cid, set())
        g = gold[cid]
        for pid in p:
            if pid not in g:
                pid_int = int(pid)
                # Check if adjacent paragraph is gold
                neighbors = {str(pid_int-1).zfill(3), str(pid_int+1).zfill(3),
                             str(pid_int-2).zfill(3), str(pid_int+2).zfill(3)}
                if neighbors & g:
                    near_misses += 1
                    print(f"    {name} case {cid}: predicted {pid}, near gold {sorted(g & neighbors)}")
    print(f"  {name} total near-misses (within ±2): {near_misses}")
    print()

print(f"\n{'='*70}")
print(f"  COMPETITION PRECISION RANKING")
print(f"{'='*70}")
teams = [
    ("NOWJ nowj001", 0.7604),
    ("DU DU2", 0.7529),
    ("ClaUSurf cls4b27bp11", 0.7400),
    ("NOWJ nowj003", 0.7037),
    ("NOWJ nowj002", 0.7034),
    ("DU DU3", 0.6907),
    ("JUNLLP sub1", 0.6721),
    ("JUNLLP sub2", 0.6721),
    ("DU DU1", 0.6632),
    ("ClaUSurf clsop3d", 0.6357),
    ("74688 ualbanyllm", 0.6500),
    ("CityUMO", 0.6000),
]
teams.sort(key=lambda x: -x[1])
for i, (t, p) in enumerate(teams, 1):
    marker = " <<<" if "DU" in t else ""
    print(f"  {i:2d}. {t:30s}  Precision = {p:.4f}{marker}")
