"""Experiment A - Task 1 protocol-ceiling and per-cardinality decomposition under fixed k=5.

Reproduces every number in the M2 claim ledger section "expA". Run from anywhere:
    python3 expA_t1_cardinality.py
Writes expA_numbers.json next to this file. 2026-07-21.
"""
import json, collections
from pathlib import Path
import os

ROOT = os.environ.get("COLIEE_ROOT", "data")

T1 = Path(os.path.join(ROOT, "TASK1/FINAL_SUBMISSION"))
OUT = Path(__file__).with_name('expA_numbers.json')

gold = {q: set(v) for q, v in json.load(open(T1 / 'task1_test_labels_2026.json')).items()}
counts = [len(v) for v in gold.values()]
n_q, total_gold = len(gold), sum(counts)

res = {'meta': {'date': '2026-07-21', 'queries': n_q, 'total_gold': total_gold,
                'avg_gold': round(total_gold / n_q, 3)},
       'gold_count_distribution': dict(sorted(collections.Counter(counts).items())),
       'queries_gt5': sum(1 for c in counts if c > 5),
       'gold_unreachable_at_k5': sum(c - 5 for c in counts if c > 5)}

ceil = {}
for k in (3, 4, 5, 6, 7, 8, 10):
    tp = sum(min(k, c) for c in counts)
    P, R = tp / (k * n_q), tp / total_gold
    ceil[f'k{k}'] = {'max_tp': tp, 'P': round(P, 4), 'R': round(R, 4),
                     'F1': round(2 * P * R / (P + R), 4)}
res['perfect_system_ceiling'] = ceil

# Per-cardinality decomposition of the submitted DU3 run
preds = collections.defaultdict(set)
for line in open(T1 / 'DU3.txt'):
    q, c, _ = line.split()
    preds[q + '.txt' if (q + '.txt') in gold else q].add(c + '.txt' if not c.endswith('.txt') else c)

def bucket_of(g):
    return 'g1' if len(g) == 1 else ('g2_5' if len(g) <= 5 else 'g_gt5')

agg = {b: {'n_q': 0, 'tp': 0, 'n_pred': 0, 'n_gold': 0} for b in ('g1', 'g2_5', 'g_gt5')}
for q, g in gold.items():
    p = preds.get(q, set())
    b = bucket_of(g)
    agg[b]['n_q'] += 1
    agg[b]['tp'] += len(p & g)
    agg[b]['n_pred'] += len(p)
    agg[b]['n_gold'] += len(g)

dec = {}
for b, a in agg.items():
    P, R = a['tp'] / a['n_pred'], a['tp'] / a['n_gold']
    dec[b] = {**a, 'P': round(P, 4), 'R': round(R, 4),
              'F1': round(2 * P * R / (P + R), 4)}
res['du3_by_cardinality'] = dec

tp = sum(a['tp'] for a in agg.values())
P, R = tp / sum(a['n_pred'] for a in agg.values()), tp / total_gold
res['du3_overall_sanity'] = {'tp': tp, 'P': round(P, 4), 'R': round(R, 4),
                             'F1': round(2 * P * R / (P + R), 4),
                             'must_equal_official': 0.3141}

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != 'gold_count_distribution'}, indent=2))
