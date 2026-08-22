"""Experiment C — Task 2 cardinality sweep on cached MonoT5-v2+Qwen3 reranker scores (no LLM).

Reproduces every number in the M2 claim ledger section "expC". Run from anywhere:
    python3 expC_cardinality_sweep.py
Writes expC_numbers.json next to this file. 2026-07-21.
"""
import json, pickle, collections
from pathlib import Path
import numpy as np
import os

ROOT = os.environ.get("COLIEE_ROOT", "data")

T2 = Path(os.path.join(ROOT, "TASK2_code"))
OUT = Path(__file__).with_name('expC_numbers.json')

cache = pickle.load(open(T2 / 'archive/runs_final_2026/test_cache_monot5v2.pkl', 'rb'))
raw_gold = json.load(open(T2 / 'task2_test_labels_2026(1).json'))
gold = {cid: {x.strip().replace('.txt', '').zfill(3) for x in val.split(',')}
        for cid, val in raw_gold.items()}
TOTAL = sum(len(g) for g in gold.values())

# Exact submission-time ensemble: per-query min-max, 0.8*MonoT5-v2 + 0.2*Qwen3 (run_multiselect_experiment.py)
scores = {}
for row in cache['rows']:
    m5, q3 = np.array(row['m5']), np.array(row['q3'])
    pids = [p.zfill(3) for p in row['cand_ids']]
    n1 = np.ones_like(m5) if (m5.max() - m5.min()) < 1e-9 else (m5 - m5.min()) / (m5.max() - m5.min())
    n2 = np.ones_like(q3) if (q3.max() - q3.min()) < 1e-9 else (q3 - q3.min()) / (q3.max() - q3.min())
    c = 0.8 * n1 + 0.2 * n2
    order = np.argsort(-c)
    scores[row['cid']] = [(pids[i], float(c[i])) for i in order]

def ev(preds):
    tp = sum(len(set(p) & gold[c]) for c, p in preds.items())
    n = sum(len(p) for p in preds.values())
    P = tp / n if n else 0.0
    R = tp / TOTAL
    return {'P': round(P, 4), 'R': round(R, 4),
            'F1': round(2 * P * R / (P + R), 4) if P + R else 0.0, 'n_pred': n}

res = {'meta': {'date': '2026-07-21', 'gold_paras': TOTAL, 'cases': len(gold),
                'official_DU3': 0.3427, 'official_winner_IAI': 0.4899,
                'llm_multiselect_best_single': 0.5487, 'llm_multiselect_majority': 0.555}}

res['fixed_topk'] = {f'k{k}': ev({c: [p for p, _ in s[:k]] for c, s in scores.items()})
                     for k in range(1, 7)}

res['abs_threshold_min1'] = {
    f'{th:.2f}': ev({c: [p for p, sc in s if sc >= th] or [s[0][0]] for c, s in scores.items()})
    for th in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}

res['rel_threshold_min1'] = {
    f'{a:.2f}': ev({c: [p for p, sc in s if sc >= a * s[0][1]] or [s[0][0]] for c, s in scores.items()})
    for a in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}

res['oracle_cardinality'] = ev({c: [p for p, _ in s[:len(gold[c])]] for c, s in scores.items()})

# fine-grid search over relative threshold (stored so the 0.39 figure is ledgered)
best_a, best = None, None
for a in np.arange(0.40, 0.99, 0.01):
    preds = {c: [p for p, sc in s if sc >= a * s[0][1]] or [s[0][0]] for c, s in scores.items()}
    e = ev(preds)
    if best is None or e['F1'] > best['F1']:
        best, best_a = e, float(a)
res['rel_threshold_finegrid_best'] = {'alpha': round(best_a, 2), **best}

# Count calibration: LLM multi-select runs vs gold counts vs best-threshold counts
cids = sorted(gold, key=int)
gn = np.array([len(gold[c]) for c in cids])
calib = {}
for name, path in [('v3_fewshot', 'runs_multiselect/test2026_v3_multiselect_rag.txt'),
                   ('r1_fewshot', 'runs_multiselect/test2026_r1_multiselect_rag.txt'),
                   ('v3_zeroshot', 'runs_multiselect/test2026_v3_multiselect_zero.txt')]:
    preds = collections.defaultdict(set)
    for line in open(T2 / path):
        parts = line.split()
        preds[parts[0]].add(parts[1].replace('.txt', '').zfill(3))
    pn = np.array([len(preds.get(c, set())) for c in cids])
    calib[name] = {'F1': ev({c: list(p) for c, p in preds.items()})['F1'],
                   'mean_count': round(float(pn.mean()), 2),
                   'corr': round(float(np.corrcoef(pn, gn)[0, 1]), 3),
                   'MAE': round(float(np.abs(pn - gn).mean()), 2)}
res['count_calibration'] = calib
res['count_calibration']['gold_mean'] = round(float(gn.mean()), 2)

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
