"""Experiment A2 - Task 1 cardinality ladder on full DU9 rankings (no retraining).

Loads the saved DU9 booster + cached 2026 test features, predicts full rankings, applies
the ORIGINAL step8 postprocess script at depth 30, then computes: fixed-k sweep (k=1..10),
oracle-count top-n_i, and per-cardinality decomposition. Sanity gates: raw top-5 F1 must
equal 0.3211 and step8 top-5 F1 must equal 0.3456 (recorded DU9 metrics) or the script aborts.
Writes expA2_numbers.json. 2026-07-21.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
import os

ROOT = os.environ.get("COLIEE_ROOT", "data")

HERE = Path(__file__).parent
T1 = Path(os.path.join(ROOT, "TASK1"))
ARCHIVE = T1 / 'ARCHIVE'
MODEL = T1 / 'runs/du7_tuning/du9/model.txt'
GOLD = T1 / 'FINAL_SUBMISSION/task1_test_labels_2026.json'
STEP8 = ARCHIVE / 'code_AUTHORITY_v2/step8_postprocess_filters_v2.py'
CORPUS = ARCHIVE / 'task_one_ready_to_use/data/task1_test_files_2026/task1_test_files_2026'
CACHE = ARCHIVE / 'cache_2026/test_cache_dotxt.pkl'
OUT = HERE / 'expA2_numbers.json'
DEPTH = 30

def norm(s): return s.replace('.txt', '')

gold = {norm(k): {norm(v) for v in vs} for k, vs in json.load(open(GOLD)).items()}
TOTAL = sum(len(g) for g in gold.values())

def ev(preds):
    tp = sum(len(set(p) & gold[q]) for q, p in preds.items() if q in gold)
    n = sum(len(p) for p in preds.values())
    P = tp / n if n else 0.0
    R = tp / TOTAL
    return {'P': round(P, 4), 'R': round(R, 4),
            'F1': round(2 * P * R / (P + R), 4) if P + R else 0.0, 'n_pred': n}

print('Loading model + features...')
booster = lgb.Booster(model_file=str(MODEL))
d = np.load(ARCHIVE / 'cache_features/feature_matrix_test2026.npz', allow_pickle=True)
X, qids, cids = d['X'].astype(np.float32), d['qids'].tolist(), d['cids'].tolist()
scores = booster.predict(X)

per_query = {}
for s, q, c in zip(scores, qids, cids):
    per_query.setdefault(q, []).append((c, float(s)))
ranked = {q: [norm(c) for c, _ in sorted(p, key=lambda x: x[1], reverse=True)]
          for q, p in per_query.items()}

raw5 = ev({q: r[:5] for q, r in ranked.items()})
print('sanity raw top-5:', raw5)
assert abs(raw5['F1'] - 0.3211) < 0.0002, f'raw sanity FAILED: {raw5}'

# step8 at depth 30 on top-100 base lists (original script, original flags)
base = {q + '.txt': [c + '.txt' for c in r[:100]] for q, r in ranked.items()}
bp = HERE / 'expA2_base_preds.json'
bp.write_text(json.dumps(base))
s8_out = HERE / 'expA2_step8_deep.json'
subprocess.run([sys.executable, str(STEP8), '--corpus', str(CORPUS), '--cache', str(CACHE),
                '--base_preds', str(bp), '--out', str(s8_out), '--out_k', str(DEPTH),
                '--rrf_k', '5', '--remove_query_cases', '--filter_future'],
               check=True, capture_output=True, text=True)
deep = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(s8_out)).items()}

s8_5 = ev({q: r[:5] for q, r in deep.items()})
print('sanity step8 top-5:', s8_5)
assert abs(s8_5['F1'] - 0.3456) < 0.0002, f'step8 sanity FAILED: {s8_5}'

res = {'meta': {'date': '2026-07-21', 'model': 'DU9 (post-competition, 4000 trees)',
                'official_DU3': 0.3141, 'winner_NOWJ': 0.4220,
                'sanity_raw_top5': raw5, 'sanity_step8_top5': s8_5, 'depth': DEPTH}}

res['fixed_k'] = {f'k{k}': ev({q: r[:k] for q, r in deep.items()}) for k in range(1, 11)}
res['oracle_count'] = ev({q: deep[q][:len(gold[q])] for q in deep if q in gold})

buckets = {'g1': [], 'g2_5': [], 'g_gt5': []}
for q, g in gold.items():
    r = deep.get(q, [])
    b = 'g1' if len(g) == 1 else ('g2_5' if len(g) <= 5 else 'g_gt5')
    buckets[b].append((len(set(r[:len(g)]) & g), len(g)))
res['oracle_by_cardinality'] = {}
for b, rows in buckets.items():
    tp = sum(t for t, _ in rows)
    n = sum(g for _, g in rows)
    res['oracle_by_cardinality'][b] = {'n_q': len(rows), 'tp': tp, 'n_gold': n,
                                       'F1': round(tp / n, 4)}  # P=R=F1 at oracle count

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
