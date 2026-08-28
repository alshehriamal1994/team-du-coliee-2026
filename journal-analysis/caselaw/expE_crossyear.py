"""Experiment E - cross-year robustness of the feature ranker (Task 1).

DU9 itself saw 2025 test in training, so the honest cross-year design retrains from scratch:
  E-a: train on train2026_clean only (1,712 q) -> evaluate on test2025 (400 q, raw top-5).
  E-b: train on train2025 only (1,678 q)      -> evaluate on test2026 (400 q, raw top-5).
Reference points: DU9 (all-years training) raw top-5 on test2026 = 0.3211. Step8 omitted
(date filters are year-specific); all numbers are RAW top-5, like-for-like within this
experiment. DU9-config LightGBM, seed 42. Writes expE_numbers.json. ~30 min. 2026-07-22.
"""
import json, time
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
import os

ROOT = os.environ.get("COLIEE_ROOT", "data")

HERE = Path(__file__).parent
CACHE = Path(os.path.join(ROOT, "TASK1/ARCHIVE/cache_features"))
OUT = HERE / 'expE_numbers.json'
PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5], num_leaves=255,
              learning_rate=0.02, min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
              reg_alpha=0.05, reg_lambda=0.05, verbose=-1, seed=42)

def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return (d['X'].astype(np.float32), d['labels'].astype(np.int8),
            d['qids'].tolist(), d['cids'].tolist(), d['feature_names'].tolist())

def fit(X, y, qseq, fnames):
    mask = y >= 0
    qm = [q for q, m in zip(qseq, mask) if m]
    cnt = Counter(qm)
    groups = [cnt[q] for q in list(dict.fromkeys(qm))]
    ds = lgb.Dataset(X[mask], label=y[mask], group=groups, feature_name=fnames,
                     free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=4000)

def eval_top5(model, X, y, qseq):
    sc = model.predict(X)
    per_q, gold_n, tp = {}, Counter(), 0
    for s, q, lab in zip(sc, qseq, y):
        per_q.setdefault(q, []).append((float(s), int(lab)))
        if lab == 1:
            gold_n[q] += 1
    total_gold = sum(gold_n.values())
    for q, pairs in per_q.items():
        pairs.sort(key=lambda x: -x[0])
        tp += sum(lab for _, lab in pairs[:5])
    P = tp / (5 * len(per_q))
    R = tp / total_gold
    return {'P': round(P, 4), 'R': round(R, 4), 'F1': round(2 * P * R / (P + R), 4),
            'n_queries': len(per_q), 'total_gold': total_gold}

GOLD26 = {k.replace('.txt', ''): {v.replace('.txt', '') for v in vs}
          for k, vs in json.load(open(os.path.join(ROOT, "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"))).items()}

def eval_top5_gold26(model, X, qseq, cseq):
    sc = model.predict(X)
    per_q = {}
    for s_, q, c in zip(sc, qseq, cseq):
        per_q.setdefault(q, []).append((float(s_), c.replace('.txt', '')))
    tp = 0
    total_gold = sum(len(g) for g in GOLD26.values())
    for q, pairs in per_q.items():
        pairs.sort(key=lambda x: -x[0])
        g = GOLD26.get(q.replace('.txt', ''), set())
        tp += sum(1 for _, c in pairs[:5] if c in g)
    P = tp / (5 * len(per_q))
    R = tp / total_gold
    return {'P': round(P, 4), 'R': round(R, 4), 'F1': round(2 * P * R / (P + R), 4),
            'n_queries': len(per_q), 'total_gold': total_gold}

print('loading...', flush=True)
X25, y25, q25, c25, fn = load_npz(CACHE / 'feature_matrix_train2025.npz')
Xt25, yt25, qt25, ct25, _ = load_npz(CACHE / 'feature_matrix_test2025.npz')
X26, y26, q26, c26, _ = load_npz(CACHE / 'feature_matrix_train2026_clean.npz')
Xt26, yt26, qt26, ct26, _ = load_npz(CACHE / 'feature_matrix_test2026.npz')

res = {'meta': {'date': '2026-07-22', 'config': 'DU9 params, raw top-5, no step8',
                'reference_DU9_all_years_raw_test2026': 0.3211}}
t0 = time.time()
m_a = fit(X26, y26, q26, fn)
res['E_a_train2026clean_test2025'] = eval_top5(m_a, Xt25, yt25, qt25)
res['E_a_train2026clean_test2026'] = eval_top5_gold26(m_a, Xt26, qt26, ct26)
print('E-a done', time.time() - t0, flush=True)
m_b = fit(X25, y25, q25, fn)
res['E_b_train2025_test2026'] = eval_top5_gold26(m_b, Xt26, qt26, ct26)
res['E_b_train2025_test2025'] = eval_top5(m_b, Xt25, yt25, qt25)
print('E-b done', time.time() - t0, flush=True)

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
