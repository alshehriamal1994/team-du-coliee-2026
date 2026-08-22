"""Experiment G — learned answer-set-size predictor for Task 1 (the constructive leg).

Predicts per-query citation count from LEAKAGE-FREE features only: query-side document
metadata (length, paragraphs, year, suppression markers) + stage-1 BM25/dense score-
distribution shape (stage-1 is not label-trained). LightGBM regressor, 5-fold CV on the
2,001 training queries for honest count metrics; retrained on all train, applied to the 400
test queries; predicted sizes truncate the expA2 step8-filtered deep ranking (DU9).
Baselines: constant round(train-mean)=4 and suppressions-only. Ladder context: fixed-5
0.3456, oracle-count 0.4143, winner 0.4220. Writes expG_numbers.json. 2026-07-22.
"""
import json, pickle
from pathlib import Path
import numpy as np
import lightgbm as lgb
import os

ROOT = os.environ.get("COLIEE_ROOT", "data")

class CacheItem:  # pickle shim for stage-1 caches
    pass

HERE = Path(__file__).parent
ARCHIVE = Path(os.path.join(ROOT, "TASK1/ARCHIVE"))
TRAIN_LBL = ARCHIVE / 'task_one_ready_to_use/data/combined_2026_labels.json'
TEST_LBL = Path(os.path.join(ROOT, "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"))
DEEP = HERE / 'expA2_step8_deep.json'
OUT = HERE / 'expG_numbers.json'
SEED = 42

def norm(s): return s.replace('.txt', '')

def curve_feats(scores, prefix):
    s = np.array(scores[:1000], dtype=float)
    if len(s) == 0 or s[0] <= 0:
        return {f'{prefix}_{n}': 0.0 for n in
                ('top1', 'mean10', 'std10', 'gap12', 'gap56', 'n50', 'n70', 'n90')}
    r = s / s[0]
    return {f'{prefix}_top1': float(s[0]), f'{prefix}_mean10': float(s[:10].mean()),
            f'{prefix}_std10': float(s[:10].std()),
            f'{prefix}_gap12': float(r[0] - r[1]) if len(r) > 1 else 0.0,
            f'{prefix}_gap56': float(r[4] - r[5]) if len(r) > 5 else 0.0,
            f'{prefix}_n50': float((r >= 0.5).sum()), f'{prefix}_n70': float((r >= 0.7).sum()),
            f'{prefix}_n90': float((r >= 0.9).sum())}

def build(meta_path, cache_path, qids):
    meta = pickle.load(open(meta_path, 'rb'))
    s1 = pickle.load(open(cache_path, 'rb'))
    rows, names = [], None
    for q in qids:
        m = meta[q]
        c = s1[q + '.txt'].__dict__
        f = {'n_tokens': float(len(m['tokens'])),
             'n_paras': float(len(m['segments']['paragraphs'])),
             'header_words': float(len(m['segments']['header'].split())),
             'body_words': float(len(m['segments']['body'].split())),
             'year': float(m['year'] or 0), 'n_suppr': float(m['n_suppressions'])}
        f.update(curve_feats(c['bm25_scores'], 'bm25'))
        f.update(curve_feats(c['dense_scores'], 'dense'))
        f['s1_overlap10'] = float(len(set(c['bm25_ids'][:10]) & set(c['dense_ids'][:10])))
        if names is None:
            names = list(f)
        rows.append([f[n] for n in names])
    return np.array(rows), names

train_lbl = {norm(k): len(v) for k, v in json.load(open(TRAIN_LBL)).items()}
test_gold = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(TEST_LBL)).items()}
tq = sorted(train_lbl)
eq = sorted(test_gold)

Xtr, names = build(ARCHIVE / 'cache_features/doc_meta_train2026.pkl',
                   ARCHIVE / 'cache_2026/train_cache_dotxt.pkl', tq)
ytr = np.array([train_lbl[q] for q in tq], dtype=float)
Xte, _ = build(ARCHIVE / 'cache_features/doc_meta_test2026.pkl',
               ARCHIVE / 'cache_2026/test_cache_dotxt.pkl', eq)
yte = np.array([len(test_gold[q]) for q in eq], dtype=float)

PARAMS = dict(objective='regression', metric='l1', num_leaves=31, learning_rate=0.05,
              n_estimators=500, subsample=0.8, colsample_bytree=0.8, seed=SEED, verbose=-1)

# 5-fold CV on train for honest count metrics
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(tq))
folds = np.array_split(idx, 5)
cv_pred = np.zeros(len(tq))
for i in range(5):
    te_i = folds[i]
    tr_i = np.concatenate([folds[j] for j in range(5) if j != i])
    m = lgb.LGBMRegressor(**PARAMS).fit(Xtr[tr_i], ytr[tr_i])
    cv_pred[te_i] = m.predict(Xtr[te_i])
cv_round = np.clip(np.round(cv_pred), 1, 30)

model = lgb.LGBMRegressor(**PARAMS).fit(Xtr, ytr)
te_pred = np.clip(np.round(model.predict(Xte)), 1, 30)

# suppressions-only baseline (linear fit on train)
A = np.vstack([Xtr[:, names.index('n_suppr')], np.ones(len(tq))]).T
coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
sup_pred = np.clip(np.round(coef[0] * Xte[:, names.index('n_suppr')] + coef[1]), 1, 30)

def count_metrics(pred, gold):
    return {'MAE': round(float(np.abs(pred - gold).mean()), 3),
            'corr': round(float(np.corrcoef(pred, gold)[0, 1]), 3),
            'mean_pred': round(float(pred.mean()), 2)}

deep = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(DEEP)).items()}
TOTAL = sum(len(v) for v in test_gold.values())

def f1_at(pred_counts):
    tp = n = 0
    for q, k in zip(eq, pred_counts):
        sel = deep.get(q, [])[:int(k)]
        tp += len(set(sel) & set(test_gold[q]))
        n += len(sel)
    P, R = tp / n, tp / TOTAL
    return {'P': round(P, 4), 'R': round(R, 4), 'F1': round(2 * P * R / (P + R), 4)}

gap = 0.4143 - 0.3456
res = {'meta': {'date': '2026-07-22', 'features': names, 'seed': SEED,
                'ladder': {'fixed5': 0.3456, 'oracle': 0.4143, 'winner_NOWJ': 0.4220}},
       'count_cv_train': count_metrics(cv_round, ytr),
       'count_test': {'learned': count_metrics(te_pred, yte),
                      'suppr_only': count_metrics(sup_pred, yte),
                      'constant4': count_metrics(np.full(len(eq), 4.0), yte),
                      'gold_mean': round(float(yte.mean()), 2)},
       'f1_test': {'learned': f1_at(te_pred), 'suppr_only': f1_at(sup_pred),
                   'constant4': f1_at(np.full(len(eq), 4))},
       'feature_importance_top8': dict(sorted(zip(names, model.feature_importances_.tolist()),
                                              key=lambda x: -x[1])[:8])}
res['gap_closed_pct'] = round(100 * (res['f1_test']['learned']['F1'] - 0.3456) / gap, 1)
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
