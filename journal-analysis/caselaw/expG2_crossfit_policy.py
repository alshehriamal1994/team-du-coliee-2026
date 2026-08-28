"""Experiment G2 - how far can features go on 'how many'? Cross-fitted ranker-score features.

The fair version of the winner's (NOWJ) approach on our system: count predictor fed by the
FINAL ranker's score shape. To avoid leakage, the DU9-config ranker is retrained 5x on
[2025train + 2025test + train2026_clean minus fold]; each held-out fold's queries get score
features from a model that never saw them. Test queries use the saved full DU9 model (never
saw test). Count regressor = LightGBM on [G1 features + ranker-score-shape features], trained
on the 1,712 clean 2026 train queries, applied to the 400 test queries; predicted counts
truncate the expA2 step8-filtered deep ranking. Ladder: fixed-5 0.3456, G1 0.3489,
oracle 0.4143. Writes expG2_numbers.json. ~70 min CPU. 2026-07-22.
"""
import json
import os, pickle, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT = os.environ.get("COLIEE_ROOT", "data")

class CacheItem:
    pass

HERE = Path(__file__).parent
ARCHIVE = Path(os.path.join(ROOT, "TASK1/ARCHIVE"))
CACHE = ARCHIVE / 'cache_features'
TRAIN_LBL = ARCHIVE / 'task_one_ready_to_use/data/combined_2026_labels.json'
TEST_LBL = Path(os.path.join(ROOT, "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"))
DU9_MODEL = Path(os.path.join(ROOT, "TASK1/runs/du7_tuning/du9/model.txt"))
DEEP = HERE / 'expA2_step8_deep.json'
OUT = HERE / 'expG2_numbers.json'
SEED = 42
RANKER = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5], num_leaves=255,
              learning_rate=0.02, min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
              reg_alpha=0.05, reg_lambda=0.05, verbose=-1, seed=SEED)
N_TREES = 4000

def norm(s): return s.replace('.txt', '')

def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return (d['X'].astype(np.float32), d['labels'].astype(np.int8),
            d['qids'].tolist(), d['cids'].tolist(), d['feature_names'].tolist())

def shape_feats(scores):
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    if len(s) < 12:
        s = np.pad(s, (0, 12 - len(s)), constant_values=s.min() if len(s) else 0.0)
    spread = s[0] - np.median(s) or 1.0
    r = (s[0] - s) / spread  # 0 at top, ~1 at median
    return {'sh_gap12': float(s[0] - s[1]), 'sh_gap25': float(s[1] - s[4]),
            'sh_gap510': float(s[4] - s[9]), 'sh_std10': float(s[:10].std()),
            'sh_n_r10': float((r <= 0.10).sum()), 'sh_n_r25': float((r <= 0.25).sum()),
            'sh_n_r50': float((r <= 0.50).sum()), 'sh_pool': float(len(scores))}

SH_NAMES = ['sh_gap12', 'sh_gap25', 'sh_gap510', 'sh_std10',
            'sh_n_r10', 'sh_n_r25', 'sh_n_r50', 'sh_pool']

print('Loading feature matrices...', flush=True)
X1, y1, q1, c1, fnames = load_npz(CACHE / 'feature_matrix_train2025.npz')
X2, y2, q2, c2, _ = load_npz(CACHE / 'feature_matrix_test2025.npz')
X3, y3, q3, c3, _ = load_npz(CACHE / 'feature_matrix_train2026_clean.npz')
Xt, yt, qt, ct, _ = load_npz(CACHE / 'feature_matrix_test2026.npz')

clean_q = sorted(set(q3))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(clean_q))
fold_of = {clean_q[i]: k for k, part in enumerate(np.array_split(order, 5)) for i in part}
q3a = np.array(q3)

def fit_ranker(X, y, qseq):
    mask = y >= 0
    Xm, ym = X[mask], y[mask]
    qm = [q for q, m in zip(qseq, mask) if m]
    qorder = list(dict.fromkeys(qm))
    from collections import Counter
    cnt = Counter(qm)
    groups = [cnt[q] for q in qorder]
    ds = lgb.Dataset(Xm, label=ym, group=groups, feature_name=fnames, free_raw_data=False)
    return lgb.train(RANKER, ds, num_boost_round=N_TREES)

# cross-fitted score features for the 1,712 clean train queries
cf_shape = {}
for k in range(5):
    t0 = time.time()
    hold = np.array([fold_of[q] == k for q in q3])
    Xtr = np.vstack([X1, X2, X3[~hold]])
    ytr = np.concatenate([y1, y2, y3[~hold]])
    qtr = q1 + q2 + list(q3a[~hold])
    m = fit_ranker(Xtr, ytr, qtr)
    sc = m.predict(X3[hold])
    per_q = {}
    for s, q in zip(sc, q3a[hold]):
        per_q.setdefault(q, []).append(float(s))
    for q, ss in per_q.items():
        cf_shape[q] = shape_feats(ss)
    print(f'fold {k}: {time.time()-t0:.0f}s, {len(per_q)} queries', flush=True)

# test-side score features from the saved full DU9 model (never saw test queries)
du9 = lgb.Booster(model_file=str(DU9_MODEL))
sc_t = du9.predict(Xt)
per_qt = {}
for s, q in zip(sc_t, qt):
    per_qt.setdefault(q, []).append(float(s))
te_shape = {q: shape_feats(ss) for q, ss in per_qt.items()}

# G1 base features (same construction as expG_learned_policy)
import importlib.util
import os

spec = importlib.util.spec_from_file_location('expg', HERE / 'expG_learned_policy.py')

def g1_features(meta_path, cache_path, qids):
    meta = pickle.load(open(meta_path, 'rb'))
    s1 = pickle.load(open(cache_path, 'rb'))
    def curve(scores, prefix):
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
    rows, names = [], None
    for q in qids:
        m = meta[q]
        c = s1[q + '.txt'].__dict__
        f = {'n_tokens': float(len(m['tokens'])),
             'n_paras': float(len(m['segments']['paragraphs'])),
             'header_words': float(len(m['segments']['header'].split())),
             'body_words': float(len(m['segments']['body'].split())),
             'year': float(m['year'] or 0), 'n_suppr': float(m['n_suppressions'])}
        f.update(curve(c['bm25_scores'], 'bm25'))
        f.update(curve(c['dense_scores'], 'dense'))
        f['s1_overlap10'] = float(len(set(c['bm25_ids'][:10]) & set(c['dense_ids'][:10])))
        if names is None:
            names = list(f)
        rows.append([f[n] for n in names])
    return np.array(rows), names

train_lbl = {norm(k): len(v) for k, v in json.load(open(TRAIN_LBL)).items()}
test_gold = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(TEST_LBL)).items()}
tq = [q for q in sorted(set(q3)) if q in train_lbl and q in cf_shape]
eq = sorted(test_gold)

Gtr, gnames = g1_features(CACHE / 'doc_meta_train2026.pkl',
                          ARCHIVE / 'cache_2026/train_cache_dotxt.pkl', tq)
Gte, _ = g1_features(CACHE / 'doc_meta_test2026.pkl',
                     ARCHIVE / 'cache_2026/test_cache_dotxt.pkl', eq)
Str = np.array([[cf_shape[q][n] for n in SH_NAMES] for q in tq])
Ste = np.array([[te_shape[q][n] for n in SH_NAMES] for q in eq])
XtrC = np.hstack([Gtr, Str])
XteC = np.hstack([Gte, Ste])
allnames = gnames + SH_NAMES
ytrC = np.array([train_lbl[q] for q in tq], dtype=float)
yteC = np.array([len(test_gold[q]) for q in eq], dtype=float)

REG = dict(objective='regression', metric='l1', num_leaves=31, learning_rate=0.05,
           n_estimators=500, subsample=0.8, colsample_bytree=0.8, seed=SEED, verbose=-1)
idx = rng.permutation(len(tq))
folds = np.array_split(idx, 5)
cv_pred = np.zeros(len(tq))
for i in range(5):
    tr_i = np.concatenate([folds[j] for j in range(5) if j != i])
    m = lgb.LGBMRegressor(**REG).fit(XtrC[tr_i], ytrC[tr_i])
    cv_pred[folds[i]] = m.predict(XtrC[folds[i]])
model = lgb.LGBMRegressor(**REG).fit(XtrC, ytrC)
te_pred = np.clip(np.round(model.predict(XteC)), 1, 30)

deep = {norm(k): [norm(v) for v in vs] for k, vs in json.load(open(DEEP)).items()}
TOTAL = sum(len(v) for v in test_gold.values())

def f1_at(counts):
    tp = n = 0
    for q, k in zip(eq, counts):
        sel = deep.get(q, [])[:int(k)]
        tp += len(set(sel) & set(test_gold[q]))
        n += len(sel)
    P, R = tp / n, tp / TOTAL
    return {'P': round(P, 4), 'R': round(R, 4), 'F1': round(2 * P * R / (P + R), 4)}

def cmet(pred, gold):
    return {'MAE': round(float(np.abs(pred - gold).mean()), 3),
            'corr': round(float(np.corrcoef(pred, gold)[0, 1]), 3),
            'mean_pred': round(float(pred.mean()), 2)}

gap = 0.4143 - 0.3456
res = {'meta': {'date': '2026-07-22', 'n_train_q': len(tq), 'seed': SEED,
                'ladder': {'fixed5': 0.3456, 'G1_learned': 0.3489, 'oracle': 0.4143,
                           'winner_NOWJ': 0.4220}},
       'count_cv_train': cmet(np.clip(np.round(cv_pred), 1, 30), ytrC),
       'count_test': cmet(te_pred, yteC),
       'f1_test': f1_at(te_pred),
       'feature_importance_top10': dict(sorted(zip(allnames, model.feature_importances_.tolist()),
                                               key=lambda x: -x[1])[:10])}
res['gap_closed_pct'] = round(100 * (res['f1_test']['F1'] - 0.3456) / gap, 1)
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
