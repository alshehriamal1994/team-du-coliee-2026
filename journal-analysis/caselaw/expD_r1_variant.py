"""Experiment D — confidence-elicited multi-selection (Task 2, DeepSeek-V3 few-shot).

Identical to run_multiselect_experiment.py's v3_multiselect_rag configuration (same demos,
same reranker top-20, same 1200-char truncation, same rules, temperature 0) EXCEPT the
output instruction: the model returns a JSON object {paragraph_id: confidence}. A threshold
sweep over elicited confidence then yields the adaptive-cardinality curve from ONE run.

Key read from the OPENROUTER_API_KEY environment variable. Raw responses appended to expD_raw_r1.jsonl (resumable);
analysis writes expD_r1_numbers.json. 2026-07-21.
"""
import json, pickle, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
import os
import sys

ROOT = os.environ.get("COLIEE_ROOT", "data")

HERE = Path(__file__).parent
T2 = Path(os.path.join(ROOT, "TASK2_code"))
DATA_DIR = Path(os.path.join(ROOT, "RESTORED_FROM_TRASH/TASK2/real_data_we_should_use/task2_test_files_2026(1)/task2_test_files_2026"))
RAW_PATH = HERE / 'expD_raw_r1.jsonl'
OUT_PATH = HERE / 'expD_r1_numbers.json'
KEY = os.environ.get("OPENROUTER_API_KEY")
if not KEY:
    sys.exit("OPENROUTER_API_KEY not set")
MODEL = 'deepseek/deepseek-r1'
TOP_K = 20

PROMPT = """\
You are an expert in Canadian Federal Court legal case entailment.

Here are {n_examples} examples of CORRECT legal entailment from similar cases \
(each shows a decision fragment and the paragraph(s) that legally entail it):

{examples}
---
Now, for the NEW decision fragment below, identify ALL candidate paragraphs \
whose legal reasoning NECESSARILY and DIRECTLY entails the decision.

Decision fragment:
{query}

Candidate paragraphs:
{paragraphs}

IMPORTANT RULES:
1. Most cases have 2-3 entailing paragraphs, not just one.
2. Select EVERY paragraph that states a legal rule, principle, or reasoning \
   that DIRECTLY and NECESSARILY supports or produces the decision.
3. Different paragraphs may contribute different parts of the legal basis — \
   one may state the general rule, another may apply it, another may address \
   an exception. Select ALL such paragraphs.
4. REJECT paragraphs that are merely topically related, provide background \
   facts only, or state final dispositions without reasoning.
5. If no paragraph genuinely entails the decision, return an empty object.

For each selected paragraph, also give your confidence that it entails the \
decision, from 0.00 to 1.00. Return ONLY a JSON object mapping each selected \
paragraph ID to its confidence (e.g., {{"033": 0.95}} or \
{{"012": 0.90, "033": 0.60, "047": 0.35}}), or {{}} if none. Nothing else."""


def call_openrouter(prompt, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                              json={'model': MODEL,
                                    'messages': [{'role': 'user', 'content': prompt}],
                                    'temperature': 0.0, 'max_tokens': 512},
                              headers={'Authorization': f'Bearer {KEY}',
                                       'Content-Type': 'application/json'},
                              timeout=120)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'    API error (attempt {attempt+1}): {e}', file=sys.stderr)
            time.sleep(2 ** (attempt + 1))
    return ''


def parse_conf(text, valid_ids):
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    norm = {p.lstrip('0') or '0': p for p in valid_ids}
    out = {}
    for k, v in obj.items():
        pid = str(k).zfill(3)
        canon = pid if pid in valid_ids else norm.get(str(k).lstrip('0') or '0')
        if canon is None:
            continue
        try:
            out[canon] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            continue
    return out


# ── data (exact reproduction of the original run's inputs) ──
cache = pickle.load(open(T2 / 'archive/runs_final_2026/test_cache_monot5v2.pkl', 'rb'))
fewshot = json.load(open(T2 / 'archive/runs_experiments/fewshot_cache_test.json'))
raw_gold = json.load(open(T2 / 'task2_test_labels_2026(1).json'))
gold = {cid: {x.strip().replace('.txt', '').zfill(3) for x in val.split(',')}
        for cid, val in raw_gold.items()}

scores_index = {}
for row in cache['rows']:
    m5, q3 = np.array(row['m5']), np.array(row['q3'])
    pids = [p.zfill(3) for p in row['cand_ids']]
    n1 = np.ones_like(m5) if (m5.max() - m5.min()) < 1e-9 else (m5 - m5.min()) / (m5.max() - m5.min())
    n2 = np.ones_like(q3) if (q3.max() - q3.min()) < 1e-9 else (q3 - q3.min()) / (q3.max() - q3.min())
    c = 0.8 * n1 + 0.2 * n2
    order = np.argsort(-c)
    scores_index[row['cid']] = [(pids[i], float(c[i])) for i in order]

ALL = sorted(gold, key=int)
TOTAL = sum(len(g) for g in gold.values())

done = set()
if RAW_PATH.exists():
    for line in open(RAW_PATH):
        done.add(json.loads(line)['cid'])


def build_prompt(cid):
    case_dir = DATA_DIR / cid
    query = (case_dir / 'entailed_fragment.txt').read_text(encoding='utf-8', errors='replace').strip()
    para_texts = {f.stem.zfill(3): f.read_text(encoding='utf-8', errors='replace').strip()
                  for f in (case_dir / 'paragraphs').glob('*.txt')}
    top = [(pid, para_texts[pid]) for pid, _ in scores_index[cid][:TOP_K] if pid in para_texts]
    blocks = [f'[{pid}] ' + (t[:1200] + '...' if len(t) > 1200 else t) for pid, t in top]
    ex = fewshot[cid][:3]
    exb = [f'EXAMPLE {i}:\nDecision fragment: '
           + (e['query'][:300] + '...' if len(e['query']) > 300 else e['query'])
           + f"\nEntailing paragraph [{e['gold_pid']}]: "
           + (e['gold_text'][:500] + '...' if len(e['gold_text']) > 500 else e['gold_text'])
           for i, e in enumerate(ex, 1)]
    return (PROMPT.format(n_examples=len(ex), examples='\n\n'.join(exb),
                          query=query, paragraphs='\n\n'.join(blocks)),
            {pid for pid, _ in top})


def run_case(cid):
    prompt, valid = build_prompt(cid)
    resp = call_openrouter(prompt)
    return cid, resp, parse_conf(resp, valid)


todo = [c for c in ALL if c not in done]
print(f'{len(done)} cached, {len(todo)} to run')
if todo:
    with ThreadPoolExecutor(max_workers=6) as pool, open(RAW_PATH, 'a') as fh:
        futures = {pool.submit(run_case, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            cid, resp, conf = fut.result()
            fh.write(json.dumps({'cid': cid, 'response': resp, 'conf': conf}) + '\n')
            fh.flush()
            hits = len(set(conf) & gold[cid])
            print(f'  [{i:3d}/{len(todo)}] case {cid}: {len(conf)} selected, hit {hits}/{len(gold[cid])}')

# ── analysis ──
conf_by_case = {}
for line in open(RAW_PATH):
    d = json.loads(line)
    conf_by_case[d['cid']] = d['conf']
parse_fail = [c for c in ALL if not conf_by_case.get(c) and conf_by_case.get(c) != {}]
empty = [c for c in ALL if conf_by_case.get(c) == {}]

def ev(preds):
    tp = sum(len(set(p) & gold[c]) for c, p in preds.items())
    n = sum(len(p) for p in preds.values())
    P = tp / n if n else 0.0
    R = tp / TOTAL
    return {'P': round(P, 4), 'R': round(R, 4),
            'F1': round(2 * P * R / (P + R), 4) if P + R else 0.0, 'n_pred': n}

res = {'meta': {'date': '2026-07-21', 'model': MODEL, 'config': 'few-shot RAG + confidence JSON',
                'baseline_plain_multiselect': 0.5487, 'majority_vote': 0.555,
                'cases_empty_selection': len(empty), 'cases_unparsed': len(parse_fail)}}

sweep = {}
for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    preds = {c: [p for p, v in conf_by_case.get(c, {}).items() if v >= tau] for c in ALL}
    sweep[f'{tau:.2f}'] = ev(preds)
res['tau_sweep_raw'] = sweep

sweep1 = {}
for tau in [0.3, 0.5, 0.7, 0.8, 0.9]:
    preds = {}
    for c in ALL:
        cc = conf_by_case.get(c, {})
        keep = [p for p, v in cc.items() if v >= tau]
        preds[c] = keep or ([max(cc, key=cc.get)] if cc else [])
    sweep1[f'{tau:.2f}'] = ev(preds)
res['tau_sweep_min1'] = sweep1

# reliability of elicited confidence: bucket -> empirical precision
buckets = {}
for c in ALL:
    for p, v in conf_by_case.get(c, {}).items():
        b = min(int(v * 5), 4)  # 5 buckets of 0.2
        buckets.setdefault(b, [0, 0])
        buckets[b][0] += 1
        buckets[b][1] += int(p in gold[c])
res['reliability'] = {f'{b*0.2:.1f}-{(b+1)*0.2:.1f}': {'n': n, 'precision': round(k / n, 3)}
                      for b, (n, k) in sorted(buckets.items())}

# count calibration at tau=0
gn = np.array([len(gold[c]) for c in ALL])
pn = np.array([len(conf_by_case.get(c, {})) for c in ALL])
res['count_calibration_tau0'] = {'mean_pred': round(float(pn.mean()), 2), 'gold_mean': round(float(gn.mean()), 2),
                                 'corr': round(float(np.corrcoef(pn, gn)[0, 1]), 3),
                                 'MAE': round(float(np.abs(pn - gn).mean()), 2)}

OUT_PATH.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
