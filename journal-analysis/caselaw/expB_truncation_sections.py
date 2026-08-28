"""Experiment B - what a truncation window discards, per section (Task 1, 2026 test).

Uses the cached segmentation (doc_meta_test2026.pkl) for section/paragraph texts and the raw
corpus files for positions. Two analyses:
  B1: where ANALYSIS / DECISION sections start (token offset); share of docs whose
      Analysis/Decision begins beyond a 512 / 4096 / 8192-token window.
  B2: for each of the 1,750 gold (query, cited-case) pairs, locate the best-matching
      candidate paragraph (max Jaccard against any query paragraph - the retrieval evidence)
      and measure its token offset + host section; share of evidence beyond each window.
Writes expB_numbers.json. 2026-07-22.
"""
import json, pickle, re
from pathlib import Path
import numpy as np
import os

ROOT = os.environ.get("COLIEE_ROOT", "data")

HERE = Path(__file__).parent
ARCHIVE = Path(os.path.join(ROOT, "TASK1/ARCHIVE"))
CORPUS = ARCHIVE / 'task_one_ready_to_use/data/task1_test_files_2026/task1_test_files_2026'
GOLD = Path(os.path.join(ROOT, "TASK1/FINAL_SUBMISSION/task1_test_labels_2026.json"))
OUT = HERE / 'expB_numbers.json'
WINDOWS = (512, 4096, 8192)
TOK = re.compile(r'\b\w+\b')

meta = pickle.load(open(ARCHIVE / 'cache_features/doc_meta_test2026.pkl', 'rb'))
gold = {k.replace('.txt', ''): [v.replace('.txt', '') for v in vs]
        for k, vs in json.load(open(GOLD)).items()}

texts = {}
def text_of(cid):
    if cid not in texts:
        texts[cid] = (CORPUS / f'{cid}.txt').read_text(encoding='utf-8', errors='replace')
    return texts[cid]

def tok_offset(text, char_off):
    return len(TOK.findall(text[:char_off]))

# ── B1: section start positions over the full 1,848-doc corpus ──
res = {'meta': {'date': '2026-07-22', 'corpus_docs': len(meta), 'windows': list(WINDOWS)}}
sec_stats = {}
n_struct = 0
doc_tokens = {}
for cid, m in meta.items():
    t = text_of(cid)
    doc_tokens[cid] = len(TOK.findall(t))
    if not m['segments'].get('has_structure'):
        continue
    n_struct += 1
    for sec in ('ANALYSIS', 'DECISION', 'BACKGROUND'):
        st = m['segments']['sections'].get(sec)
        if not st:
            continue
        probe = st[:80].strip()
        off = t.find(probe) if probe else -1
        if off < 0:
            continue
        sec_stats.setdefault(sec, []).append(tok_offset(t, off))
res['docs_with_structure'] = n_struct
res['doc_length_tokens'] = {'median': int(np.median(list(doc_tokens.values()))),
                            'mean': int(np.mean(list(doc_tokens.values())))}
res['section_start'] = {}
for sec, offs in sec_stats.items():
    a = np.array(offs)
    res['section_start'][sec] = {
        'n_docs': len(a), 'median_token': int(np.median(a)),
        **{f'beyond_{w}': round(float((a > w).mean()), 3) for w in WINDOWS}}

# ── B2: position of best-matching evidence paragraph per gold pair ──
def para_sets(cid):
    return [(num, frozenset(w.lower() for w in TOK.findall(txt)))
            for num, txt in meta[cid]['segments']['paragraphs']]

def marker_offset(text, num):
    m = re.search(rf'^\s*\[{num}\]\s*$', text, re.MULTILINE)
    return m.start() if m else -1

evid_off, evid_sec, skipped = [], {}, 0
sec_bounds_cache = {}
def sec_bounds(cid):
    if cid not in sec_bounds_cache:
        t = text_of(cid)
        b = []
        for sec, st in meta[cid]['segments']['sections'].items():
            probe = st[:80].strip()
            off = t.find(probe) if probe else -1
            if off >= 0:
                b.append((off, sec))
        sec_bounds_cache[cid] = sorted(b)
    return sec_bounds_cache[cid]

for q, cands in gold.items():
    if q not in meta:
        skipped += len(cands); continue
    qp = para_sets(q)
    if not qp:
        skipped += len(cands); continue
    for c in cands:
        if c not in meta:
            skipped += 1; continue
        cp = para_sets(c)
        if not cp:
            skipped += 1; continue
        best, best_num = -1.0, None
        for _, qs in qp:
            if not qs: continue
            for cnum, cs in cp:
                if not cs: continue
                j = len(qs & cs) / len(qs | cs)
                if j > best:
                    best, best_num = j, cnum
        t = text_of(c)
        off = marker_offset(t, best_num)
        if off < 0:
            skipped += 1; continue
        toff = tok_offset(t, off)
        evid_off.append(toff)
        host = 'UNKNOWN'
        for boff, sec in sec_bounds(c):
            if boff <= off:
                host = sec
        evid_sec[host] = evid_sec.get(host, 0) + 1

a = np.array(evid_off)
res['evidence_paragraph'] = {
    'n_pairs': len(a), 'skipped': skipped,
    'median_token_offset': int(np.median(a)),
    **{f'beyond_{w}': round(float((a > w).mean()), 3) for w in WINDOWS},
    'host_section_counts': dict(sorted(evid_sec.items(), key=lambda x: -x[1])),
    'offsets': evid_off}

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
