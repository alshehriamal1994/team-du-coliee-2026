#!/usr/bin/env python3
"""
Post-competition experiment: Re-run LLMs with multi-select prompts.
Uses existing reranker cache + fewshot cache, calls OpenRouter API.

Runs 3 configurations:
  1. DeepSeek-V3 with Legal-RAG multi-select (top-all, max 3)
  2. DeepSeek-R1 with Legal-RAG multi-select (top-all, max 3)
  3. DeepSeek-V3 zero-shot multi-select (no RAG, max 3)
"""
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
DATA_DIR = Path("../data/task2")
CACHE_PATH = "cache/runs_final_2026/test_cache_monot5v2.pkl"
FEWSHOT_PATH = "cache/runs_experiments/fewshot_cache_test.json"
LABELS_PATH = "./task2_test_labels_2026(1).json"
OUTPUT_DIR = Path("./runs_multiselect")
TOP_K = 20

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────

PROMPT_MULTISELECT_RAG = """\
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
5. If no paragraph genuinely entails the decision, return "none".

Return ONLY the paragraph ID(s) separated by spaces \
(e.g., "033" or "012 033 047") or "none". Nothing else."""


PROMPT_MULTISELECT_ZERO = """\
You are an expert in legal case entailment.

TASK: From the candidate paragraphs below, identify ALL paragraphs \
whose legal reasoning NECESSARILY and DIRECTLY ENTAILS the decision fragment.

Decision fragment (what must be entailed):
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
5. If no paragraph genuinely entails the decision, return "none".

Return ONLY the paragraph ID(s) separated by spaces \
(e.g., "033" or "012 033 047") or "none". Nothing else."""


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

def call_openrouter(prompt, model, temperature=0.0, max_tokens=512, retries=3):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/coliee-task2",
        "X-Title": "COLIEE Task 2 Post-Competition",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return "none"


def parse_para_ids(text, valid_ids):
    text = text.strip().lower()
    if "none" in text and not re.search(r"\b\d+\b", text):
        return []
    raw_ids = re.findall(r"\b\d+\b", text)
    norm_map = {p.lstrip("0") or "0": p for p in valid_ids}
    result = []
    seen = set()
    for rid in raw_ids:
        pid = rid.zfill(3)
        pid_norm = rid.lstrip("0") or "0"
        canonical = pid if pid in valid_ids else norm_map.get(pid_norm)
        if canonical and canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

print("Loading data...")
with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)

with open(FEWSHOT_PATH) as f:
    fewshot_cache = json.load(f)

with open(LABELS_PATH) as f:
    raw_gold = json.load(f)
gold = {}
for cid, val in raw_gold.items():
    gold[cid] = {x.strip().replace(".txt", "").zfill(3) for x in val.split(",")}

# Build reranker index
scores_index = {}
for row in cache['rows']:
    cid = row['cid']
    items = []
    m5 = np.array(row['m5'])
    q3 = np.array(row['q3'])
    pids = [p.zfill(3) for p in row['cand_ids']]
    r1 = m5.max() - m5.min()
    r2 = q3.max() - q3.min()
    n1 = np.ones_like(m5) if r1 < 1e-9 else (m5 - m5.min()) / r1
    n2 = np.ones_like(q3) if r2 < 1e-9 else (q3 - q3.min()) / r2
    combined = 0.8 * n1 + 0.2 * n2
    order = np.argsort(-combined)
    scores_index[cid] = [(pids[i], combined[i]) for i in order]

ALL_CASES = sorted(gold.keys(), key=int)
TOTAL_RELEVANT = sum(len(g) for g in gold.values())

def evaluate(preds, name):
    correct = retrieved = 0
    for cid in ALL_CASES:
        g = gold[cid]
        p = preds.get(cid, set())
        correct += len(g & p)
        retrieved += len(p)
    micro_p = correct / retrieved if retrieved else 0
    micro_r = correct / TOTAL_RELEVANT
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) else 0
    avg_n = np.mean([len(preds.get(c, set())) for c in ALL_CASES])
    print(f"  {name:55s}  P={micro_p:.4f} R={micro_r:.4f} F1={micro_f1:.4f}  correct={correct} ret={retrieved} avg={avg_n:.1f}")
    return micro_p, micro_r, micro_f1


# ─────────────────────────────────────────────────────────────
# RUN EXPERIMENTS
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("v3_multiselect_rag",  "deepseek/deepseek-chat",  PROMPT_MULTISELECT_RAG, True),
    ("r1_multiselect_rag",  "deepseek/deepseek-r1",    PROMPT_MULTISELECT_RAG, True),
    ("v3_multiselect_zero", "deepseek/deepseek-chat",  PROMPT_MULTISELECT_ZERO, False),
]

for exp_name, model, prompt_template, use_rag in EXPERIMENTS:
    out_path = OUTPUT_DIR / f"test2026_{exp_name}.txt"

    # Skip if already done
    if out_path.exists():
        existing = load_existing(out_path) if False else None
        print(f"\n{'='*70}")
        print(f"  {exp_name} — ALREADY EXISTS, loading...")
        preds = defaultdict(set)
        with open(out_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    preds[parts[0]].add(parts[1].zfill(3))
        evaluate(dict(preds), exp_name)
        continue

    print(f"\n{'='*70}")
    print(f"  Running: {exp_name} (model={model}, RAG={use_rag})")
    print(f"{'='*70}")

    results = []
    preds = {}

    for idx, cid in enumerate(ALL_CASES):
        # Load case text
        case_dir = DATA_DIR / str(cid)
        query = (case_dir / "entailed_fragment.txt").read_text(encoding="utf-8", errors="replace").strip()
        para_dir = case_dir / "paragraphs"
        para_texts = {}
        for f in para_dir.glob("*.txt"):
            para_texts[f.stem.zfill(3)] = f.read_text(encoding="utf-8", errors="replace").strip()

        # Get top-K from reranker
        ranked = scores_index.get(cid, [])
        top_items = []
        for pid, score in ranked[:TOP_K]:
            if pid in para_texts:
                top_items.append((pid, para_texts[pid]))

        valid_ids = {pid for pid, _ in top_items}

        # Build paragraphs block
        blocks = []
        for pid, text in top_items:
            snippet = text[:1200] + "..." if len(text) > 1200 else text
            blocks.append(f"[{pid}] {snippet}")
        paragraphs_str = "\n\n".join(blocks)

        # Build prompt
        if use_rag and cid in fewshot_cache:
            examples = fewshot_cache[cid][:3]
            example_blocks = []
            for i, ex in enumerate(examples, 1):
                ex_query = ex["query"][:300] + "..." if len(ex["query"]) > 300 else ex["query"]
                ex_para = ex["gold_text"][:500] + "..." if len(ex["gold_text"]) > 500 else ex["gold_text"]
                example_blocks.append(
                    f"EXAMPLE {i}:\n"
                    f"Decision fragment: {ex_query}\n"
                    f"Entailing paragraph [{ex['gold_pid']}]: {ex_para}"
                )
            examples_str = "\n\n".join(example_blocks)
            prompt = prompt_template.format(
                query=query, paragraphs=paragraphs_str,
                examples=examples_str, n_examples=len(examples)
            )
        else:
            prompt = prompt_template.format(query=query, paragraphs=paragraphs_str)

        # Call API
        response = call_openrouter(prompt, model)
        selected = parse_para_ids(response, valid_ids)

        preds[cid] = set(selected)

        # Write incrementally
        with open(out_path, "a") as f:
            for pid in selected:
                f.write(f"{cid} {pid} {exp_name}\n")

        # Progress
        hits = gold[cid] & set(selected)
        status = f"hit={len(hits)}/{len(gold[cid])}" if selected else "NONE"
        print(f"  [{idx+1:3d}/100] Case {cid}: selected={selected} gold={sorted(gold[cid])} → {status}")

        # Rate limit
        time.sleep(0.5)

    # Evaluate
    print(f"\n  --- Results for {exp_name} ---")
    evaluate(preds, exp_name)

# ─────────────────────────────────────────────────────────────
# FINAL: Evaluate all and compare
# ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  FINAL COMPARISON")
print(f"{'='*70}")

# Load all result files
for fname in sorted(OUTPUT_DIR.glob("*.txt")):
    preds = defaultdict(set)
    with open(fname) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                preds[parts[0]].add(parts[1].zfill(3))
    evaluate(dict(preds), fname.stem)

# Also show submitted runs for comparison
print()
for name, path in [
    ("DU1 (submitted)", FINAL + "DU1/task2_DU1.txt"),
    ("DU2 (submitted)", FINAL + "DU2/task2_DU2.txt"),
    ("DU3 (submitted)", FINAL + "DU3/task2_DU3.txt"),
]:
    preds = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                preds[parts[0]].add(parts[1].zfill(3))
    evaluate(dict(preds), name)

FINAL = "predictions/"
print(f"\n  Competition winner (IAI run2): P=0.4501 R=0.5374 F1=0.4899")
