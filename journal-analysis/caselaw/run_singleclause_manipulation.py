#!/usr/bin/env python3
"""A controlled manipulation of the selection limit (journal article, Section 6).

Two arms on the 100 Task 2 test cases, identical configuration (DeepSeek-V3, retrieved
worked examples, monoT5-v2 fused top-20, temperature 0). The control arm runs the
deployed single-selection prompt verbatim. The treatment arm is identical except for the
three phrases that express the selection limit, the entailment criterion and every other
component being unchanged, so the difference between the arms is the effect of the count
permission alone, measured on the same day against the same endpoint.

The API key is read from the OPENROUTER_API_KEY environment variable and is never written
or printed. Predictions are written incrementally and the run resumes from partial
output. Scores are reported only if all cases completed.
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

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY not set")

DATA_DIR = ROOT / "RESTORED_FROM_TRASH/task2_test_files_2026"
CACHE_PATH = str(ROOT / "TASK2_code/archive/runs_final_2026/test_cache_monot5v2.pkl")
FEWSHOT_PATH = str(ROOT / "TASK2_code/archive/runs_experiments/fewshot_cache_test.json")
LABELS_PATH = str(ROOT / "TASK2_code/task2_test_labels_2026(1).json")
OUTPUT_DIR = ROOT / "TASK2_code/runs_singleclause"
MODEL = "deepseek/deepseek-chat"
TOP_K = 20
FAILED_CALLS = []

PROMPT_CONTROL = """\
You are an expert in Canadian Federal Court legal case entailment.

Here are {n_examples} examples of CORRECT legal entailment from similar cases \
(each shows a decision fragment and the paragraph that legally entails it):

{examples}
---
Now, for the NEW decision fragment below, apply the same reasoning.
Identify the SINGLE candidate paragraph (if any) whose legal rule or principle \
NECESSARILY and DIRECTLY produces this decision.

Decision fragment:
{query}

Candidate paragraphs:
{paragraphs}

Critical rules:
- Select AT MOST ONE paragraph — the single most directly entailing
- A paragraph entails ONLY if it states the specific legal rule/principle that FORCES the decision
- Topical similarity, background facts, or final dispositions are NOT entailment
- If no paragraph truly entails the decision, return "none"

Return ONLY the paragraph ID (e.g., "033") or "none". Nothing else."""

PROMPT_TREATMENT = """\
You are an expert in Canadian Federal Court legal case entailment.

Here are {n_examples} examples of CORRECT legal entailment from similar cases \
(each shows a decision fragment and the paragraph that legally entails it):

{examples}
---
Now, for the NEW decision fragment below, apply the same reasoning.
Identify EVERY candidate paragraph whose legal rule or principle \
NECESSARILY and DIRECTLY produces this decision.

Decision fragment:
{query}

Candidate paragraphs:
{paragraphs}

Critical rules:
- Select EVERY entailing paragraph
- A paragraph entails ONLY if it states the specific legal rule/principle that FORCES the decision
- Topical similarity, background facts, or final dispositions are NOT entailment
- If no paragraph truly entails the decision, return "none"

Return ONLY the paragraph ID(s) separated by spaces (e.g., "033" or "012 033") or "none". Nothing else."""


def call_openrouter(prompt, retries=8):
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL, "temperature": 0.0, "max_tokens": 512,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=180)
            if r.status_code != 200:
                time.sleep(min(2 ** attempt, 60)); continue
            content = r.json()["choices"][0]["message"].get("content")
            if content is None or not content.strip():
                time.sleep(min(2 ** attempt, 60)); continue
            return content
        except Exception:
            time.sleep(min(2 ** attempt, 60))
    return None


def parse_para_ids(text, valid_ids):
    if text is None:
        return None
    if "none" in text.strip().lower()[:30] and not re.search(r"\d", text[:30]):
        return []
    found = re.findall(r"\b(\d{3})\b", text)
    seen, out = set(), []
    for f in found:
        if f in valid_ids and f not in seen:
            seen.add(f); out.append(f)
    return out


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_gold = json.load(open(LABELS_PATH))
    gold = {cid: {x.strip().replace(".txt", "").zfill(3)
                  for x in val.split(",") if x.strip()}
            for cid, val in raw_gold.items()}
    ALL_CASES = sorted(gold, key=int)
    TOTAL_RELEVANT = sum(len(g) for g in gold.values())
    print(f"{len(ALL_CASES)} cases, {TOTAL_RELEVANT} gold paragraphs", flush=True)

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    fewshot_cache = json.load(open(FEWSHOT_PATH))

    # reranker fusion, copied from run_multiselect_noprior.py
    scores_index = {}
    for row in cache["rows"]:
        cid = row["cid"]
        m5 = np.array(row["m5"]); q3 = np.array(row["q3"])
        pids = [p.zfill(3) for p in row["cand_ids"]]
        r1 = m5.max() - m5.min(); r2 = q3.max() - q3.min()
        n1 = np.ones_like(m5) if r1 < 1e-9 else (m5 - m5.min()) / r1
        n2 = np.ones_like(q3) if r2 < 1e-9 else (q3 - q3.min()) / r2
        combined = 0.8 * n1 + 0.2 * n2
        order = np.argsort(-combined)
        scores_index[cid] = [(pids[i], combined[i]) for i in order]

    def evaluate(preds, name):
        if FAILED_CALLS:
            print(f"  !! {len(FAILED_CALLS)} permanent failures; "
                  f"THIS SCORE IS NOT VALID.")
        correct = sum(len(gold[c] & preds.get(c, set())) for c in ALL_CASES)
        retrieved = sum(len(preds.get(c, set())) for c in ALL_CASES)
        P = correct / retrieved if retrieved else 0.0
        R = correct / TOTAL_RELEVANT
        F = 2 * P * R / (P + R) if P + R else 0.0
        avg = np.mean([len(preds.get(c, set())) for c in ALL_CASES])
        print(f"  {name:12s} P={P:.4f} R={R:.4f} F1={F:.4f} "
              f"correct={correct} ret={retrieved} avg={avg:.2f}", flush=True)

    for arm, template in (("control", PROMPT_CONTROL), ("treatment", PROMPT_TREATMENT)):
        out_path = OUTPUT_DIR / f"test2026_{arm}.txt"
        done = set()
        if out_path.exists():
            done = {l.split()[0] for l in out_path.read_text().splitlines() if l.split()}
        print(f"\n=== {arm} ({len(done)} cases already done) ===", flush=True)
        preds = defaultdict(set)
        for line in (out_path.read_text().splitlines() if out_path.exists() else []):
            parts = line.split()
            if len(parts) >= 2:
                preds[parts[0]].add(parts[1].zfill(3))
        for idx, cid in enumerate(ALL_CASES):
            if cid in done:
                continue
            case_dir = DATA_DIR / str(cid)
            query = (case_dir / "entailed_fragment.txt").read_text(
                encoding="utf-8", errors="replace").strip()
            para_dir = case_dir / "paragraphs"
            para_texts = {f.stem.zfill(3): f.read_text(encoding="utf-8", errors="replace").strip()
                          for f in para_dir.glob("*.txt")}
            top_items = [(pid, para_texts[pid]) for pid, _ in scores_index[cid][:TOP_K]
                         if pid in para_texts]
            valid_ids = {pid for pid, _ in top_items}
            blocks = [f"[{pid}] " + (t[:1200] + "..." if len(t) > 1200 else t)
                      for pid, t in top_items]
            paragraphs_str = "\n\n".join(blocks)
            examples = fewshot_cache[cid][:3]
            example_blocks = []
            for i, ex in enumerate(examples, 1):
                exq = ex["query"][:300] + "..." if len(ex["query"]) > 300 else ex["query"]
                exp = ex["gold_text"][:500] + "..." if len(ex["gold_text"]) > 500 else ex["gold_text"]
                example_blocks.append(f"EXAMPLE {i}:\nDecision fragment: {exq}\n"
                                      f"Entailing paragraph [{ex['gold_pid']}]: {exp}")
            prompt = template.format(query=query, paragraphs=paragraphs_str,
                                     examples="\n\n".join(example_blocks),
                                     n_examples=len(examples))
            response = call_openrouter(prompt)
            selected = parse_para_ids(response, valid_ids)
            if selected is None:
                FAILED_CALLS.append((arm, cid))
                print(f"  [{arm} {idx+1:3d}/100] {cid}: PERMANENT FAILURE", flush=True)
                continue
            preds[cid] = set(selected)
            with open(out_path, "a") as f:
                for pid in selected:
                    f.write(f"{cid} {pid} {arm}\n")
                if not selected:
                    f.write(f"{cid} __none__ {arm}\n")
            hit = len(set(selected) & gold[cid])
            print(f"  [{arm} {idx+1:3d}/100] {cid}: sel={selected or 'none'} "
                  f"hit={hit}/{len(gold[cid])}", flush=True)
        evaluate({c: {p for p in v if p != "__none__"} for c, v in preds.items()}, arm)

    print("\nFinal scores:")
    for arm in ("control", "treatment"):
        preds = defaultdict(set)
        pth = OUTPUT_DIR / f"test2026_{arm}.txt"
        for line in pth.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] != "__none__":
                preds[parts[0]].add(parts[1].zfill(3))
        evaluate(preds, arm)


if __name__ == "__main__":
    main()
