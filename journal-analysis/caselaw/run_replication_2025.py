#!/usr/bin/env python3
"""Replication of the count-clause manipulation on the COLIEE 2025 Task 2 test collection.

Amir's request: reproduce the frozen manipulation on a collection our development never
touched. COLIEE folds each year's test collection into the next year's training data, so
no earlier collection is unseen by the trained reranker. This design therefore contains
no trained component at all:

  cases       the 71 of 100 COLIEE 2025 test cases that appear in neither the
              prompt-development split nor the stress split
              (replication2025_untouched_ids.json)
  candidates  plain BM25 over each case's own paragraphs, top 20
              (bm25_top20_2025.json; recall 101/122 = 0.828, the A/B ceiling)
  examples    Legal-RAG worked examples retrieved by BM25 from a bank with every
              2025 test case removed (fewshot_cache_2025.json; verified zero overlap)
  prompts     the expO control and treatment templates, byte-identical, asserted
              against the run_singleclause_manipulation.py source at startup

Both arms run on the same day and endpoint, temperature 0, DeepSeek-V3. The paper's
mechanism account predicts an intermediate effect here: this collection averages 1.72
gold paragraphs per case on the untouched subset, between the development split's 1.22
and the 2026 test's 2.94. The result is reported whichever way it lands.

API key from the OPENROUTER_API_KEY environment variable only; never written, never
printed.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

import numpy as np
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY not set")

HERE = Path(__file__).parent
DATA_DIR = Path(ROOT) / "task2_test_files_2025"
# The cached candidate lists, the worked-example index and the manipulation
# source ship beside this script in the repository. Fall back to COLIEE_ROOT for
# anyone running it from a private working tree.
def _resolve(name):
    here = HERE / name
    return here if here.exists() else Path(ROOT) / name

RUN_DIR = HERE if (HERE / "bm25_top20_2025.json").exists() else Path(ROOT) / "runs_replication_2025"
LABELS_PATH = Path(ROOT) / "task2_test_labels_2025.json"
UNTOUCHED_PATH = HERE / "replication2025_untouched_ids.json"
EXPO_SOURCE = _resolve("run_singleclause_manipulation.py")
MODEL = "deepseek/deepseek-chat"
TOP_K = 20
FAILED_CALLS = []


def extract_templates(source_path):
    """Pull the two prompt templates out of the expO script source, verbatim."""
    src = source_path.read_text()
    out = {}
    for name in ("PROMPT_CONTROL", "PROMPT_TREATMENT"):
        m = re.search(rf'{name} = """\\\n(.*?)"""', src, re.S)
        if not m:
            sys.exit(f"could not extract {name} from {source_path}")
        out[name] = m.group(1).replace("\\\n", "")
    return out["PROMPT_CONTROL"], out["PROMPT_TREATMENT"]


PROMPT_CONTROL, PROMPT_TREATMENT = extract_templates(EXPO_SOURCE)
for probe, tpl, arm in [
    ("Identify the SINGLE candidate paragraph (if any)", PROMPT_CONTROL, "control"),
    ("- Select AT MOST ONE paragraph", PROMPT_CONTROL, "control"),
    ("Identify EVERY candidate paragraph whose", PROMPT_TREATMENT, "treatment"),
    ("- Select EVERY entailing paragraph", PROMPT_TREATMENT, "treatment"),
]:
    if probe not in tpl:
        sys.exit(f"GATE FAILED: '{probe}' missing from extracted {arm} template")


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


TRAIN_DIR = Path(ROOT) / "task2_train_files_2026"


def read_example(ex):
    """Return the worked example's fragment and gold paragraph text.

    The cache shipped with this repository holds identifiers and retrieval
    scores only. The COLIEE corpus is licensed and is not redistributed, so the
    text is read from the reader's own copy under COLIEE_ROOT.
    """
    case_dir = TRAIN_DIR / ex["cid"]
    frag = case_dir / "entailed_fragment.txt"
    para = case_dir / "paragraphs" / f'{ex["gold_pid"]}.txt'
    if not frag.exists() or not para.exists():
        sys.exit(
            f"Worked example {ex['cid']}/{ex['gold_pid']} not found under "
            f"{TRAIN_DIR}. Set COLIEE_ROOT to your licensed copy of the "
            "COLIEE Task 2 training release."
        )
    return (frag.read_text(encoding="utf-8", errors="replace").strip(),
            para.read_text(encoding="utf-8", errors="replace").strip())


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
    RUN_DIR.mkdir(exist_ok=True)
    untouched = json.load(open(UNTOUCHED_PATH))
    raw_gold = json.load(open(LABELS_PATH))
    gold = {}
    for cid in untouched:
        v = raw_gold[cid]
        paras = v if isinstance(v, list) else [x for x in str(v).split(",") if x.strip()]
        gold[cid] = {p.strip().replace(".txt", "").zfill(3) for p in paras}
    CASES = sorted(gold, key=int)
    TOTAL_RELEVANT = sum(len(g) for g in gold.values())
    print(f"{len(CASES)} untouched cases, {TOTAL_RELEVANT} gold paragraphs", flush=True)
    assert len(CASES) == 71, "untouched set must hold 71 cases"

    shortlists = json.load(open(RUN_DIR / "bm25_top20_2025.json"))
    fewshot_cache = json.load(open(RUN_DIR / "fewshot_cache_2025.json"))
    lab25_all = set(raw_gold)
    leak = [ex["cid"] for c in CASES for ex in fewshot_cache[c] if ex["cid"] in lab25_all]
    assert not leak, f"GATE FAILED: worked examples drawn from 2025 cases: {leak[:5]}"

    def evaluate(preds, name):
        if FAILED_CALLS:
            print(f"  !! {len(FAILED_CALLS)} permanent failures; "
                  f"THIS SCORE IS NOT VALID.")
        correct = sum(len(gold[c] & preds.get(c, set())) for c in CASES)
        retrieved = sum(len(preds.get(c, set())) for c in CASES)
        P = correct / retrieved if retrieved else 0.0
        R = correct / TOTAL_RELEVANT
        F = 2 * P * R / (P + R) if P + R else 0.0
        avg = np.mean([len(preds.get(c, set())) for c in CASES])
        print(f"  {name:12s} P={P:.4f} R={R:.4f} F1={F:.4f} "
              f"correct={correct} ret={retrieved} avg={avg:.2f}", flush=True)

    for arm, template in (("control", PROMPT_CONTROL), ("treatment", PROMPT_TREATMENT)):
        out_path = RUN_DIR / f"test2025_{arm}.txt"
        done = set()
        preds = defaultdict(set)
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    done.add(parts[0])
                    if parts[1] != "__none__":
                        preds[parts[0]].add(parts[1].zfill(3))
        print(f"\n=== {arm} ({len(done)} cases already done) ===", flush=True)
        for idx, cid in enumerate(CASES):
            if cid in done:
                continue
            case_dir = DATA_DIR / cid
            query = (case_dir / "entailed_fragment.txt").read_text(
                encoding="utf-8", errors="replace").strip()
            para_texts = {f.stem.zfill(3): f.read_text(encoding="utf-8",
                                                       errors="replace").strip()
                          for f in (case_dir / "paragraphs").glob("*.txt")}
            top_items = [(pid, para_texts[pid]) for pid in shortlists[cid][:TOP_K]
                         if pid in para_texts]
            valid_ids = {pid for pid, _ in top_items}
            blocks = [f"[{pid}] " + (t[:1200] + "..." if len(t) > 1200 else t)
                      for pid, t in top_items]
            paragraphs_str = "\n\n".join(blocks)
            examples = fewshot_cache[cid][:3]
            example_blocks = []
            for i, ex in enumerate(examples, 1):
                ex_q, ex_g = read_example(ex)
                exq = ex_q[:300] + "..." if len(ex_q) > 300 else ex_q
                exp = ex_g[:500] + "..." if len(ex_g) > 500 else ex_g
                example_blocks.append(f"EXAMPLE {i}:\nDecision fragment: {exq}\n"
                                      f"Entailing paragraph [{ex['gold_pid']}]: {exp}")
            prompt = template.format(query=query, paragraphs=paragraphs_str,
                                     examples="\n\n".join(example_blocks),
                                     n_examples=len(examples))
            response = call_openrouter(prompt)
            selected = parse_para_ids(response, valid_ids)
            if selected is None:
                FAILED_CALLS.append((arm, cid))
                print(f"  [{arm} {idx+1:3d}/71] {cid}: PERMANENT FAILURE", flush=True)
                continue
            preds[cid] = set(selected)
            with open(out_path, "a") as f:
                for pid in selected:
                    f.write(f"{cid} {pid} {arm}\n")
                if not selected:
                    f.write(f"{cid} __none__ {arm}\n")
            hit = len(set(selected) & gold[cid])
            print(f"  [{arm} {idx+1:3d}/71] {cid}: sel={selected or 'none'} "
                  f"hit={hit}/{len(gold[cid])}", flush=True)
        evaluate(preds, arm)

    print("\nFinal scores (recomputed from files):")
    for arm in ("control", "treatment"):
        preds = defaultdict(set)
        for line in (RUN_DIR / f"test2025_{arm}.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] != "__none__":
                preds[parts[0]].add(parts[1].zfill(3))
        evaluate(preds, arm)


if __name__ == "__main__":
    main()
