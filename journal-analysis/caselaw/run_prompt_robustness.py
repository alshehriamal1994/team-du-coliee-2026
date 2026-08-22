"""Is the count-permission effect a property of one wording, or of the permission?

run_singleclause_manipulation.py measured a single control against a single
treatment, one run each. The effect it reports is therefore conditioned on one
pair of wordings and on one sample from a stochastic endpoint. This script varies
both: three paraphrases of the count permission, each against the same deployed
control, and every arm run three times on the same day against the same endpoint.

Only the count-bearing wording changes between control and treatment. The
entailment criterion, the worked examples, the candidate list, the reranker
scores, the model and the temperature are identical across every arm, because the
candidates and demonstrations are read from the same frozen caches the deployed
run used.

Reads OPENROUTER_API_KEY from the environment. Writes expAA_numbers.json.
"""

import json
import os
import pickle
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "archive/runs_final_2026/test_cache_monot5v2.pkl"
FEWSHOT_PATH = HERE / "archive/runs_experiments/fewshot_cache_test.json"
LABELS_PATH = HERE / "task2_test_labels_2026(1).json"
SOURCE = HERE / "run_singleclause_manipulation.py"
DATA_DIR = Path(os.environ.get("COLIEE_ROOT", "data")) / "task2_test_files_2026"
OUTPUT_DIR = HERE / "runs_prompt_robustness"
OUT = HERE / "expAA_numbers.json"

MODEL = "deepseek/deepseek-chat"
TOP_K = 20
REPEATS = 3
B = 10_000
SEED = 20260822

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY not set")


def extract_prompt(name):
    """Take a prompt verbatim from the manipulation script, so the control here
    is byte-identical to the one that produced the published figure."""
    src = SOURCE.read_text()
    i = src.index(f"{name} = \"\"\"\\\n") + len(f"{name} = \"\"\"\\\n")
    j = src.index('"""', i)
    return src[i:j]


CONTROL = extract_prompt("PROMPT_CONTROL")
TREATMENT_A = extract_prompt("PROMPT_TREATMENT")

# Paraphrases B and C alter only the three count-bearing phrases of the control.
# Every other line, including the entailment criterion, is the control's.
TREATMENT_B = (CONTROL
    .replace("Identify the SINGLE candidate paragraph (if any) whose legal rule or principle",
             "Identify ALL candidate paragraphs whose legal rule or principle")
    .replace("- Select AT MOST ONE paragraph — the single most directly entailing",
             "- Select ALL paragraphs that entail the decision, however many there are")
    .replace('Return ONLY the paragraph ID (e.g., "033") or "none". Nothing else.',
             'Return ONLY the paragraph ID(s) separated by spaces (e.g., "033" or "012 033") or "none". Nothing else.'))

TREATMENT_C = (CONTROL
    .replace("Identify the SINGLE candidate paragraph (if any) whose legal rule or principle",
             "Identify each candidate paragraph whose legal rule or principle")
    .replace("- Select AT MOST ONE paragraph — the single most directly entailing",
             "- There is no limit on the number of paragraphs you may select")
    .replace('Return ONLY the paragraph ID (e.g., "033") or "none". Nothing else.',
             'Return ONLY the paragraph ID(s) separated by spaces (e.g., "033" or "012 033") or "none". Nothing else.'))

ARMS = {"control": CONTROL, "treatment_A": TREATMENT_A,
        "treatment_B": TREATMENT_B, "treatment_C": TREATMENT_C}

for name, text in ARMS.items():
    if name == "control":
        continue
    if text == CONTROL:
        sys.exit(f"GATE FAILED: {name} is identical to the control")
    if "FORCES the decision" not in text:
        sys.exit(f"GATE FAILED: {name} lost the entailment criterion")


def call(prompt, retries=8):
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


def parse_ids(text, valid):
    if text is None:
        return None
    if "none" in text.strip().lower()[:30] and not re.search(r"\d", text[:30]):
        return []
    seen, out = set(), []
    for f in re.findall(r"\b(\d{3})\b", text):
        if f in valid and f not in seen:
            seen.add(f); out.append(f)
    return out


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = json.load(open(LABELS_PATH))
    gold = {cid: {x.strip().replace(".txt", "").zfill(3)
                  for x in val.split(",") if x.strip()}
            for cid, val in raw.items()}
    cases = sorted(gold, key=int)
    total_gold = sum(len(g) for g in gold.values())

    cache = pickle.load(open(CACHE_PATH, "rb"))
    fewshot = json.load(open(FEWSHOT_PATH))

    # reranker fusion, copied from run_singleclause_manipulation.py
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

    def score(preds):
        tp = sum(len(set(p) & gold[c]) for c, p in preds.items())
        n = sum(len(p) for p in preds.values())
        P = tp / n if n else 0.0
        R = tp / total_gold
        return (2 * P * R / (P + R)) if P + R else 0.0

    results, per_case, failures = {}, {}, {}
    for arm, template in ARMS.items():
        for rep in range(1, REPEATS + 1):
            tag = f"{arm}_run{rep}"
            path = OUTPUT_DIR / f"{tag}.json"
            if path.exists():
                preds = {k: v for k, v in json.load(open(path)).items()}
            else:
                preds, fails = {}, 0
                for n, cid in enumerate(cases, 1):
                    case_dir = DATA_DIR / str(cid)
                    query = (case_dir / "entailed_fragment.txt").read_text(
                        encoding="utf-8", errors="replace").strip()
                    para_texts = {f.stem.zfill(3): f.read_text(
                        encoding="utf-8", errors="replace").strip()
                        for f in (case_dir / "paragraphs").glob("*.txt")}
                    top_items = [(pid, para_texts[pid])
                                 for pid, _ in scores_index[cid][:TOP_K]
                                 if pid in para_texts]
                    valid = {pid for pid, _ in top_items}
                    blocks = [f"[{pid}] " + (t[:1200] + "..." if len(t) > 1200 else t)
                              for pid, t in top_items]
                    ex = fewshot[cid][:3]
                    exb = []
                    for i, e in enumerate(ex, 1):
                        q = e["query"][:300] + "..." if len(e["query"]) > 300 else e["query"]
                        g = e["gold_text"][:500] + "..." if len(e["gold_text"]) > 500 else e["gold_text"]
                        exb.append(f"EXAMPLE {i}:\nDecision fragment: {q}\n"
                                   f"Entailing paragraph [{e['gold_pid']}]: {g}")
                    prompt = template.format(n_examples=len(ex),
                                             examples="\n\n".join(exb),
                                             query=query,
                                             paragraphs="\n\n".join(blocks))
                    out = parse_ids(call(prompt), valid)
                    if out is None:
                        fails += 1
                        out = []
                    preds[cid] = out
                    if n % 25 == 0:
                        print(f"    {tag}: {n}/{len(cases)}", flush=True)
                json.dump(preds, open(path, "w"))
                failures[tag] = fails
            per_case[tag] = preds
            results[tag] = round(score(preds), 4)
            print(f"  {tag}: F1 = {results[tag]}", flush=True)

    by_arm = {a: [results[f"{a}_run{r}"] for r in range(1, REPEATS + 1)]
              for a in ARMS}

    def paired_delta(arm):
        """Bootstrap the treatment-minus-control difference, averaging over runs."""
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(cases))
        deltas = np.empty(B)
        for b in range(B):
            take = rng.choice(idx, size=len(idx), replace=True)
            sub = [cases[i] for i in take]
            def sc(tag):
                tp = sum(len(set(per_case[tag][c]) & gold[c]) for c in sub)
                n = sum(len(per_case[tag][c]) for c in sub)
                g = sum(len(gold[c]) for c in sub)
                P, R = (tp / n if n else 0.0), (tp / g if g else 0.0)
                return (2 * P * R / (P + R)) if P + R else 0.0
            t = statistics.mean(sc(f"{arm}_run{r}") for r in range(1, REPEATS + 1))
            c = statistics.mean(sc(f"control_run{r}") for r in range(1, REPEATS + 1))
            deltas[b] = t - c
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return {"points": round(100 * float(deltas.mean()), 2),
                "ci95": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
                "prob_positive": round(float((deltas > 0).mean()), 4)}

    out = {
        "question": "is the count-permission effect a property of one wording and "
                    "one run, or of the permission",
        "protocol": {
            "model": MODEL, "temperature": 0.0, "repeats": REPEATS,
            "arms": list(ARMS), "cases": len(cases), "top_k": TOP_K,
            "control_and_treatment_A": "extracted verbatim from "
                                       "run_singleclause_manipulation.py",
            "paraphrases": "B and C alter only the three count-bearing phrases of "
                           "the control; the entailment criterion is unchanged",
            "bootstrap": f"case-level, B={B}, seed {SEED}, "
                         "difference of run-averaged scores",
        },
        "failed_calls": failures,
        "per_run_F1": results,
        "by_arm": {a: {"runs": v, "mean": round(statistics.mean(v), 4),
                       "spread": round(max(v) - min(v), 4)} for a, v in by_arm.items()},
        "effect_vs_control": {a: paired_delta(a) for a in ARMS if a != "control"},
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n--- summary ---")
    for a, v in out["by_arm"].items():
        print(f"  {a:12s} mean {v['mean']:.4f}  runs {v['runs']}  spread {v['spread']:.4f}")
    for a, e in out["effect_vs_control"].items():
        print(f"  {a} vs control: {e['points']:+.2f} pts  CI {e['ci95']}  "
              f"P(>0) {e['prob_positive']}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
