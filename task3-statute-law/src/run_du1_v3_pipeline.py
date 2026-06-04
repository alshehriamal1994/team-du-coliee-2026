#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DU1-v3: Dynamic K + LLM Reranking Pipeline
============================================

What this adds over v2:
  1. Dynamic K: elbow detection on 3-way RRF scores per query
     - Returns K*=3–30 articles (vs fixed 30 in v2)
     - Improves F2 tiebreaker score on confident queries
  2. LLM reranking: Qwen2.5-7B listwise reranking (1 call/query)
     - Puts the most relevant article first in the entailment context
     - Much cheaper than 30-article-scoring, smarter than RRF order alone
  3. Entailment: Qwen2.5-72B, top-15 from reranked list (was top-25 from RRF order)

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  cd .
  python3 DU1/run_du1_v3_pipeline.py 2>&1 | tee DU1/output/v3_run.log
"""

import os
import argparse
import json, math, os, re, time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
QUERIES_PATH    = "data/test_task3_norm.jsonl"
CIVIL_INDEX     = "system/cache/civil_index_v2.json"
BM25_CACHE      = "system/cache/du1_bm25_test.jsonl"
TFIDF_CACHE     = "system/cache/du1_tfidf_test.jsonl"
BGEM3_CACHE     = "system/cache/du1v2_bgem3_test.jsonl"
OUT_CACHE_DIR   = "DU1/output/cache"
OUT_SUB_DIR     = "DU1"
RUN_TAG         = "DU1"

# Models
RERANKER_MODEL  = "qwen/qwen-2.5-7b-instruct"    # cheap + fast, for ranking only
ENTAIL_MODEL    = "qwen/qwen-2.5-72b-instruct"   # powerful, for Y/N decision
RERANK_TOP      = 10   # rerank only top-10 (shorter prompt, faster, avoids timeout)

# Dynamic K parameters
DYN_K_MIN       = 3     # never return fewer than this
DYN_K_MAX       = 30    # cap (so we don't go beyond what we retrieved)
DYN_K_GAP       = 0.20  # 20% relative drop in RRF score = elbow

# Entailment parameters  
RRF_K           = 60
ENTAIL_TOPM     = 15    # top articles after reranking (was 25 in v2 before reranking)
SC_SAMPLES      = 5
SC_TEMP         = 0.4
MIN_CONF        = 70
SLEEP           = 0.3

# Legal markers for snippet extraction
MARKERS = ["ただし","この限りでない","を除く","にかかわらず",
           "場合","とき","できない","してはならない","妨げない",
           "善意","悪意","前条","前項","次条","第"]

EXCEPTION_CUES = ["ただし", "この限りでない", "を除く", "にかかわらず", "善意", "悪意", "妨げない"]

PROFILES = {
    "safe": {
        "dyn_k_gap": 0.20,
        "dyn_k_min": 3,
        "dyn_k_max": 30,
        "rerank_top": 10,
        "entail_topm": 15,
        "sc_samples": 5,
        "sc_temp": 0.4,
        "min_conf": 70,
        "exception_sc_policy": "query",
    },
    "balanced": {
        "dyn_k_gap": 0.12,
        "dyn_k_min": 2,
        "dyn_k_max": 20,
        "rerank_top": 12,
        "entail_topm": 15,
        "sc_samples": 6,
        "sc_temp": 0.4,
        "min_conf": 72,
        "exception_sc_policy": "query",
    },
    "aggressive": {
        "dyn_k_gap": 0.05,
        "dyn_k_min": 1,
        "dyn_k_max": 15,
        "rerank_top": 12,
        "entail_topm": 12,
        "sc_samples": 7,
        "sc_temp": 0.45,
        "min_conf": 75,
        "exception_sc_policy": "context",
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

GOLD_ID_RE = re.compile(r"^\d+(?:-\d+)?$")
def norm_id(x):
    s = str(x).strip()
    if GOLD_ID_RE.match(s): return s
    m = re.search(r"(\d+(?:-\d+)?)", s)
    if not m: return None
    s = m.group(1)
    return s if GOLD_ID_RE.match(s) else None


def parse_args():
    ap = argparse.ArgumentParser(description="DU1-v3 Dynamic-K + rerank + entailment pipeline")
    ap.add_argument("--run-tag", default=RUN_TAG, help="Run tag used in submission files")
    ap.add_argument("--profile", choices=["custom", "safe", "balanced", "aggressive"], default="custom")
    ap.add_argument("--dyn-k-min", type=int, default=DYN_K_MIN)
    ap.add_argument("--dyn-k-max", type=int, default=DYN_K_MAX)
    ap.add_argument("--dyn-k-gap", type=float, default=DYN_K_GAP)
    ap.add_argument("--rrf-k", type=int, default=RRF_K)
    ap.add_argument("--rerank-top", type=int, default=RERANK_TOP)
    ap.add_argument("--entail-topm", type=int, default=ENTAIL_TOPM)
    ap.add_argument("--sc-samples", type=int, default=SC_SAMPLES)
    ap.add_argument("--sc-temp", type=float, default=SC_TEMP)
    ap.add_argument("--min-conf", type=int, default=MIN_CONF)
    ap.add_argument("--sleep", type=float, default=SLEEP)
    ap.add_argument(
        "--exception-sc-policy",
        choices=["off", "query", "context"],
        default="query",
        help="Force self-consistency on exception-heavy queries/context",
    )
    return ap.parse_args()


def apply_profile(args):
    if args.profile == "custom":
        return args
    p = PROFILES[args.profile]
    args.dyn_k_gap = float(p["dyn_k_gap"])
    args.dyn_k_min = int(p["dyn_k_min"])
    args.dyn_k_max = int(p["dyn_k_max"])
    args.rerank_top = int(p["rerank_top"])
    args.entail_topm = int(p["entail_topm"])
    args.sc_samples = int(p["sc_samples"])
    args.sc_temp = float(p["sc_temp"])
    args.min_conf = int(p["min_conf"])
    args.exception_sc_policy = str(p["exception_sc_policy"])
    return args


# ─────────────────────────────────────────────
# STEP 1: 3-way RRF with scores
# ─────────────────────────────────────────────
def rrf_fuse_with_scores(runs: List[Dict[str, List[str]]], qids: List[str],
                          rrf_k: int = 60, max_len: int = 30
                          ) -> Dict[str, List[Tuple[str, float]]]:
    """RRF fusion returning (article, score) pairs sorted descending."""
    fused = {}
    for qid in qids:
        score: Dict[str, float] = defaultdict(float)
        for run in runs:
            for rank, art in enumerate(run.get(qid, []), 1):
                score[art] += 1.0 / (rrf_k + rank)
        ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = ranked[:max_len]
    return fused


# ─────────────────────────────────────────────
# STEP 2: Dynamic K elbow detection
# ─────────────────────────────────────────────
def dynamic_k(scored: List[Tuple[str, float]],
              min_k: int = 3, max_k: int = 30,
              gap_threshold: float = 0.20) -> int:
    """
    Find the elbow in the RRF score distribution.
    Returns K* = first index where relative drop exceeds gap_threshold.
    Clamped to [min_k, max_k].
    """
    scores = [s for _, s in scored[:max_k]]
    n = len(scores)
    if n <= min_k:
        return n

    for i in range(min_k - 1, n - 1):  # i is 0-indexed → K = i+1
        if scores[i] > 1e-9:
            relative_drop = (scores[i] - scores[i + 1]) / scores[i]
            if relative_drop >= gap_threshold:
                return i + 1  # include articles 0..i

    return min(n, max_k)


def apply_dynamic_k(fused: Dict[str, List[Tuple[str, float]]],
                    min_k=3, max_k=30, gap=0.20
                    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Returns {qid: [arts]} with dynamic K, and {qid: K*} for diagnostics."""
    retrieval = {}
    k_values  = {}
    for qid, scored in fused.items():
        k = dynamic_k(scored, min_k, max_k, gap)
        retrieval[qid] = [art for art, _ in scored[:k]]
        k_values[qid]  = k
    return retrieval, k_values


# ─────────────────────────────────────────────
# STEP 3: LLM Reranking (Qwen2.5-7B, listwise)
# ─────────────────────────────────────────────
RERANK_SYSTEM = (
    "あなたは日本の民法の検索エキスパートです。"
    "与えられた条文候補を、設問（問い）への法的関連度が高い順に並べ替えてください。"
    "関連度の判断基準：条文の内容が設問の状況に直接適用されるか。"
    "JSONのみを出力してください。"
)

def snippet_short(text: str, cap: str, max_len: int = 180) -> str:
    base = (cap.strip() + "　" + text.strip()) if cap and cap.strip() else text.strip()
    if not base: return ""
    hits = [p for m in MARKERS if (p := base.find(m)) != -1]
    if hits:
        p = min(hits)
        return base[max(0, p-80): p+max_len].replace("\n", " ").strip()
    return base[:max_len].replace("\n", " ").strip()


def build_rerank_prompt(query: str, arts: List[str],
                        text_map: Dict, cap_map: Dict) -> str:
    lines = [f"【設問】{query}\n\n以下の条文候補を関連度の高い順に並べ替えよ。\n"]
    for i, aid in enumerate(arts, 1):
        nid = norm_id(aid) or aid
        raw = text_map.get(aid) or text_map.get(nid) or {}
        at = raw.get("text", "") if isinstance(raw, dict) else (raw if isinstance(raw, str) else "")
        cap = cap_map.get(aid) or cap_map.get(nid) or ""
        s = snippet_short(at, cap if isinstance(cap, str) else "")
        lines.append(f"[{i}] 第{nid}条: {s}")
    lines.append(f'\nJSONのみ出力: {{"ranking": [候補番号1, 候補番号2, ...]}} (1〜{len(arts)}の全番号を含む)')
    return "\n".join(lines)


def parse_ranking(resp: str, n: int) -> List[int]:
    """Parse JSON ranking output. Returns 1-indexed positions list."""
    m = re.search(r'"ranking"\s*:\s*\[([^\]]+)\]', resp)
    if m:
        try:
            vals = [int(x.strip()) for x in m.group(1).split(",")]
            valid = [v for v in vals if 1 <= v <= n]
            # Append any missing indices (safety)
            seen = set(valid)
            for i in range(1, n + 1):
                if i not in seen:
                    valid.append(i)
            return valid[:n]
        except Exception:
            pass
    return list(range(1, n + 1))  # fallback: keep original order


def llm_call(api_key: str, model: str, system: str, user: str,
             temperature: float = 0.0, max_tokens: int = 512) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_reranking(queries, retrieval_full: Dict[str, List[str]],
                  text_map, cap_map, api_key: str, out_path: str,
                  top_n: int = 10) -> Dict[str, List[str]]:
    """Rerank the top-N retrieved articles per query for context ordering.
    After reranking top-N, the remaining articles are appended in original RRF order."""
    done = {}
    if os.path.exists(out_path):
        for o in read_jsonl(out_path):
            done[o["id"]] = o["reranked_arts"]
        print(f"[rerank] Resuming — {len(done)} done")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results = dict(done)

    with open(out_path, "a", encoding="utf-8") as fout:
        for q in queries:
            qid = q["id"]
            if qid in done:
                continue
            query = q["t2"]
            # Only rerank top-N (shorter, faster prompt)
            all_arts = retrieval_full.get(qid, [])
            arts     = all_arts[:top_n]
            if not arts:
                results[qid] = []
                fout.write(json.dumps({"id": qid, "reranked_arts": []}) + "\n")
                continue

            prompt_user = build_rerank_prompt(query, arts, text_map, cap_map)
            try:
                resp = llm_call(api_key, RERANKER_MODEL, RERANK_SYSTEM,
                                prompt_user, temperature=0.0, max_tokens=128)
                ranking = parse_ranking(resp, len(arts))
                reranked = [arts[i - 1] for i in ranking]
            except Exception as e:
                print(f"  [WARN rerank] {qid}: {e} — keeping original order")
                reranked = arts  # fallback

            # Append tail (articles beyond top_n) in original RRF order
            tail         = [a for a in all_arts if a not in set(reranked)]
            full_reranked = reranked + tail

            results[qid] = full_reranked
            fout.write(json.dumps({"id": qid, "reranked_arts": full_reranked},
                                   ensure_ascii=False) + "\n")
            print(f"  [rerank] {qid}: rank1={reranked[0] if reranked else 'none'} (was {arts[0] if arts else 'none'})")
            time.sleep(SLEEP)

    print(f"[rerank] Done. {len(results)} queries reranked.")
    return results


# ─────────────────────────────────────────────
# STEP 4: Entailment (Qwen2.5-72B, top-15 reranked)
# ─────────────────────────────────────────────
ENTAIL_SYSTEM = """\
あなたは日本の民法に精通した法務エキスパートです。与えられた「条文の断片」のみを根拠として、問いに対する「正誤（Entailment）」を判定してください。

【判定の厳格な手順】
1. 要件の分解：条文が規定する「法律要件」をすべて書き出し、問いの状況がそれらをすべて満たしているか確認せよ。
2. 「ただし書」と「除外規定」の優先確認：条文内に「ただし」「この限りでない」「〜を除き」「妨げない」という表現がある場合、それが問いのケースに該当しないか最優先でチェックせよ。
3. 消極的判断の優先：根拠が不十分な場合、または条文の要件を一つでも欠く場合は、安易に「Y」とせず「N」と判定せよ。
4. 出力形式：最後にJSONのみを出力せよ。{"label":"Y|N","confidence":0-100}"""


def snippet_long(text: str, cap: str, max_len: int = 300) -> str:
    base = (cap.strip() + "\n" + text.strip()).strip() if cap and cap.strip() else text.strip()
    if not base: return ""
    hits = [p for m in MARKERS if (p := base.find(m)) != -1]
    if hits:
        p = min(hits)
        return base[max(0, p-120): p+max_len].replace("\n", " ").strip()
    return base[:max_len].replace("\n", " ").strip()


def build_entail_context(arts, text_map, cap_map, topm):
    parts = []
    for aid in arts[:topm]:
        nid = norm_id(aid) or aid
        raw = text_map.get(aid) or text_map.get(nid) or {}
        at  = raw.get("text", "") if isinstance(raw, dict) else (raw if isinstance(raw, str) else "")
        cap = cap_map.get(aid) or cap_map.get(nid) or ""
        s = snippet_long(at, cap if isinstance(cap, str) else "")
        if s:
            parts.append(f"【第{nid}条】 {s}")
    return "\n".join(parts)


def build_entail_user(query, ctx):
    return f"""【参照条文】
{ctx}
---
【問い】
{query}
---
【分析ステップ】
1. 本問の状況に直接適用される条文番号を特定せよ。
2. その条文の「原則」と「例外（ただし書）」を対比せよ。
3. 本問の結論を導け。

出力（JSONのみ）:
{{"label":"Y|N","confidence":0-100}}"""


def parse_response(resp):
    m = re.search(r'\{.*\}', resp, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            lab = str(o.get("label","")).strip().upper()
            if lab in ("Y","N"):
                return lab, int(o.get("confidence", 60))
        except Exception:
            pass
    for pat in [r'判定\s*[:：]\s*([YN])', r'Result:\s*([YN])', r'\b([YN])\b']:
        m = re.search(pat, resp, re.IGNORECASE)
        if m: return m.group(1).upper(), 60
    return "N", 0

def majority(votes): return "Y" if votes.count("Y") > votes.count("N") else "N"


def should_force_self_consistency(query: str, context: str, policy: str) -> bool:
    if policy == "off":
        return False
    hay = query if policy == "query" else (query + "\n" + context)
    return any(cue in hay for cue in EXCEPTION_CUES)


def run_entailment(queries, reranked: Dict[str, List[str]],
                   text_map, cap_map, api_key, out_path,
                   entail_topm: int, sc_samples: int, sc_temp: float,
                   min_conf: int, sleep_sec: float, exception_sc_policy: str):
    done = {}
    if os.path.exists(out_path):
        for o in read_jsonl(out_path): done[o["id"]] = o
        print(f"[entail] Resuming — {len(done)} done")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results = dict(done)

    with open(out_path, "a", encoding="utf-8") as fout:
        for q in queries:
            qid = q["id"]
            if qid in done: continue
            query = q["t2"]
            arts  = reranked.get(qid, [])

            if not arts:
                rec = {"id":qid,"pred_label":"N","confidence":0,"votes":[],"note":"no_arts"}
                fout.write(json.dumps(rec) + "\n"); results[qid] = rec; continue

            # Top-15 from reranked list (highest quality context after reranking)
            ctx      = build_entail_context(arts, text_map, cap_map, entail_topm)
            user_msg = build_entail_user(query, ctx)

            try:
                resp0 = llm_call(api_key, ENTAIL_MODEL, ENTAIL_SYSTEM, user_msg, 0.0, 512)
                lab0, conf0 = parse_response(resp0)
            except Exception as e:
                print(f"  [WARN entail] {qid}: {e}")
                lab0, conf0 = "N", 0

            votes = [lab0]
            confs = [conf0]

            need_sc = (conf0 < min_conf) or should_force_self_consistency(query, ctx, exception_sc_policy)
            if need_sc:
                for _ in range(max(0, sc_samples - 1)):
                    time.sleep(sleep_sec)
                    try:
                        r = llm_call(api_key, ENTAIL_MODEL, ENTAIL_SYSTEM, user_msg, sc_temp, 512)
                        lb, cf = parse_response(r)
                        if lb in ("Y","N"): votes.append(lb); confs.append(cf)
                    except Exception: pass

            pred     = majority(votes)
            conf_med = sorted(confs)[len(confs)//2]
            margin   = abs(votes.count("Y") - votes.count("N"))

            # Escalate on split vote
            if margin <= 1 or conf_med < min_conf:
                ctx2  = build_entail_context(arts, text_map, cap_map, 25)
                user2 = build_entail_user(query, ctx2)
                v2, c2 = [], []
                for _ in range(sc_samples):
                    time.sleep(sleep_sec)
                    try:
                        r = llm_call(api_key, ENTAIL_MODEL, ENTAIL_SYSTEM, user2, sc_temp, 512)
                        lb, cf = parse_response(r)
                        if lb in ("Y","N"): v2.append(lb); c2.append(cf)
                    except Exception: pass
                if v2:
                    pred = majority(v2); conf_med = sorted(c2)[len(c2)//2]; votes = v2

            rec = {"id":qid,"pred_label":pred,"confidence":int(conf_med),"votes":votes}
            fout.write(json.dumps(rec, ensure_ascii=False)+"\n")
            results[qid] = rec
            print(f"  {qid}: {pred} conf={conf_med} votes={votes}")
            time.sleep(sleep_sec)

    preds = list(results.values())
    y = sum(1 for p in preds if p.get("pred_label")=="Y")
    n = sum(1 for p in preds if p.get("pred_label")=="N")
    un = sum(1 for p in preds if len(set(p.get("votes",[])))<=1)
    ac = sum(p.get("confidence",0) for p in preds)/max(len(preds),1)
    print(f"[entail] Done: Y={y}, N={n}, unanimous={un}/{len(preds)}, avg_conf={ac:.1f}")
    return results


# ─────────────────────────────────────────────
# STEP 5: Export submission files
# ─────────────────────────────────────────────
def export(queries, retrieval_dynK, predictions, run_tag, trec_path, ent_path):
    os.makedirs(os.path.dirname(trec_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ent_path)  or ".", exist_ok=True)

    with open(ent_path, "w", encoding="ascii", errors="strict") as fe:
        for q in queries:
            qid = q["id"]
            lab = (predictions.get(qid) or {}).get("pred_label", "N")
            fe.write(f"{qid} {lab} {run_tag}\n")

    with open(trec_path, "w", encoding="ascii", errors="strict") as ft:
        n_lines = 0
        for q in queries:
            qid  = q["id"]
            arts = retrieval_dynK.get(qid, [])   # ← Dynamic K set
            for rank, art in enumerate(arts, 1):
                ft.write(f"{qid} Q0 {art} {rank} {1.0/rank:.6f} {run_tag}\n")
                n_lines += 1

    print(f"[export] Entailment: {ent_path} ({len(queries)} rows)")
    print(f"[export] Retrieval : {trec_path} ({n_lines} lines)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = apply_profile(parse_args())
    api_key = os.environ.get("OPENROUTER_API_KEY","").strip()
    if not api_key: raise SystemExit("Set OPENROUTER_API_KEY first.")

    safe_tag = re.sub(r"[^A-Za-z0-9_]+", "", str(args.run_tag).strip()) or RUN_TAG
    out_retrieval = os.path.join(OUT_CACHE_DIR, f"v3_retrieval_dynamicK_{safe_tag}.jsonl")
    out_reranked = os.path.join(OUT_CACHE_DIR, f"v3_reranked_{safe_tag}.jsonl")
    out_entail = os.path.join(OUT_CACHE_DIR, f"v3_entailment_preds_{safe_tag}.jsonl")
    out_trec = os.path.join(OUT_SUB_DIR, f"task3_retrieval_{safe_tag}.txt")
    out_ent_file = os.path.join(OUT_SUB_DIR, f"task3_entailment_{safe_tag}.txt")

    print("="*60)
    print("DU1-v3 Pipeline: Dynamic K + LLM Reranking")
    print("="*60)
    print(
        f"profile={args.profile} run_tag={safe_tag} "
        f"dyn_k(min={args.dyn_k_min},max={args.dyn_k_max},gap={args.dyn_k_gap}) "
        f"rerank_top={args.rerank_top} entail_topm={args.entail_topm} "
        f"sc_samples={args.sc_samples} min_conf={args.min_conf} "
        f"exception_sc={args.exception_sc_policy}"
    )

    # Load index
    d = json.load(open(CIVIL_INDEX, encoding="utf-8"))
    text_map = d["text"]; cap_map = d.get("caption", {})

    # Load queries
    queries = list(read_jsonl(QUERIES_PATH))
    qids = [q["id"] for q in queries]
    print(f"Queries: {len(queries)}")

    # Load 3 retrieval lists (already cached)
    def load_run(path, field="retrieved_arts"):
        return {o["id"]: o.get(field) or o.get("articles") or []
                for o in read_jsonl(path)}

    bm25  = load_run(BM25_CACHE)
    tfidf = load_run(TFIDF_CACHE)
    bgem3 = load_run(BGEM3_CACHE)
    print("Loaded BM25, TF-IDF, BGE-M3 caches.")

    # Step 1: 3-way RRF with scores (top-30 for reranking + dynamic K)
    fused_scored = rrf_fuse_with_scores([bm25, tfidf, bgem3], qids,
                                         rrf_k=args.rrf_k, max_len=30)

    # Step 2: Dynamic K detection (for FINAL retrieval submission file)
    retrieval_dynK, k_values = apply_dynamic_k(
        fused_scored, min_k=args.dyn_k_min, max_k=args.dyn_k_max, gap=args.dyn_k_gap)

    k_dist = Counter(k_values.values())
    avg_k  = sum(k_values.values()) / len(k_values)
    print(f"\n[Dynamic K] K distribution: {dict(sorted(k_dist.items()))}")
    print(f"[Dynamic K] Avg K* = {avg_k:.1f} (was fixed 30 in v2)")

    # Save Dynamic K retrieval for export
    os.makedirs(os.path.dirname(out_retrieval) or ".", exist_ok=True)
    with open(out_retrieval, "w", encoding="utf-8") as f:
        for qid in qids:
            f.write(json.dumps({"id": qid, "retrieved_arts": retrieval_dynK.get(qid, []),
                                 "k": k_values.get(qid, 0)}, ensure_ascii=False) + "\n")

    # Step 3: LLM Reranking of FULL top-30 (for entailment context ordering)
    # Use full top-30 (not dynamic K) for reranking — we want all candidates considered
    retrieval_full = {qid: [art for art, _ in scored] for qid, scored in fused_scored.items()}
    reranked = run_reranking(
        queries, retrieval_full, text_map, cap_map, api_key, out_reranked, top_n=args.rerank_top
    )

    # Step 4: Entailment with reranked context
    predictions = run_entailment(
        queries=queries,
        reranked=reranked,
        text_map=text_map,
        cap_map=cap_map,
        api_key=api_key,
        out_path=out_entail,
        entail_topm=args.entail_topm,
        sc_samples=args.sc_samples,
        sc_temp=args.sc_temp,
        min_conf=args.min_conf,
        sleep_sec=args.sleep,
        exception_sc_policy=args.exception_sc_policy,
    )

    # Step 5: Export — use Dynamic K for retrieval, reranked for entailment
    export(queries, retrieval_dynK, predictions, safe_tag, out_trec, out_ent_file)

    # Final validation summary
    print("\n=== FINAL VALIDATION ===")
    ent_lines = open(out_ent_file).readlines()
    y = sum(1 for l in ent_lines if " Y " in l)
    n = sum(1 for l in ent_lines if " N " in l)
    bad = [l for l in ent_lines if len(l.split()) != 3 or l.split()[1] not in ("Y","N")]
    ret_lines = open(out_trec).readlines()
    q_ret = set(l.split()[0] for l in ret_lines)
    q_ent = set(l.split()[0] for l in ent_lines)
    print(f"Entailment: {len(ent_lines)} rows, Y={y}, N={n}, bad={len(bad)}")
    print(f"Retrieval:  {len(ret_lines)} lines, {len(q_ret)} queries, match={q_ret==q_ent}")
    print(f"✓ DU1-v3 complete." if not bad and q_ret==q_ent else "✗ ERRORS FOUND")


if __name__ == "__main__":
    main()
