from __future__ import annotations

import json
import re
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.tp_retrieval_utils import build_retrieval_case_text, fit_retriever_bundle, predict_topk_probs


DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_GENERATION_MAX_NEW_TOKENS = 96
DEFAULT_BINARY_POSITIVE_LABEL = "1"
DEFAULT_BINARY_NEGATIVE_LABEL = "0"

_LLM_CACHE: dict[tuple[str, str], tuple[Any, Any, torch.device]] = {}


def _resolve_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_local_model_ref(model_name: str) -> str | PathLike[str]:
    direct = Path(model_name)
    if direct.exists():
        return direct
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_name.replace("/", "--"))
    snapshots = hub_dir / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()])
        if snapshot_dirs:
            return snapshot_dirs[-1]
    return model_name


def load_chat_bundle(model_name: str = DEFAULT_LLM_MODEL, *, device: str | None = None):
    dev = _resolve_device(device)
    cache_key = (model_name, str(dev))
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model_ref = _resolve_local_model_ref(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = model.to(dev).eval()
    _LLM_CACHE[cache_key] = (model, tokenizer, dev)
    return _LLM_CACHE[cache_key]


def fit_precedent_bundle(train_torts: list[Any], train_labels: list[int], *, max_chars: int = 2400) -> dict[str, Any]:
    texts = [build_retrieval_case_text(tort, max_chars=max_chars) for tort in train_torts]
    bundle = fit_retriever_bundle(texts, train_labels)
    bundle["texts"] = texts
    bundle["tort_ids"] = [str(tort.tort_id) for tort in train_torts]
    bundle["max_chars"] = int(max_chars)
    return bundle


def retrieve_precedents(
    query_tort: Any,
    precedent_bundle: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    vectorizer = precedent_bundle["vectorizer"]
    ref_matrix = precedent_bundle["ref_matrix"]
    ref_labels = precedent_bundle["ref_labels"]
    text = build_retrieval_case_text(query_tort, max_chars=int(precedent_bundle.get("max_chars", 2400)))
    query_matrix = vectorizer.transform([text])
    probs = predict_topk_probs(
        query_matrix,
        ref_matrix,
        ref_labels,
        top_k=top_k,
        fallback_prob=float(precedent_bundle["positive_rate"]),
    )
    sims = (query_matrix @ ref_matrix.T).toarray()[0]
    k = min(int(top_k), ref_matrix.shape[0])
    idx = list(range(len(sims))) if k == len(sims) else list(torch.tensor(sims).topk(k).indices.cpu().tolist())
    rows: list[dict[str, Any]] = []
    for rank, j in enumerate(idx, start=1):
        rows.append({
            "rank": rank,
            "tort_id": precedent_bundle["tort_ids"][j],
            "label": bool(float(ref_labels[j]) >= 0.5),
            "similarity": float(sims[j]),
            "text": precedent_bundle["texts"][j],
        })
    # attach aggregate retrieval prob to first row set consumers can use separately
    if rows:
        rows[0]["aggregate_prob"] = float(probs[0])
    return rows


def shorten(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def build_japanese_arbiter_prompt(
    tort: Any,
    meta: dict[str, Any],
    precedents: list[dict[str, Any]],
) -> list[dict[str, str]]:
    uu = shorten(" ".join(item.description for item in tort.undisputed_facts), 500)
    p_claims = [shorten(item.description, 180) for item in tort.plaintiff_claims[:4]]
    d_claims = [shorten(item.description, 180) for item in tort.defendant_claims[:4]]
    p_accepted = [shorten(str(text), 220) for text in meta.get("accepted_plaintiff_claims", [])][:6]
    d_accepted = [shorten(str(text), 220) for text in meta.get("accepted_defendant_claims", [])][:6]
    tp_threshold = float(meta.get("tp_threshold", 0.5))
    case_prob = float(meta.get("case_prob", meta.get("case_prob_joint", 0.5)))
    retrieval_prob = meta.get("tp_retrieval_prob")
    dominance = meta.get("dominance_score")

    precedent_lines = []
    for row in precedents:
        label = "True" if bool(row["label"]) else "False"
        precedent_lines.append(
            f"- 類似判例{row['rank']} (tort_id={row['tort_id']}, 類似度={row['similarity']:.3f}, 判決={label})\n"
            f"  {shorten(row['text'], 260)}"
        )
    precedent_text = "\n".join(precedent_lines) if precedent_lines else "なし"

    user_prompt = (
        "以下は日本の不法行為事件です。あなたは裁判官補助AIとして、与えられた事件と類似判例を参考に、"
        "最終的な court_decision（True/False）のみを慎重に判断してください。\n\n"
        "判断ルール:\n"
        "- True は原告側の不法行為主張を裁判所が認める方向\n"
        "- False は原告側の不法行為主張を裁判所が認めない方向\n"
        "- まず『受け入れられた主張』の整合性を最優先で確認すること\n"
        "- 類似判例は参考情報であり、現在事件の事実と主張を優先すること\n"
        "- 出力は必ずJSONのみ。余計な説明は禁止\n\n"
        f"現在システムの暫定情報:\n"
        f"- 現在のTP確率: {case_prob:.4f}\n"
        f"- 現在のTP閾値: {tp_threshold:.4f}\n"
        + (f"- 取得判例ベースの確率: {float(retrieval_prob):.4f}\n" if retrieval_prob is not None else "")
        + (f"- 原告優勢スコア: {int(dominance)}\n" if dominance is not None else "")
        + "\n事件情報:\n"
        f"- 争いのない事実: {uu or 'なし'}\n"
        "- 原告の主張(先頭のみ):\n"
        + ("\n".join(f"  - {claim}" for claim in p_claims) if p_claims else "  - なし")
        + "\n- 被告の主張(先頭のみ):\n"
        + ("\n".join(f"  - {claim}" for claim in d_claims) if d_claims else "  - なし")
        + "\n\nシステムが受け入れた主張（RE予測）:\n"
        + "- 原告側で受け入れ:\n"
        + ("\n".join(f"  - {claim}" for claim in p_accepted) if p_accepted else "  - なし")
        + "\n- 被告側で受け入れ:\n"
        + ("\n".join(f"  - {claim}" for claim in d_accepted) if d_accepted else "  - なし")
        + "\n\n類似判例:\n"
        + precedent_text
        + "\n\n"
        + "出力フォーマット:\n"
        + '{"court_decision": true/false, "confidence": 0.0-1.0, "rationale": "accepted-claims consistency"}'
    )

    system_prompt = (
        "あなたは日本の民事不法行為訴訟（民法709条）の判決予測を補助する厳格な法務AIです。"
        "与えられた情報だけで慎重に判断し、JSON以外を出力してはいけません。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_japanese_binary_judge_prompt(
    tort: Any,
    meta: dict[str, Any],
    precedents: list[dict[str, Any]],
) -> list[dict[str, str]]:
    uu = shorten(" ".join(item.description for item in tort.undisputed_facts), 360)
    p_claims = [shorten(item.description, 120) for item in tort.plaintiff_claims[:3]]
    d_claims = [shorten(item.description, 120) for item in tort.defendant_claims[:3]]
    tp_threshold = float(meta.get("tp_threshold", 0.5))
    case_prob = float(meta.get("case_prob", meta.get("case_prob_joint", 0.5)))
    retrieval_prob = meta.get("tp_retrieval_prob")

    precedent_lines = []
    for row in precedents:
        label = "1" if bool(row["label"]) else "0"
        precedent_lines.append(
            f"- precedent_{row['rank']}: sim={row['similarity']:.3f}, decision={label}, "
            f"facts={shorten(row['text'], 180)}"
        )
    precedent_text = "\n".join(precedent_lines) if precedent_lines else "- none"

    user_prompt = (
        "日本の民事不法行為事件について最終 court_decision を二値判定してください。\n"
        "意味:\n"
        "- 1 = 原告の不法行為主張を認める\n"
        "- 0 = 原告の不法行為主張を認めない\n"
        "出力は 1 または 0 の1文字のみ。説明禁止。\n\n"
        f"base_prob={case_prob:.4f}\n"
        f"base_threshold={tp_threshold:.4f}\n"
        + (f"retrieval_prob={float(retrieval_prob):.4f}\n" if retrieval_prob is not None else "")
        + f"facts={uu or 'none'}\n"
        + "plaintiff_claims:\n"
        + ("\n".join(f"- {claim}" for claim in p_claims) if p_claims else "- none")
        + "\n"
        + "defendant_claims:\n"
        + ("\n".join(f"- {claim}" for claim in d_claims) if d_claims else "- none")
        + "\n"
        + "similar_precedents:\n"
        + precedent_text
        + "\n"
        + "Answer with 1 or 0 only."
    )

    system_prompt = (
        "You are a strict Japanese tort-law decision assistant. "
        "Return only a single binary label: 1 or 0."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_chat_json(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    device: torch.device,
    max_new_tokens: int = DEFAULT_GENERATION_MAX_NEW_TOKENS,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_binary_choice(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    device: torch.device,
    positive_label: str = DEFAULT_BINARY_POSITIVE_LABEL,
    negative_label: str = DEFAULT_BINARY_NEGATIVE_LABEL,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]

    pos_ids = tokenizer.encode(positive_label, add_special_tokens=False)
    neg_ids = tokenizer.encode(negative_label, add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError("Binary labels must tokenize to a single token.")

    pos_logit = float(logits[pos_ids[0]])
    neg_logit = float(logits[neg_ids[0]])
    pair = torch.tensor([neg_logit, pos_logit], dtype=torch.float32)
    probs = torch.softmax(pair, dim=0)
    prob_false = float(probs[0])
    prob_true = float(probs[1])
    pred_true = prob_true >= prob_false
    raw_label = positive_label if pred_true else negative_label
    confidence = max(prob_true, prob_false)
    return {
        "court_decision": bool(pred_true),
        "confidence": float(confidence),
        "prob_true": prob_true,
        "prob_false": prob_false,
        "raw_label": raw_label,
    }


def parse_arbiter_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.S))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if "court_decision" not in obj:
            continue
        try:
            decision = bool(obj["court_decision"])
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            continue
        rationale = str(obj.get("rationale", ""))
        return {
            "court_decision": decision,
            "confidence": confidence,
            "rationale": rationale,
            "raw_json": cand,
        }
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
