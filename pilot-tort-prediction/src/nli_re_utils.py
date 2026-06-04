from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_NLI_MODEL = "akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick"
DEFAULT_PREMISE_MODE = "uu_opponent"
DEFAULT_SCORE_MODE = "binary_margin"
SCORE_MODES = (
    "entailment",
    "binary_margin",
    "entail_plus_half_neutral",
    "entail_vs_contra",
)
PREMISE_MODES = (
    "uu",
    "uu_opponent",
    "uu_all",
)


def load_nli_bundle(
    model_name: str = DEFAULT_NLI_MODEL,
    *,
    device: torch.device | None = None,
) -> tuple[Any, Any, dict[str, int], int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model = model.to(device).eval()
    label_roles = infer_label_roles(model, tokenizer, device)
    max_len = int(getattr(model.config, "max_position_embeddings", 512))
    return model, tokenizer, label_roles, max_len


@torch.inference_mode()
def infer_label_roles(model, tokenizer, device: torch.device) -> dict[str, int]:
    probes = [
        ("太郎は学生です。", "太郎は学生です。"),
        ("今日は雨です。", "今日は晴れです。"),
    ]
    enc = tokenizer(
        [pair[0] for pair in probes],
        [pair[1] for pair in probes],
        max_length=128,
        truncation=True,
        padding="max_length",
        verbose=False,
        return_tensors="pt",
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    logits = model(**enc).logits

    entailment_idx = int(torch.argmax(logits[0]).item())
    contradiction_idx = int(torch.argmax(logits[1]).item())
    if entailment_idx == contradiction_idx:
        raise RuntimeError("Failed to infer NLI label mapping: identical and contradiction probes collapsed.")

    remaining = [idx for idx in range(int(logits.shape[-1])) if idx not in {entailment_idx, contradiction_idx}]
    if len(remaining) != 1:
        raise RuntimeError("Expected exactly one remaining neutral label.")

    return {
        "entailment": entailment_idx,
        "neutral": remaining[0],
        "contradiction": contradiction_idx,
    }


def build_claim_premise(
    case: dict[str, Any],
    party: str,
    *,
    mode: str = DEFAULT_PREMISE_MODE,
    max_chars: int = 1600,
) -> str:
    if mode not in PREMISE_MODES:
        raise ValueError(f"Unknown premise mode: {mode}")

    uu = " ".join(item["description"] for item in case.get("undisputed_facts", []))
    pp = " ".join(item["description"] for item in case.get("plaintiff_claims", []))
    dd = " ".join(item["description"] for item in case.get("defendant_claims", []))

    pieces = [f"[UU] {uu}"]
    if mode == "uu_opponent":
        if party == "P" and dd:
            pieces.append(f"[OPP] {dd}")
        elif party == "D" and pp:
            pieces.append(f"[OPP] {pp}")
    elif mode == "uu_all":
        if pp:
            pieces.append(f"[PP] {pp}")
        if dd:
            pieces.append(f"[DD] {dd}")

    return " ".join(piece for piece in pieces if piece).strip()[:max_chars]


def claim_hypothesis(claim: dict[str, Any], party: str) -> str:
    marker = "[原告]" if party == "P" else "[被告]"
    return f"{marker} {claim['description']}"


def score_from_nli(
    probs: tuple[float, float, float],
    logits: tuple[float, float, float],
    label_roles: dict[str, int],
    *,
    mode: str = DEFAULT_SCORE_MODE,
) -> float:
    if mode not in SCORE_MODES:
        raise ValueError(f"Unknown score mode: {mode}")

    p_entail = float(probs[label_roles["entailment"]])
    p_neutral = float(probs[label_roles["neutral"]])
    p_contra = float(probs[label_roles["contradiction"]])
    l_entail = float(logits[label_roles["entailment"]])
    l_contra = float(logits[label_roles["contradiction"]])

    if mode == "entailment":
        return p_entail
    if mode == "binary_margin":
        return 1.0 / (1.0 + math.exp(-(l_entail - l_contra)))
    if mode == "entail_plus_half_neutral":
        return p_entail + 0.5 * p_neutral

    denom = p_entail + p_contra
    if denom <= 1e-8:
        return 0.5
    return p_entail / denom


@torch.inference_mode()
def predict_nli_triplets(
    model,
    tokenizer,
    premises: list[str],
    hypotheses: list[str],
    *,
    device: torch.device,
    max_len: int,
    batch_size: int = 64,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    prob_rows: list[tuple[float, float, float]] = []
    logit_rows: list[tuple[float, float, float]] = []
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    pad_id = int(tokenizer.pad_token_id)

    for start in range(0, len(premises), batch_size):
        batch_premises = premises[start:start + batch_size]
        batch_hypotheses = hypotheses[start:start + batch_size]
        input_id_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        token_type_rows: list[list[int]] = []

        for premise, hypothesis in zip(batch_premises, batch_hypotheses):
            premise_ids = tokenizer.encode(
                premise,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )
            hypothesis_ids = tokenizer.encode(
                hypothesis,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )

            max_hyp_len = max(0, max_len - 3)
            if len(hypothesis_ids) > max_hyp_len:
                hypothesis_ids = hypothesis_ids[:max_hyp_len]

            max_prem_len = max(0, max_len - len(hypothesis_ids) - 3)
            if len(premise_ids) > max_prem_len:
                premise_ids = premise_ids[:max_prem_len]

            input_ids = [cls_id] + premise_ids + [sep_id] + hypothesis_ids + [sep_id]
            token_type_ids = [0] * (len(premise_ids) + 2) + [1] * (len(hypothesis_ids) + 1)
            attention_mask = [1] * len(input_ids)

            pad_len = max_len - len(input_ids)
            if pad_len < 0:
                input_ids = input_ids[:max_len]
                token_type_ids = token_type_ids[:max_len]
                attention_mask = attention_mask[:max_len]
            elif pad_len > 0:
                input_ids.extend([pad_id] * pad_len)
                token_type_ids.extend([0] * pad_len)
                attention_mask.extend([0] * pad_len)

            input_id_rows.append(input_ids)
            attention_rows.append(attention_mask)
            token_type_rows.append(token_type_ids)

        enc = {
            "input_ids": torch.tensor(input_id_rows, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
            "token_type_ids": torch.tensor(token_type_rows, dtype=torch.long, device=device),
        }
        logits = model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1)

        logit_rows.extend(tuple(float(x) for x in row) for row in logits.cpu().tolist())
        prob_rows.extend(tuple(float(x) for x in row) for row in probs.cpu().tolist())

    return prob_rows, logit_rows
