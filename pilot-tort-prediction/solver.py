"""
Integrated solver for the LJPJT-26 pilot task.

Core model:
  models/joint_bert_v2_final

Current best validated production profile:
- joint TP case head
- RE stacker v3 (joint + auxiliary RE + rationale retrieval)
- TP case fuser
- precedent-retrieval TP branch
- ultra-tight TP repair only near the decision threshold

The lower-level `solve_with_details()` function still exposes all switches for
offline ablations. The top-level `solve()` function now runs the current
best-known integrated pipeline by default so the production path matches the
best measured local system instead of the older plain joint baseline.
"""

from __future__ import annotations

import configparser
import json
import logging
import math
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.nli_re_utils import (
    DEFAULT_NLI_MODEL,
    DEFAULT_PREMISE_MODE,
    DEFAULT_SCORE_MODE,
    build_claim_premise,
    claim_hypothesis,
    load_nli_bundle,
    predict_nli_triplets,
    score_from_nli,
)
from src.re_rationale_retrieval_utils import (
    build_claim_retrieval_text,
    predict_retrieval_probs as predict_re_rationale_probs,
)
from src.models.defendant_claim import DefendantClaim
from src.models.plaintiff_claim import PlaintiffClaim
from src.models.tort import Tort
from src.tp_case_fuser_utils import build_tp_case_feature_vector
from src.tp_pairwise_conflict_utils import build_pairwise_conflict_feature_vector
from src.tp_retrieval_utils import build_retrieval_case_text, predict_retrieval_probs as predict_tp_retrieval_probs
from src.utils import main


# ─── Config ───────────────────────────────────────────────────────────────────
PATH_TO_ROOT: Final[Path] = Path(__file__).parent
PATH_TO_CONF: Final[Path] = PATH_TO_ROOT / "app.ini"
PATH_TO_MODEL: Final[Path] = PATH_TO_ROOT / "models" / "joint_bert_v2_final"
PATH_TO_RE_AUX_MODEL: Final[Path] = PATH_TO_ROOT / "models" / "bert_re_512"
PATH_TO_RE_STACKER: Final[Path] = PATH_TO_ROOT / "models" / "re_stacker_v3.json"
PATH_TO_RE_RETRIEVAL_MODEL: Final[Path] = PATH_TO_ROOT / "models" / "re_rationale_retrieval_v1.json"
PATH_TO_TP_FUSER: Final[Path] = PATH_TO_ROOT / "models" / "tp_case_fuser_v1.json"
PATH_TO_TP_RETRIEVAL: Final[Path] = PATH_TO_ROOT / "models" / "tp_retrieval_v1.json"
PATH_TO_TP_PAIRWISE: Final[Path] = PATH_TO_ROOT / "models" / "tp_pairwise_conflict_v1.json"
RE_NLI_MODEL_NAME: Final[str] = DEFAULT_NLI_MODEL

CONFIG: Final[configparser.ConfigParser] = configparser.ConfigParser()
CONFIG.read(PATH_TO_CONF, encoding="utf-8")
TEST_DATA_FILENAME: Final[str] = CONFIG["settings"]["TEST_DATA"]

_PARTY_MARKER: Final[dict[str, str]] = {"P": "[原告]", "D": "[被告]"}
_DEVICE: Final[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def _case_text(tort: Tort, max_chars: int = 2000) -> str:
    uu = " ".join(f.description for f in tort.undisputed_facts)
    pp = " ".join(c.description for c in tort.plaintiff_claims)
    dd = " ".join(c.description for c in tort.defendant_claims)
    return f"[UU] {uu} [PP] {pp} [DD] {dd}"[:max_chars]


class _JointModel(nn.Module):
    def __init__(self, backbone_dir: Path) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(str(backbone_dir))
        hidden = self.bert.config.hidden_size
        self.re_head = nn.Linear(hidden, 1)
        self.tp_head = nn.Linear(hidden, 1)
        self.use_tti = getattr(self.bert.config, "type_vocab_size", 0) > 0

    def _encode(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_tti and token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.bert(**kw)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]
        return pooled

    def forward_claims(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        return self.re_head(self._encode(input_ids, attention_mask, token_type_ids)).squeeze(-1)

    def forward_case(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        return self.tp_head(self._encode(input_ids, attention_mask, token_type_ids)).squeeze(-1)


class _ClaimModel(nn.Module):
    def __init__(self, backbone_dir: Path) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(str(backbone_dir))
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, 1)
        self.use_tti = getattr(self.bert.config, "type_vocab_size", 0) > 0

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_tti and token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.bert(**kw)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(pooled).squeeze(-1)


def _load_joint_model(model_dir: Path):
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "backbone"))
    model = _JointModel(model_dir / "backbone")
    model.re_head.load_state_dict(
        torch.load(str(model_dir / "re_classifier_head.pt"), map_location=_DEVICE)
    )
    model.tp_head.load_state_dict(
        torch.load(str(model_dir / "tp_classifier_head.pt"), map_location=_DEVICE)
    )
    model = model.to(_DEVICE).eval()
    return model, tokenizer, cfg


_MODEL, _TOKENIZER, _CFG = _load_joint_model(PATH_TO_MODEL)
_MAX_LEN: Final[int] = int(_CFG["max_length"])
_RE_THRESHOLD: Final[float] = float(_CFG["re_threshold"])
_TP_THRESHOLD: Final[float] = float(_CFG["tp_threshold"])
_TP_DOMINANCE_THRESHOLD: Final[int] = 2
_TP_REPAIR_MARGIN: Final[float] = 0.0
_BEST_RE_THRESHOLD: Final[float] = 0.34
_BEST_TP_THRESHOLD: Final[float] = 0.49
_BEST_TP_REPAIR_MARGIN: Final[float] = 0.02
_BEST_TP_DOMINANCE_THRESHOLD: Final[int] = 4
_AUX_RE_CACHE: dict[str, tuple[_ClaimModel, Any, dict[str, Any]]] = {}
_RE_STACKER_CACHE: dict[str, dict[str, Any]] = {}
_RE_RETRIEVAL_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
_NLI_RE_CACHE: dict[str, tuple[Any, Any, dict[str, int], int]] = {}
_TP_FUSER_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
_TP_RETRIEVAL_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
_TP_PAIRWISE_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


def _read_torts(path: Path) -> list[Tort]:
    rows: list[Tort] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(Tort.from_dict(json.loads(line)))
    return rows


def _tokenize_single_or_pair(text_a, text_b=None, *, tokenizer=None, max_len=None):
    tokenizer = _TOKENIZER if tokenizer is None else tokenizer
    max_len = _MAX_LEN if max_len is None else int(max_len)
    if text_b is None:
        enc = tokenizer(
            text_a,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            verbose=False,
            return_tensors="pt",
        )
    else:
        enc = tokenizer(
            text_a,
            text_b,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            verbose=False,
            return_tensors="pt",
        )
    return {k: v.to(_DEVICE) for k, v in enc.items()}


def _load_claim_model(model_dir: Path):
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "backbone"))
    model = _ClaimModel(model_dir / "backbone")
    model.classifier.load_state_dict(
        torch.load(str(model_dir / "classifier_head.pt"), map_location=_DEVICE)
    )
    model = model.to(_DEVICE).eval()
    return model, tokenizer, cfg


def _get_aux_re_bundle(model_dir: Path | None = None):
    model_dir = PATH_TO_RE_AUX_MODEL if model_dir is None else Path(model_dir)
    cache_key = str(model_dir)
    if cache_key not in _AUX_RE_CACHE:
        _AUX_RE_CACHE[cache_key] = _load_claim_model(model_dir)
    return _AUX_RE_CACHE[cache_key]


def _get_re_stacker_cfg(path: Path | None = None) -> dict[str, Any]:
    path = PATH_TO_RE_STACKER if path is None else Path(path)
    cache_key = str(path)
    if cache_key not in _RE_STACKER_CACHE:
        _RE_STACKER_CACHE[cache_key] = json.loads(path.read_text(encoding="utf-8"))
    return _RE_STACKER_CACHE[cache_key]


def _get_re_retrieval_bundle(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = PATH_TO_RE_RETRIEVAL_MODEL if path is None else Path(path)
    cache_key = str(path)
    if cache_key not in _RE_RETRIEVAL_CACHE:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        model_path = Path(cfg.get("model_path") or path.with_suffix(".joblib"))
        if not model_path.is_absolute():
            model_path = PATH_TO_ROOT / model_path
        bundle = joblib.load(model_path)
        _RE_RETRIEVAL_CACHE[cache_key] = (cfg, bundle)
    return _RE_RETRIEVAL_CACHE[cache_key]


def _get_nli_re_bundle(model_name: str | None = None):
    model_name = RE_NLI_MODEL_NAME if model_name is None else model_name
    if model_name not in _NLI_RE_CACHE:
        _NLI_RE_CACHE[model_name] = load_nli_bundle(model_name, device=_DEVICE)
    return _NLI_RE_CACHE[model_name]


def _get_tp_fuser_bundle(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = PATH_TO_TP_FUSER if path is None else Path(path)
    cache_key = str(path)
    if cache_key not in _TP_FUSER_CACHE:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        model_path = Path(cfg.get("model_path") or path.with_suffix(".joblib"))
        if not model_path.is_absolute():
            model_path = PATH_TO_ROOT / model_path
        bundle = joblib.load(model_path)
        _TP_FUSER_CACHE[cache_key] = (cfg, bundle)
    return _TP_FUSER_CACHE[cache_key]


def _get_tp_retrieval_bundle(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = PATH_TO_TP_RETRIEVAL if path is None else Path(path)
    cache_key = str(path)
    if cache_key not in _TP_RETRIEVAL_CACHE:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        task_name = str(cfg.get("task") or "")
        if task_name == "tp_retrieval_qwen_embed":
            model_path = Path(cfg.get("model_path") or path.with_suffix(".joblib"))
            if not model_path.is_absolute():
                model_path = PATH_TO_ROOT / model_path
            model_ref_bundle = joblib.load(model_path)
            model_ref = str(model_ref_bundle["model_ref"])
            data_path = Path(cfg.get("data") or "dataset/train001.jsonl")
            if not data_path.is_absolute():
                data_path = PATH_TO_ROOT / data_path
            train_rows = _read_torts(data_path)
            max_chars = int(cfg.get("case_text_max_chars", 2400))
            ref_texts = [
                build_retrieval_case_text(tort, max_chars=max_chars)
                for tort in train_rows
            ]
            ref_labels = np.asarray(
                [1.0 if bool(tort.court_decision) else 0.0 for tort in train_rows],
                dtype=float,
            )
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_ref, device=str(_DEVICE), trust_remote_code=False)
            ref_embeddings = encoder.encode(
                ref_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            lexical_path = Path(cfg.get("tp_retrieval") or PATH_TO_TP_RETRIEVAL)
            if not lexical_path.is_absolute():
                lexical_path = PATH_TO_ROOT / lexical_path
            lexical_cfg, lexical_bundle = _get_tp_retrieval_bundle(lexical_path)
            bundle = {
                "kind": "tp_retrieval_qwen_embed",
                "encoder": encoder,
                "ref_embeddings": np.asarray(ref_embeddings, dtype=float),
                "ref_labels": ref_labels,
                "lexical_cfg": lexical_cfg,
                "lexical_bundle": lexical_bundle,
            }
        else:
            model_path = Path(cfg.get("model_path") or path.with_suffix(".joblib"))
            if not model_path.is_absolute():
                model_path = PATH_TO_ROOT / model_path
            bundle = joblib.load(model_path)
        _TP_RETRIEVAL_CACHE[cache_key] = (cfg, bundle)
    return _TP_RETRIEVAL_CACHE[cache_key]


def _get_tp_pairwise_bundle(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = PATH_TO_TP_PAIRWISE if path is None else Path(path)
    cache_key = str(path)
    if cache_key not in _TP_PAIRWISE_CACHE:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        model_path = Path(cfg.get("model_path") or path.with_suffix(".joblib"))
        if not model_path.is_absolute():
            model_path = PATH_TO_ROOT / model_path
        bundle = joblib.load(model_path)
        _TP_PAIRWISE_CACHE[cache_key] = (cfg, bundle)
    return _TP_PAIRWISE_CACHE[cache_key]


@torch.inference_mode()
def _predict_case_probs(cases: list[Tort], batch_size: int = 32) -> list[float]:
    probs: list[float] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        texts = [_case_text(tort) for tort in batch]
        enc = _tokenize_single_or_pair(texts)
        logits = _MODEL.forward_case(
            enc["input_ids"],
            enc["attention_mask"],
            enc.get("token_type_ids"),
        )
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs


def _predict_retrieval_case_probs(
    cases: list[Tort],
    retrieval_cfg: dict[str, Any],
    retrieval_bundle: dict[str, Any],
    *,
    batch_size: int = 64,
) -> list[float]:
    top_k = int(retrieval_cfg["top_k"])
    max_chars = int(retrieval_cfg.get("vectorizer", {}).get("case_text_max_chars", 2400))
    probs: list[float] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        texts = [
            build_retrieval_case_text(tort, max_chars=max_chars)
            for tort in batch
        ]
        batch_probs = predict_tp_retrieval_probs(texts, retrieval_bundle, top_k=top_k)
        probs.extend(float(value) for value in batch_probs.tolist())
    return probs


def _predict_qwen_dense_case_probs(
    cases: list[Tort],
    retrieval_cfg: dict[str, Any],
    retrieval_bundle: dict[str, Any],
    *,
    batch_size: int = 32,
) -> list[float]:
    top_k = int(retrieval_cfg["top_k"])
    max_chars = int(retrieval_cfg.get("case_text_max_chars", 2400))
    encoder = retrieval_bundle["encoder"]
    ref_embeddings = np.asarray(retrieval_bundle["ref_embeddings"], dtype=float)
    ref_labels = np.asarray(retrieval_bundle["ref_labels"], dtype=float)
    fallback_prob = float(ref_labels.mean()) if len(ref_labels) else 0.5
    probs: list[float] = []

    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        texts = [
            build_retrieval_case_text(tort, max_chars=max_chars)
            for tort in batch
        ]
        query_embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = np.matmul(np.asarray(query_embeddings, dtype=float), ref_embeddings.T)
        k = min(int(top_k), ref_embeddings.shape[0])
        for row in sims:
            row = np.asarray(row, dtype=float).ravel()
            if k == len(row):
                idx = np.arange(len(row))
            else:
                idx = np.argpartition(row, -k)[-k:]
            order = np.argsort(row[idx])[::-1]
            idx = idx[order]
            weights = np.clip(row[idx], 0.0, None)
            if float(weights.sum()) <= 0.0:
                probs.append(fallback_prob)
            else:
                probs.append(float(np.dot(weights, ref_labels[idx]) / weights.sum()))
    return probs


@torch.inference_mode()
def _predict_claim_probs(
    uu_text: str,
    party: str,
    claims: list[PlaintiffClaim] | list[DefendantClaim],
    batch_size: int = 64,
    *,
    model=None,
    tokenizer=None,
    max_len=None,
) -> list[float]:
    if not claims:
        return []

    probs: list[float] = []
    marker = _PARTY_MARKER[party]
    claim_texts = [f"{marker} {claim.description}" for claim in claims]
    uu_batch = [uu_text] * len(claim_texts)
    model = _MODEL if model is None else model

    for start in range(0, len(claim_texts), batch_size):
        text_a = uu_batch[start:start + batch_size]
        text_b = claim_texts[start:start + batch_size]
        enc = _tokenize_single_or_pair(text_a, text_b, tokenizer=tokenizer, max_len=max_len)
        if isinstance(model, _JointModel):
            logits = model.forward_claims(
                enc["input_ids"],
                enc["attention_mask"],
                enc.get("token_type_ids"),
            )
        else:
            logits = model(
                enc["input_ids"],
                enc["attention_mask"],
                enc.get("token_type_ids"),
            )
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())

    return probs


@torch.inference_mode()
def _predict_nli_claim_probs(
    tort: Tort,
    party: str,
    claims: list[PlaintiffClaim] | list[DefendantClaim],
    *,
    model_name: str | None = None,
    batch_size: int = 64,
) -> list[float]:
    if not claims:
        return []

    model, tokenizer, label_roles, max_len = _get_nli_re_bundle(model_name)
    case_view = {
        "undisputed_facts": [{"description": item.description} for item in tort.undisputed_facts],
        "plaintiff_claims": [{"description": item.description} for item in tort.plaintiff_claims],
        "defendant_claims": [{"description": item.description} for item in tort.defendant_claims],
    }
    premise = build_claim_premise(
        case_view,
        party,
        mode=DEFAULT_PREMISE_MODE,
        max_chars=1600,
    )
    premises = [premise] * len(claims)
    hypotheses = [
        claim_hypothesis({"description": claim.description}, party)
        for claim in claims
    ]
    prob_triplets, logit_triplets = predict_nli_triplets(
        model,
        tokenizer,
        premises,
        hypotheses,
        device=_DEVICE,
        max_len=max_len,
        batch_size=batch_size,
    )
    return [
        score_from_nli(
            prob_triplets[idx],
            logit_triplets[idx],
            label_roles,
            mode=DEFAULT_SCORE_MODE,
        )
        for idx in range(len(prob_triplets))
    ]


def _stack_re_probs(source_prob_lists: list[list[float]], stacker_cfg: dict[str, Any]) -> list[float]:
    coeffs = [float(x) for x in stacker_cfg["coefficients"]]
    if len(source_prob_lists) != len(coeffs):
        raise ValueError(
            f"RE stacker expected {len(coeffs)} sources, got {len(source_prob_lists)}."
        )
    if not source_prob_lists:
        return []
    claim_count = len(source_prob_lists[0])
    for probs in source_prob_lists[1:]:
        if len(probs) != claim_count:
            raise ValueError("RE stacker sources must produce the same number of claim probabilities.")
    intercept = float(stacker_cfg["intercept"])
    combined: list[float] = []
    for claim_idx in range(claim_count):
        logit = intercept
        for coeff, probs in zip(coeffs, source_prob_lists):
            logit += float(coeff) * float(probs[claim_idx])
        combined.append(1.0 / (1.0 + math.exp(-logit)))
    return combined


def _looks_like_nli_source(name: str) -> bool:
    lower = name.lower()
    return ("nli" in lower) or ("entail" in lower) or ("akifqc" in lower)


def _looks_like_joint_source(name: str) -> bool:
    return "joint" in name.lower()


def _looks_like_retrieval_source(name: str) -> bool:
    return "retrieval" in name.lower()


def _stack_named_re_sources(
    source_names: list[str],
    stacker_cfg: dict[str, Any],
    *,
    joint_probs: list[float],
    aux_probs: list[float] | None = None,
    nli_probs: list[float] | None = None,
    retrieval_probs: list[float] | None = None,
) -> list[float]:
    source_prob_lists: list[list[float]] = []
    for source_name in source_names:
        if _looks_like_joint_source(source_name):
            source_prob_lists.append(joint_probs)
        elif _looks_like_nli_source(source_name):
            if nli_probs is None:
                raise ValueError(f"RE stacker expects NLI source '{source_name}', but no NLI probabilities were provided.")
            source_prob_lists.append(nli_probs)
        elif _looks_like_retrieval_source(source_name):
            if retrieval_probs is None:
                raise ValueError(f"RE stacker expects retrieval source '{source_name}', but no retrieval probabilities were provided.")
            source_prob_lists.append(retrieval_probs)
        else:
            if aux_probs is None:
                raise ValueError(f"RE stacker expects auxiliary RE source '{source_name}', but no auxiliary probabilities were provided.")
            source_prob_lists.append(aux_probs)
    return _stack_re_probs(source_prob_lists, stacker_cfg)


def _predict_re_retrieval_claim_probs(
    tort: Tort,
    party: str,
    claims: list[PlaintiffClaim] | list[DefendantClaim],
    retrieval_cfg: dict[str, Any],
    retrieval_bundle: dict[str, Any],
) -> list[float]:
    texts_by_party = {
        party: [
            build_claim_retrieval_text(
                tort,
                party,
                idx,
                facts_max_chars=int(retrieval_cfg["vectorizer"]["facts_max_chars"]),
                opp_max_chars=int(retrieval_cfg["vectorizer"]["opp_max_chars"]),
            )
            for idx in range(len(claims))
        ]
    }
    pred = predict_re_rationale_probs(
        texts_by_party,
        retrieval_bundle,
        top_k=int(retrieval_cfg["top_k"]),
    )
    return pred[party].astype(float).tolist()


def _accepted_and_rejected(claim_probs: list[float], re_threshold: float) -> tuple[int, int]:
    accepted = sum(prob >= re_threshold for prob in claim_probs)
    return accepted, len(claim_probs) - accepted


def _dominance_score(pp_probs: list[float], dd_probs: list[float], re_threshold: float) -> int:
    """
    Positive means plaintiff-side rationale pattern is stronger.
    Negative means defendant-side rationale pattern is stronger.

    Score is based on:
      (accepted_plaintiff - rejected_plaintiff) - (accepted_defendant - rejected_defendant)
    """
    p_acc, p_rej = _accepted_and_rejected(pp_probs, re_threshold)
    d_acc, d_rej = _accepted_and_rejected(dd_probs, re_threshold)
    return (p_acc - p_rej) - (d_acc - d_rej)


def _repair_tp_label(
    case_prob: float,
    base_tp: bool,
    pp_probs: list[float],
    dd_probs: list[float],
    *,
    re_threshold: float,
    tp_threshold: float,
    repair_margin: float,
    dominance_threshold: int,
) -> tuple[bool, dict[str, Any]]:
    meta = {
        "tp_before": base_tp,
        "tp_after": base_tp,
        "repair_applied": False,
        "repair_reason": "",
        "repair_margin": repair_margin,
        "dominance_threshold": dominance_threshold,
    }
    if repair_margin <= 0:
        return base_tp, meta

    distance = abs(case_prob - tp_threshold)
    p_acc, p_rej = _accepted_and_rejected(pp_probs, re_threshold)
    d_acc, d_rej = _accepted_and_rejected(dd_probs, re_threshold)
    dominance = (p_acc - p_rej) - (d_acc - d_rej)

    meta.update({
        "distance_to_tp_threshold": distance,
        "plaintiff_accepted": p_acc,
        "plaintiff_rejected": p_rej,
        "defendant_accepted": d_acc,
        "defendant_rejected": d_rej,
        "dominance_score": dominance,
    })

    if distance > repair_margin:
        return base_tp, meta

    if (not base_tp) and dominance >= dominance_threshold:
        meta["tp_after"] = True
        meta["repair_applied"] = True
        meta["repair_reason"] = "flip_true:plaintiff_dominance"
        return True, meta

    if base_tp and dominance <= -dominance_threshold:
        meta["tp_after"] = False
        meta["repair_applied"] = True
        meta["repair_reason"] = "flip_false:defendant_dominance"
        return False, meta

    return base_tp, meta


def solve_with_details(
    test_data: list[Tort],
    *,
    re_threshold: float | None = None,
    tp_threshold: float | None = None,
    repair_margin: float = _TP_REPAIR_MARGIN,
    dominance_threshold: int = _TP_DOMINANCE_THRESHOLD,
    use_re_stacker: bool = False,
    re_stacker_path: str | None = None,
    re_aux_model_dir: str | None = None,
    re_nli_model_name: str | None = None,
    re_retrieval_model_path: str | None = None,
    use_tp_fuser: bool = False,
    tp_fuser_path: str | None = None,
    use_tp_retrieval: bool = False,
    tp_retrieval_path: str | None = None,
    tp_retrieval_alpha: float | None = None,
    use_tp_pairwise: bool = False,
    tp_pairwise_path: str | None = None,
) -> tuple[list[Tort], list[dict[str, Any]]]:
    stacker_cfg: dict[str, Any] | None = None
    aux_bundle = None
    tp_fuser_cfg: dict[str, Any] | None = None
    tp_fuser_bundle: dict[str, Any] | None = None
    tp_retrieval_cfg: dict[str, Any] | None = None
    tp_retrieval_bundle: dict[str, Any] | None = None
    tp_pairwise_cfg: dict[str, Any] | None = None
    tp_pairwise_bundle: dict[str, Any] | None = None
    re_retrieval_cfg: dict[str, Any] | None = None
    re_retrieval_bundle: dict[str, Any] | None = None
    if use_re_stacker:
        stacker_cfg = _get_re_stacker_cfg(None if re_stacker_path is None else Path(re_stacker_path))
        aux_bundle = _get_aux_re_bundle(None if re_aux_model_dir is None else Path(re_aux_model_dir))
        re_retrieval_cfg, re_retrieval_bundle = _get_re_retrieval_bundle(
            None if re_retrieval_model_path is None else Path(re_retrieval_model_path)
        )
        if re_threshold is None:
            re_threshold = float(stacker_cfg["threshold"])
    if use_tp_fuser:
        tp_fuser_cfg, tp_fuser_bundle = _get_tp_fuser_bundle(
            None if tp_fuser_path is None else Path(tp_fuser_path)
        )
    if use_tp_retrieval:
        tp_retrieval_cfg, tp_retrieval_bundle = _get_tp_retrieval_bundle(
            None if tp_retrieval_path is None else Path(tp_retrieval_path)
        )
    if use_tp_pairwise:
        tp_pairwise_cfg, tp_pairwise_bundle = _get_tp_pairwise_bundle(
            None if tp_pairwise_path is None else Path(tp_pairwise_path)
        )
    re_threshold = _RE_THRESHOLD if re_threshold is None else float(re_threshold)
    if tp_threshold is None:
        if use_tp_pairwise and tp_pairwise_cfg is not None:
            tp_threshold = float(tp_pairwise_cfg["threshold"])
        elif use_tp_retrieval and tp_retrieval_cfg is not None:
            tp_threshold = float(tp_retrieval_cfg["threshold"])
        elif use_tp_fuser and tp_fuser_cfg is not None:
            tp_threshold = float(tp_fuser_cfg["threshold"])
        else:
            tp_threshold = _TP_THRESHOLD
    else:
        tp_threshold = float(tp_threshold)

    case_probs = _predict_case_probs(test_data)
    retrieval_case_probs: list[float] = []
    retrieval_case_probs_lexical: list[float] = []
    if use_tp_retrieval and tp_retrieval_cfg is not None and tp_retrieval_bundle is not None:
        tp_retrieval_task = str(tp_retrieval_cfg.get("task") or "")
        if tp_retrieval_task == "tp_retrieval_qwen_embed":
            retrieval_case_probs = _predict_qwen_dense_case_probs(
                test_data,
                tp_retrieval_cfg,
                tp_retrieval_bundle,
            )
            retrieval_case_probs_lexical = _predict_retrieval_case_probs(
                test_data,
                tp_retrieval_bundle["lexical_cfg"],
                tp_retrieval_bundle["lexical_bundle"],
            )
        else:
            retrieval_case_probs = _predict_retrieval_case_probs(
                test_data,
                tp_retrieval_cfg,
                tp_retrieval_bundle,
            )
    system_results: list[Tort] = []
    debug_rows: list[dict[str, Any]] = []

    for idx, tort in enumerate(test_data):
        uu_text = " ".join(f.description for f in tort.undisputed_facts)

        pp_joint_probs = _predict_claim_probs(uu_text, "P", tort.plaintiff_claims)
        dd_joint_probs = _predict_claim_probs(uu_text, "D", tort.defendant_claims)
        pp_probs = list(pp_joint_probs)
        dd_probs = list(dd_joint_probs)
        pp_aux_probs: list[float] = []
        dd_aux_probs: list[float] = []
        pp_nli_probs: list[float] = []
        dd_nli_probs: list[float] = []
        pp_retrieval_probs: list[float] = []
        dd_retrieval_probs: list[float] = []
        if use_re_stacker and stacker_cfg is not None and aux_bundle is not None:
            source_names = [str(name) for name in stacker_cfg.get("source_names", [])]
            needs_aux = any(
                (not _looks_like_joint_source(name))
                and (not _looks_like_nli_source(name))
                and (not _looks_like_retrieval_source(name))
                for name in source_names
            )
            needs_nli = any(_looks_like_nli_source(name) for name in source_names)
            needs_retrieval = any(_looks_like_retrieval_source(name) for name in source_names)

            if needs_aux:
                aux_model, aux_tokenizer, aux_cfg = aux_bundle
                aux_max_len = int(aux_cfg["max_length"])
                pp_aux_probs = _predict_claim_probs(
                    uu_text, "P", tort.plaintiff_claims,
                    model=aux_model, tokenizer=aux_tokenizer, max_len=aux_max_len,
                )
                dd_aux_probs = _predict_claim_probs(
                    uu_text, "D", tort.defendant_claims,
                    model=aux_model, tokenizer=aux_tokenizer, max_len=aux_max_len,
                )

            if needs_nli:
                pp_nli_probs = _predict_nli_claim_probs(
                    tort, "P", tort.plaintiff_claims, model_name=re_nli_model_name,
                )
                dd_nli_probs = _predict_nli_claim_probs(
                    tort, "D", tort.defendant_claims, model_name=re_nli_model_name,
                )

            if needs_retrieval:
                if re_retrieval_cfg is None or re_retrieval_bundle is None:
                    raise ValueError("RE stacker expects a retrieval source, but no retrieval model was loaded.")
                pp_retrieval_probs = _predict_re_retrieval_claim_probs(
                    tort,
                    "P",
                    tort.plaintiff_claims,
                    re_retrieval_cfg,
                    re_retrieval_bundle,
                )
                dd_retrieval_probs = _predict_re_retrieval_claim_probs(
                    tort,
                    "D",
                    tort.defendant_claims,
                    re_retrieval_cfg,
                    re_retrieval_bundle,
                )

            pp_probs = _stack_named_re_sources(
                source_names,
                stacker_cfg,
                joint_probs=pp_joint_probs,
                aux_probs=pp_aux_probs if needs_aux else None,
                nli_probs=pp_nli_probs if needs_nli else None,
                retrieval_probs=pp_retrieval_probs if needs_retrieval else None,
            )
            dd_probs = _stack_named_re_sources(
                source_names,
                stacker_cfg,
                joint_probs=dd_joint_probs,
                aux_probs=dd_aux_probs if needs_aux else None,
                nli_probs=dd_nli_probs if needs_nli else None,
                retrieval_probs=dd_retrieval_probs if needs_retrieval else None,
            )
        case_prob_joint = float(case_probs[idx])
        case_prob_effective = case_prob_joint
        tp_fuser_prob: float | None = None
        retrieval_prob: float | None = None
        retrieval_prob_lexical: float | None = None
        retrieval_alpha_effective: float | None = None
        tp_pairwise_prob: float | None = None
        if use_tp_fuser and tp_fuser_cfg is not None and tp_fuser_bundle is not None:
            feature_vector = build_tp_case_feature_vector(
                case_prob_joint,
                pp_probs,
                dd_probs,
                tp_anchor=float(tp_fuser_cfg.get("tp_anchor", 0.69)),
            )
            if tp_fuser_bundle["kind"] == "logreg":
                X_use = tp_fuser_bundle["scaler"].transform([feature_vector])
            else:
                X_use = [feature_vector]
            tp_fuser_prob = float(tp_fuser_bundle["model"].predict_proba(X_use)[0][1])
            case_prob_effective = tp_fuser_prob

        if use_tp_retrieval and tp_retrieval_cfg is not None and tp_retrieval_bundle is not None:
            retrieval_prob = float(retrieval_case_probs[idx])
            if str(tp_retrieval_cfg.get("task") or "") == "tp_retrieval_qwen_embed":
                retrieval_prob_lexical = float(retrieval_case_probs_lexical[idx])
                lexical_cfg = tp_retrieval_bundle["lexical_cfg"]
                lexical_alpha = float(lexical_cfg["alpha"])
                lexical_fused_prob = (
                    ((1.0 - lexical_alpha) * case_prob_effective)
                    + (lexical_alpha * retrieval_prob_lexical)
                )
                if tp_retrieval_alpha is None:
                    retrieval_alpha_effective = float(tp_retrieval_cfg["alpha"])
                else:
                    retrieval_alpha_effective = float(tp_retrieval_alpha)
                case_prob_effective = (
                    ((1.0 - retrieval_alpha_effective) * lexical_fused_prob)
                    + (retrieval_alpha_effective * retrieval_prob)
                )
            else:
                if tp_retrieval_alpha is None:
                    retrieval_alpha_effective = float(tp_retrieval_cfg["alpha"])
                else:
                    retrieval_alpha_effective = float(tp_retrieval_alpha)
                case_prob_effective = (
                    ((1.0 - retrieval_alpha_effective) * case_prob_effective)
                    + (retrieval_alpha_effective * retrieval_prob)
                )

        if use_tp_pairwise and tp_pairwise_cfg is not None and tp_pairwise_bundle is not None:
            pairwise_vector = build_pairwise_conflict_feature_vector(
                case_prob_effective,
                uu_text,
                [claim.description for claim in tort.plaintiff_claims],
                [claim.description for claim in tort.defendant_claims],
                pp_probs,
                dd_probs,
                tp_anchor=float(tp_pairwise_cfg.get("tp_anchor", 0.74)),
            )
            if tp_pairwise_bundle["kind"] == "logreg":
                X_use = tp_pairwise_bundle["scaler"].transform([pairwise_vector])
            else:
                X_use = [pairwise_vector]
            tp_pairwise_prob = float(tp_pairwise_bundle["model"].predict_proba(X_use)[0][1])
            case_prob_effective = tp_pairwise_prob

        base_tp = bool(case_prob_effective >= tp_threshold)
        final_tp, repair_meta = _repair_tp_label(
            case_prob_effective,
            base_tp,
            pp_probs,
            dd_probs,
            re_threshold=re_threshold,
            tp_threshold=tp_threshold,
            repair_margin=repair_margin,
            dominance_threshold=dominance_threshold,
        )

        plaintiff_claims = [
            PlaintiffClaim(
                id=claim.id,
                description=claim.description,
                is_accepted=bool(prob >= re_threshold),
            )
            for claim, prob in zip(tort.plaintiff_claims, pp_probs)
        ]
        defendant_claims = [
            DefendantClaim(
                id=claim.id,
                description=claim.description,
                is_accepted=bool(prob >= re_threshold),
            )
            for claim, prob in zip(tort.defendant_claims, dd_probs)
        ]

        result = Tort(
            version=tort.version,
            tort_id=tort.tort_id,
            undisputed_facts=tort.undisputed_facts,
            plaintiff_claims=plaintiff_claims,
            defendant_claims=defendant_claims,
            court_decision=final_tp,
        )
        system_results.append(result)

        debug_rows.append({
            "tort_id": tort.tort_id,
            "case_prob": float(case_prob_effective),
            "case_prob_joint": case_prob_joint,
            "tp_fuser_used": bool(use_tp_fuser),
            "tp_fuser_prob": tp_fuser_prob,
            "tp_retrieval_used": bool(use_tp_retrieval),
            "tp_retrieval_prob": retrieval_prob,
            "tp_retrieval_lexical_prob": retrieval_prob_lexical,
            "tp_retrieval_alpha": retrieval_alpha_effective,
            "tp_pairwise_used": bool(use_tp_pairwise),
            "tp_pairwise_prob": tp_pairwise_prob,
            "tp_threshold": tp_threshold,
            "re_threshold": re_threshold,
            "tp_before": base_tp,
            "tp_after": final_tp,
            "plaintiff_claim_probs": [float(prob) for prob in pp_probs],
            "defendant_claim_probs": [float(prob) for prob in dd_probs],
            "re_stacker_used": bool(use_re_stacker),
            "re_stacker_threshold": float(re_threshold) if use_re_stacker else None,
            "plaintiff_claim_probs_joint": [float(prob) for prob in pp_joint_probs],
            "defendant_claim_probs_joint": [float(prob) for prob in dd_joint_probs],
            "plaintiff_claim_probs_aux": [float(prob) for prob in pp_aux_probs],
            "defendant_claim_probs_aux": [float(prob) for prob in dd_aux_probs],
            "plaintiff_claim_probs_nli": [float(prob) for prob in pp_nli_probs],
            "defendant_claim_probs_nli": [float(prob) for prob in dd_nli_probs],
            "plaintiff_claim_probs_retrieval": [float(prob) for prob in pp_retrieval_probs],
            "defendant_claim_probs_retrieval": [float(prob) for prob in dd_retrieval_probs],
            **repair_meta,
        })

    return system_results, debug_rows


def solve(test_data: list[Tort]) -> list[Tort]:
    results, _ = solve_with_details(
        test_data,
        re_threshold=_BEST_RE_THRESHOLD,
        tp_threshold=_BEST_TP_THRESHOLD,
        repair_margin=_BEST_TP_REPAIR_MARGIN,
        dominance_threshold=_BEST_TP_DOMINANCE_THRESHOLD,
        use_re_stacker=True,
        use_tp_fuser=True,
        use_tp_retrieval=True,
    )
    return results


if __name__ == "__main__":
    main(solve)
