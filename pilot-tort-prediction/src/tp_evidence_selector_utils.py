from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_FACTS_MAX_CHARS = 900
DEFAULT_SIDE_MAX_CHARS = 700
DEFAULT_TOP_N = 3


def build_fact_evidence_text(
    tort: Any,
    fact_idx: int,
    *,
    side_max_chars: int = DEFAULT_SIDE_MAX_CHARS,
) -> str:
    fact = tort.undisputed_facts[fact_idx]
    plaintiff = " ".join(item.description for item in tort.plaintiff_claims)[:side_max_chars]
    defendant = " ".join(item.description for item in tort.defendant_claims)[:side_max_chars]
    return (
        f"[争いのない事実] {fact.description} "
        f"[原告主張要約] {plaintiff} "
        f"[被告反論要約] {defendant}"
    )


def build_claim_evidence_text(
    tort: Any,
    party: str,
    claim_idx: int,
    *,
    facts_max_chars: int = DEFAULT_FACTS_MAX_CHARS,
    side_max_chars: int = DEFAULT_SIDE_MAX_CHARS,
) -> str:
    facts = " ".join(item.description for item in tort.undisputed_facts)[:facts_max_chars]
    if party == "P":
        claim = tort.plaintiff_claims[claim_idx]
        supporting = " ".join(item.description for item in tort.plaintiff_claims)[:side_max_chars]
        opposing = " ".join(item.description for item in tort.defendant_claims)[:side_max_chars]
        return (
            f"[争いのない事実] {facts} "
            f"[原告主張] {claim.description} "
            f"[原告主張要約] {supporting} "
            f"[被告反論要約] {opposing}"
        )
    claim = tort.defendant_claims[claim_idx]
    supporting = " ".join(item.description for item in tort.defendant_claims)[:side_max_chars]
    opposing = " ".join(item.description for item in tort.plaintiff_claims)[:side_max_chars]
    return (
        f"[争いのない事実] {facts} "
        f"[被告反論] {claim.description} "
        f"[被告反論要約] {supporting} "
        f"[原告主張要約] {opposing}"
    )


def _pad_top(values: list[float], top_n: int) -> list[float]:
    ordered = sorted((float(v) for v in values), reverse=True)
    padded = ordered[:top_n]
    if len(padded) < top_n:
        padded.extend([0.0] * (top_n - len(padded)))
    return padded


def summarise_side(values: list[float], *, top_n: int) -> dict[str, float]:
    if not values:
        padded = [0.0] * top_n
        return {
            **{f"top{i + 1}": padded[i] for i in range(top_n)},
            "mean_top": 0.0,
            "sum_top": 0.0,
            "count_strong_05": 0.0,
            "count_strong_07": 0.0,
            "n_items": 0.0,
        }
    padded = _pad_top(values, top_n)
    arr = np.asarray(values, dtype=float)
    return {
        **{f"top{i + 1}": padded[i] for i in range(top_n)},
        "mean_top": float(np.mean(padded)) if padded else 0.0,
        "sum_top": float(np.sum(padded)) if padded else 0.0,
        "count_strong_05": float(np.sum(arr >= 0.50)),
        "count_strong_07": float(np.sum(arr >= 0.70)),
        "n_items": float(len(values)),
    }


def build_evidence_feature_vector(
    *,
    base_prob: float,
    retrieval_prob: float,
    plaintiff_values: list[float],
    defendant_values: list[float],
    fact_positive_values: list[float],
    fact_negative_values: list[float],
    top_n: int = DEFAULT_TOP_N,
) -> tuple[list[float], list[str]]:
    top_n = max(3, int(top_n))
    p_stats = summarise_side(plaintiff_values, top_n=top_n)
    d_stats = summarise_side(defendant_values, top_n=top_n)
    f_pos = summarise_side(fact_positive_values, top_n=min(2, top_n))
    f_neg = summarise_side(fact_negative_values, top_n=min(2, top_n))

    feature_names = [
        "base_prob",
        "retrieval_prob",
        "p_top1",
        "p_top2",
        "p_top3",
        "d_top1",
        "d_top2",
        "d_top3",
        "p_mean_top",
        "d_mean_top",
        "p_sum_top",
        "d_sum_top",
        "p_count_strong_05",
        "d_count_strong_05",
        "p_count_strong_07",
        "d_count_strong_07",
        "fact_pos_top1",
        "fact_neg_top1",
        "fact_pos_mean",
        "fact_neg_mean",
        "margin_top1",
        "margin_mean",
        "margin_sum",
        "fact_margin_top1",
        "fact_margin_mean",
        "n_plaintiff_units",
        "n_defendant_units",
        "n_fact_units",
        "abs_base_minus_retrieval",
    ]
    values = [
        float(base_prob),
        float(retrieval_prob),
        p_stats["top1"],
        p_stats["top2"],
        p_stats["top3"],
        d_stats["top1"],
        d_stats["top2"],
        d_stats["top3"],
        p_stats["mean_top"],
        d_stats["mean_top"],
        p_stats["sum_top"],
        d_stats["sum_top"],
        p_stats["count_strong_05"],
        d_stats["count_strong_05"],
        p_stats["count_strong_07"],
        d_stats["count_strong_07"],
        f_pos["top1"],
        f_neg["top1"],
        f_pos["mean_top"],
        f_neg["mean_top"],
        p_stats["top1"] - d_stats["top1"],
        p_stats["mean_top"] - d_stats["mean_top"],
        p_stats["sum_top"] - d_stats["sum_top"],
        f_pos["top1"] - f_neg["top1"],
        f_pos["mean_top"] - f_neg["mean_top"],
        p_stats["n_items"],
        d_stats["n_items"],
        float(len(fact_positive_values)),
        abs(float(base_prob) - float(retrieval_prob)),
    ]
    return values, feature_names
