from __future__ import annotations

from typing import Final


FEATURE_NAMES: Final[list[str]] = [
    "tp_prob",
    "re_mean_diff",
    "re_margin_diff",
    "re_top1_diff",
    "re_top2_diff",
    "re_top3_diff",
    "re_count_diff_03",
    "re_count_diff_05",
    "claim_count_diff",
    "total_claims",
    "abs_re_margin_diff",
    "abs_tp_minus_anchor",
]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.5


def _topk_sum(values: list[float], k: int) -> float:
    return float(sum(values[:k])) if values else 0.0


def _count_over(values: list[float], threshold: float) -> int:
    return sum(value >= threshold for value in values)


def _signed_margin(values: list[float]) -> float:
    return float(sum((2.0 * value) - 1.0 for value in values))


def build_tp_case_feature_vector(
    tp_prob: float,
    plaintiff_claim_probs: list[float],
    defendant_claim_probs: list[float],
    *,
    tp_anchor: float = 0.69,
) -> list[float]:
    pp = sorted((float(value) for value in plaintiff_claim_probs), reverse=True)
    dd = sorted((float(value) for value in defendant_claim_probs), reverse=True)

    re_mean_diff = _mean(pp) - _mean(dd)
    re_margin_diff = _signed_margin(pp) - _signed_margin(dd)

    return [
        float(tp_prob),
        float(re_mean_diff),
        float(re_margin_diff),
        float(_topk_sum(pp, 1) - _topk_sum(dd, 1)),
        float(_topk_sum(pp, 2) - _topk_sum(dd, 2)),
        float(_topk_sum(pp, 3) - _topk_sum(dd, 3)),
        float(_count_over(pp, 0.30) - _count_over(dd, 0.30)),
        float(_count_over(pp, 0.50) - _count_over(dd, 0.50)),
        float(len(pp) - len(dd)),
        float(len(pp) + len(dd)),
        float(abs(re_margin_diff)),
        float(abs(float(tp_prob) - float(tp_anchor))),
    ]
