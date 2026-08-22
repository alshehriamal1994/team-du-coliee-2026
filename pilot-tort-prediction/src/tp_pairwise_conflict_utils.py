from __future__ import annotations

from functools import lru_cache


FEATURE_NAMES = [
    "base_tp_prob",
    "abs_from_anchor",
    "plaintiff_prob_sum",
    "defendant_prob_sum",
    "plaintiff_fact_support_sum",
    "defendant_fact_support_sum",
    "plaintiff_survival_sum",
    "defendant_survival_sum",
    "plaintiff_rebuttal_sum",
    "defendant_rebuttal_sum",
    "survival_margin",
    "net_argument_margin",
    "plaintiff_max_cross_sim_mean",
    "defendant_max_cross_sim_mean",
    "plaintiff_max_cross_sim_max",
    "defendant_max_cross_sim_max",
    "conflict_mass",
    "unmatched_plaintiff_mass",
    "unmatched_defendant_mass",
    "plaintiff_count",
    "defendant_count",
]

DEFAULT_TP_ANCHOR = 0.74


@lru_cache(maxsize=200_000)
def _char_ngram_set(text: str, min_n: int = 2, max_n: int = 4, max_chars: int = 512) -> frozenset[str]:
    text = (text or "").strip()
    if not text:
        return frozenset()
    text = text[:max_chars]
    grams: set[str] = set()
    text_len = len(text)
    for n in range(min_n, max_n + 1):
        if text_len < n:
            continue
        for idx in range(text_len - n + 1):
            grams.add(text[idx:idx + n])
    if not grams:
        grams.add(text)
    return frozenset(grams)


def _jaccard_from_sets(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    if inter == 0:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return float(inter / union)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _best_opposition(
    source_texts: list[str],
    source_probs: list[float],
    target_texts: list[str],
    target_probs: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    source_sets = [_char_ngram_set(text) for text in source_texts]
    target_sets = [_char_ngram_set(text) for text in target_texts]

    max_sims: list[float] = []
    matched_target_probs: list[float] = []
    rebuttals: list[float] = []
    survivals: list[float] = []

    for src_prob, src_set in zip(source_probs, source_sets):
        best_sim = 0.0
        best_target_prob = 0.0
        for tgt_prob, tgt_set in zip(target_probs, target_sets):
            sim = _jaccard_from_sets(src_set, tgt_set)
            if sim > best_sim:
                best_sim = sim
                best_target_prob = float(tgt_prob)
        rebuttal = float(src_prob) * best_sim * best_target_prob
        survival = float(src_prob) * (1.0 - (best_sim * best_target_prob))
        max_sims.append(best_sim)
        matched_target_probs.append(best_target_prob)
        rebuttals.append(rebuttal)
        survivals.append(survival)

    return max_sims, matched_target_probs, rebuttals, survivals


def _fact_support(claim_texts: list[str], claim_probs: list[float], facts_text: str) -> float:
    fact_set = _char_ngram_set(facts_text)
    if not claim_texts or not fact_set:
        return 0.0
    total = 0.0
    for claim_text, claim_prob in zip(claim_texts, claim_probs):
        total += float(claim_prob) * _jaccard_from_sets(_char_ngram_set(claim_text), fact_set)
    return float(total)


def build_pairwise_conflict_feature_vector(
    base_tp_prob: float,
    facts_text: str,
    plaintiff_claim_texts: list[str],
    defendant_claim_texts: list[str],
    plaintiff_probs: list[float],
    defendant_probs: list[float],
    *,
    tp_anchor: float = DEFAULT_TP_ANCHOR,
) -> list[float]:
    pp_probs = [float(value) for value in plaintiff_probs]
    dd_probs = [float(value) for value in defendant_probs]
    pp_texts = list(plaintiff_claim_texts)
    dd_texts = list(defendant_claim_texts)

    p_max_sims, p_match_probs, p_rebuttals, p_survivals = _best_opposition(
        pp_texts,
        pp_probs,
        dd_texts,
        dd_probs,
    )
    d_max_sims, d_match_probs, d_rebuttals, d_survivals = _best_opposition(
        dd_texts,
        dd_probs,
        pp_texts,
        pp_probs,
    )

    p_prob_sum = float(sum(pp_probs))
    d_prob_sum = float(sum(dd_probs))
    p_fact_support = _fact_support(pp_texts, pp_probs, facts_text)
    d_fact_support = _fact_support(dd_texts, dd_probs, facts_text)
    p_survival_sum = float(sum(p_survivals))
    d_survival_sum = float(sum(d_survivals))
    p_rebuttal_sum = float(sum(p_rebuttals))
    d_rebuttal_sum = float(sum(d_rebuttals))
    conflict_mass = float(sum(prob * sim for prob, sim in zip(pp_probs, p_max_sims))) + float(
        sum(prob * sim for prob, sim in zip(dd_probs, d_max_sims))
    )
    unmatched_plaintiff_mass = float(sum(prob * (1.0 - sim) for prob, sim in zip(pp_probs, p_max_sims)))
    unmatched_defendant_mass = float(sum(prob * (1.0 - sim) for prob, sim in zip(dd_probs, d_max_sims)))
    survival_margin = p_survival_sum - d_survival_sum
    net_argument_margin = (p_survival_sum - p_rebuttal_sum) - (d_survival_sum - d_rebuttal_sum)

    return [
        float(base_tp_prob),
        float(abs(float(base_tp_prob) - float(tp_anchor))),
        p_prob_sum,
        d_prob_sum,
        p_fact_support,
        d_fact_support,
        p_survival_sum,
        d_survival_sum,
        p_rebuttal_sum,
        d_rebuttal_sum,
        survival_margin,
        net_argument_margin,
        _safe_mean(p_max_sims),
        _safe_mean(d_max_sims),
        float(max(p_max_sims, default=0.0)),
        float(max(d_max_sims, default=0.0)),
        conflict_mass,
        unmatched_plaintiff_mass,
        unmatched_defendant_mass,
        float(len(pp_texts)),
        float(len(dd_texts)),
    ]
